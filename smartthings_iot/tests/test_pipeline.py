"""오프라인 테스트: 파서/저장소/통계 모듈을 합성 픽스처로 검증 (실제 수집 데이터 아님)."""
import json
import random
import sys
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis.preprocess import Tokenizer, build_docs, tag_products  # noqa: E402
from analysis.product_keywords import bh_fdr, log_odds_dirichlet, product_keywords  # noqa: E402
from collect import naver, youtube  # noqa: E402
from collect.common import Comment, JsonlStore, clean_text, is_relevant  # noqa: E402


def test_clean_and_relevance():
    assert clean_text("<b>스마트싱스</b> &amp; 허브") == "스마트싱스 & 허브"
    assert is_relevant("SmartThings 자동화", ["smartthings"])
    assert not is_relevant("LG ThinQ", ["스마트싱스", "smartthings"])


def test_parse_blog_url():
    assert naver.parse_blog_url("https://blog.naver.com/abc/223456") == ("abc", "223456")
    assert naver.parse_blog_url("https://blog.naver.com/PostView.naver?blogId=x&logNo=1") == ("x", "1")
    assert naver.parse_blog_url("https://cafe.naver.com/foo/1") is None


def test_item_to_post_blog():
    item = {"title": "<b>스마트싱스</b> 후기", "link": "https://blog.naver.com/abc/1",
            "description": "에어컨 자동화", "bloggername": "nick", "postdate": "20260101"}
    p = naver.item_to_post("blog", item, "스마트싱스")
    assert p.post_id == "naver_blog:abc_1" and p.created_at == "2026-01-01"
    assert p.author_hash and p.author_hash != "nick"


def test_web_item_to_api():
    it = {"domainIdOrBlogId": "abc", "logNo": 1, "postUrl": "https://blog.naver.com/abc/1",
          "title": "<strong>스마트싱스</strong> 루틴", "contents": "조명 자동화",
          "addDate": 1767193200000}   # 2025-12-31T15:00Z = 2026-01-01 00:00 KST
    p = naver.item_to_post("blog", naver.web_item_to_api(it), "스마트싱스")
    assert p.post_id == "naver_blog:abc_1" and p.created_at == "2026-01-01"
    assert p.title == "스마트싱스 루틴"


def test_parse_blog_page():
    html_ = "<script>var blogNo = '123';</script><div class='se-main-container'><p>본문</p></div>"
    assert naver.parse_blog_page(html_) == ("본문", "123")


def test_parse_cbox_comments():
    items = [
        {"commentNo": "1", "parentCommentNo": "1", "contents": "에어컨 연동<br>편해요",
         "regTime": "2026-08-26T16:24:13+0900", "profileUserId": "u1", "sympathyCount": 2},
        {"commentNo": "2", "parentCommentNo": "1", "contents": "감사합니다", "userName": "owner"},
        {"commentNo": "3", "parentCommentNo": "3", "contents": "", "deleted": True},
        {"commentNo": "4", "parentCommentNo": "4", "contents": "비밀", "secret": True},
    ]
    cs = naver.parse_cbox_comments(items, "abc", "9", "naver_blog:abc_9")
    assert [c.comment_id for c in cs] == ["naver_blog:abc_9:1", "naver_blog:abc_9:2"]
    assert cs[0].text == "에어컨 연동 편해요" and cs[0].created_at == "2026-08-26"
    assert cs[0].parent_comment_id == "" and cs[1].parent_comment_id == "naver_blog:abc_9:1"
    assert cs[0].author_hash and cs[0].author_hash != "u1"


def test_jsonl_store_dedup(tmp_path):
    s = JsonlStore(tmp_path / "c.jsonl", "comment_id")
    c = Comment(comment_id="a", post_id="p", source="youtube", text="hi")
    assert s.add(c) and not s.add(c)
    assert len(JsonlStore(tmp_path / "c.jsonl", "comment_id")) == 1


def test_youtube_comments_parse():
    payload = {"items": [{"snippet": {"topLevelComment": {"id": "t1", "snippet": {
        "textDisplay": "냉장고 연동 좋네요", "publishedAt": "2026-05-01T00:00:00Z",
        "authorChannelId": {"value": "UC1"}, "likeCount": 3}}},
        "replies": {"comments": [{"id": "r1", "snippet": {"textDisplay": "동감",
                                  "publishedAt": "2026-05-02T00:00:00Z"}}]}}]}
    resp = mock.Mock(status_code=200, text=json.dumps(payload))
    resp.json.return_value = payload
    with mock.patch("collect.youtube.http_get", return_value=resp), \
         mock.patch.dict("os.environ", {"YOUTUBE_API_KEY": "k"}):
        cs = youtube.fetch_comments("vid")
    assert [c.comment_id for c in cs] == ["youtube:t1", "youtube:r1"]
    assert cs[1].parent_comment_id == "youtube:t1" and cs[0].like_count == 3


def test_log_odds_and_fdr():
    y_i = np.array([50, 1, 10.]); y_j = np.array([5, 50, 10.])
    delta, z = log_odds_dirichlet(y_i, y_j, y_i + y_j)
    assert z[0] > 0 > z[1]
    q = bh_fdr(np.array([0.01, 0.04, 0.03, 0.5]))
    assert np.all(q >= np.array([0.01, 0.04, 0.03, 0.5]) - 1e-12) and q.max() <= 1


def test_end_to_end_synthetic():
    rnd = random.Random(0)
    fridge = ["냉장고 식재료 관리 패밀리허브 화면", "냉장고 문열림 알림 온도"]
    aircon = ["에어컨 무풍 자동화 온도 습도", "에어컨 외출 예약 전기요금 절약"]
    common = ["스마트싱스 앱 연결 편리", "스마트싱스 루틴 설정 쉬움"]
    posts = [{"post_id": f"p{i}", "source": "naver_blog", "title": "스마트싱스 후기",
              "text": rnd.choice(fridge if i % 2 else aircon) + " " + rnd.choice(common),
              "created_at": ""} for i in range(60)]
    comments = [{"comment_id": f"c{i}", "post_id": f"p{i % 60}", "source": "youtube",
                 "text": rnd.choice(common), "created_at": ""} for i in range(120)]
    products = {"냉장고": ["냉장고"], "에어컨": ["에어컨"]}
    docs = tag_products(build_docs(pd.DataFrame(posts), pd.DataFrame(comments)), products)
    assert docs.loc[docs.doc_id == "c1", "products"].iloc[0] == ["냉장고"]   # 부모 상속
    tok = Tokenizer(products)
    kw = product_keywords(docs, [tok(t) for t in docs.text], products, top_n=5)
    top = {p: set(g.keyword) for p, g in kw.groupby("product")}
    assert top["냉장고"] & {"식재료", "패밀리허브", "알림", "문열림", "화면", "관리"}
    assert top["에어컨"] & {"무풍", "습도", "예약", "전기요금", "절약", "외출"}
    assert "냉장고" not in top["에어컨"]


def test_product_regex_no_false_positive():
    import yaml
    products = yaml.safe_load((Path(__file__).parent.parent / "config/settings.yaml")
                              .read_text(encoding="utf-8"))["products"]
    df = pd.DataFrame({"doc_id": ["a", "b", "c"], "kind": "post", "parent_id": ["a", "b", "c"],
                       "source": "x", "created_at": "",
                       "text": ["패밀리허브 냉장고", "TV 미러링 tving", "스마트싱스 허브 앱으로 제어"]})
    ps = tag_products(df, products)["products"].tolist()
    assert ps[0] == ["냉장고"]
    assert ps[1] == ["TV"]
    assert set(ps[2]) == {"허브_스테이션", "스마트폰_앱"}


def test_focus_log_odds_filters_common_and_self_terms():
    from analysis.focus import focus_log_odds, self_terms, split_lo
    rng = random.Random(0)
    toks = ([["큐커", "식단", "앱", "연결"] for _ in range(40)] +
            [["냉장고", "절약", "앱", "연결"] for _ in range(40)])
    rng.shuffle(toks)
    is_focus = np.array(["큐커" in t for t in toks])
    lo = focus_log_odds(toks, is_focus, exclude=self_terms({"큐커": ["큐커"], "냉장고": ["냉장고"]}))
    f, o, common = split_lo(lo)
    assert f["keyword"].tolist() == ["식단"] and o["keyword"].tolist() == ["절약"]
    assert set(common["keyword"]) == {"앱", "연결"}           # 공통어는 |z|<1.96 으로 걸러짐
    assert "큐커" not in set(lo["keyword"])                  # 제품명 자체는 제외


def test_paragraphs_merge_short_lines():
    from analysis.focus import paragraphs
    docs = pd.DataFrame([{"doc_id": "p1", "kind": "post", "stratum": "큐커", "group": "focus",
                          "text": "짧은 줄\n" + "가" * 90 + "\n끝"}])
    paras = paragraphs(docs, min_chars=80)
    assert len(paras) == 1 and paras["text"][0].startswith("짧은 줄") and paras["text"][0].endswith("끝")


def test_topic_share_percentages():
    from analysis.focus import topic_share
    topics = np.array([0, 0, 0, 1, 1, 1, 1, 0])
    group = np.array(["focus", "focus", "focus", "other", "other", "other", "focus", "other"])
    sh = topic_share(topics, group, {0: "다이어트", 1: "절약"}).set_index("topic")
    assert sh.loc[0, "pct_focus"] == 75.0 and sh.loc[0, "pct_other"] == 25.0
    assert sh.index[0] == 0 and sh.loc[0, "z_score"] > 0 > sh.loc[1, "z_score"]
