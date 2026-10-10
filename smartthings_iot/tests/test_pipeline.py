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
