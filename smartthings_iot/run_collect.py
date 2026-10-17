"""수집 실행: 게시물 ~600 / 댓글 ~5,000 목표.

    python run_collect.py                 # 키가 설정된 모든 소스
    python run_collect.py --sources naver youtube
    python run_collect.py --export        # data/raw/*.jsonl -> data/processed/*.csv
    python run_collect.py --cap naver_blog=600   # 키 없는 소스 몫을 블로그로 채울 때 등 상한 덮어쓰기
    python run_collect.py --design strata  # 큐커 중심 층화 수집 (config의 strata) -> data/strata/

재실행하면 이미 저장된 id는 건너뛰고 목표치까지 이어서 수집한다.
"""
from __future__ import annotations

import argparse
import logging
import re
from collections import Counter
from pathlib import Path

import pandas as pd
import yaml
from dotenv import load_dotenv

from collect import instagram, naver, youtube
from collect.common import DATA_DIR, JsonlStore, is_relevant, read_jsonl

log = logging.getLogger("collect")
CFG = yaml.safe_load((Path(__file__).parent / "config/settings.yaml").read_text(encoding="utf-8"))
TERMS = CFG["relevance_terms"]
# 본문 내 협찬 고지 문구 (공정위 추천·보증 심사지침에 따른 표기)
SPONSORED = re.compile(r"소정의 ?(원고료|수수료)|제공\s?받아|지원\s?받아|협찬|체험단|광고\s?포함|유료\s?광고")


class Collector:
    def __init__(self, raw_dir: Path = DATA_DIR / "raw"):
        self.posts = JsonlStore(raw_dir / "posts.jsonl", "post_id")
        self.comments = JsonlStore(raw_dir / "comments.jsonl", "comment_id")
        self.t_posts = CFG["targets"]["posts"]
        self.t_comments = CFG["targets"]["comments"]
        self.caps = CFG["source_post_caps"]

    def posts_done(self) -> bool:
        return len(self.posts) >= self.t_posts

    def comments_done(self) -> bool:
        return len(self.comments) >= self.t_comments

    def room(self, source: str) -> bool:
        return not self.posts_done() and self.posts.n_source(source) < self.caps.get(source, 0)

    def add_post(self, p) -> bool:
        if not is_relevant(f"{p.title} {p.text}", TERMS):
            return False
        return self.posts.add(p)

    def add_comments(self, cs) -> int:
        return sum(self.comments.add(c) for c in cs)

    # ---------------- 소스별 ----------------
    def run_naver(self):
        if naver.has_api_key():
            kinds = [(k, lambda q, k=k: naver.search(k, q)) for k in ("blog", "cafearticle", "news")]
        else:
            log.warning("[naver] API 키 없음 -> 블로그 섹션 검색으로 블로그만 수집 (카페/뉴스 건너뜀)")
            kinds = [("blog", naver.search_web)]
        for kind, search in kinds:
            src = naver.KIND_TO_SOURCE[kind]
            for q in CFG["queries"]:
                if not self.room(src):
                    break
                for item in search(q):
                    if not self.room(src):
                        break
                    p = naver.item_to_post(kind, item, q)
                    if p.post_id in self.posts:
                        continue
                    ids = naver.parse_blog_url(p.url) if kind == "blog" else None
                    blog_no = ""
                    if ids:
                        try:
                            body, blog_no = naver.fetch_blog_page(*ids)
                            if body:
                                p.text, p.extra["snippet_only"] = body, False
                        except Exception as e:
                            log.warning("본문 실패 %s: %s", p.url, e)
                    if not self.add_post(p):
                        continue
                    if blog_no and not self.comments_done():
                        try:
                            self.add_comments(naver.fetch_blog_comments(*ids, blog_no, p.post_id))
                        except Exception as e:
                            log.warning("댓글 실패 %s: %s", p.url, e)
                log.info("[%s] %s -> posts=%d comments=%d", src, q, len(self.posts), len(self.comments))

    def run_youtube(self):
        for q in CFG["queries"]:
            if not self.room("youtube") and self.comments_done():
                break
            vids = youtube.search_videos(q, max_items=50)
            for p in youtube.fetch_videos(vids, q):
                is_new = p.post_id not in self.posts
                if is_new and not self.room("youtube"):
                    continue
                if not self.add_post(p) and is_new:
                    continue
                if not self.comments_done() and (p.comment_count or 0) > 0:
                    need = self.t_comments - len(self.comments)
                    self.add_comments(youtube.fetch_comments(p.post_id.split(":", 1)[1],
                                                             limit=min(300, need)))
            log.info("[youtube] %s -> posts=%d comments=%d", q, len(self.posts), len(self.comments))

    def run_instagram(self):
        for tag in CFG["instagram_hashtags"]:
            if not self.room("instagram"):
                break
            cap = self.caps["instagram"] - self.posts.n_source("instagram")
            for p in instagram.hashtag_posts(tag, max_items=cap):
                # 해시태그로 찾은 글이므로 캡션에 키워드가 없어도 관련 글로 본다
                if self.room("instagram"):
                    self.posts.add(p)
            log.info("[instagram] #%s -> posts=%d", tag, len(self.posts))


class StrataCollector(Collector):
    """큐커 중심 층화 수집: 층(제품)마다 검색어와 할당량. 게시물 = 네이버 블로그, 유튜브는 댓글 보충용.

    포함 기준은 모든 층이 같다: 스마트싱스 언급 + 해당 층 제품 언급.
    """

    def __init__(self, raw_dir: Path = DATA_DIR / "strata" / "raw"):
        super().__init__(raw_dir)
        self.strata = CFG["strata"]
        self.prod_re = {s: re.compile("|".join(f"(?:{x})" for x in CFG["products"][s]))
                        for s in self.strata}
        self.n_stratum = Counter((r.get("source"), r.get("extra", {}).get("stratum"))
                                 for r in read_jsonl(raw_dir / "posts.jsonl"))

    def stratum_room(self, s: str) -> bool:
        return self.n_stratum[("naver_blog", s)] < self.strata[s]["quota"]

    def add_stratum_post(self, p, s: str) -> bool:
        t = f"{p.title} {p.text}"
        if not (is_relevant(t, TERMS) and self.prod_re[s].search(t.lower())):
            return False
        p.extra["stratum"] = s
        p.extra["sponsored"] = bool(SPONSORED.search(t))
        if self.posts.add(p):
            self.n_stratum[(p.source, s)] += 1
            return True
        return False

    def run_naver(self):
        search = (lambda q: naver.search("blog", q)) if naver.has_api_key() else naver.search_web
        for s, spec in self.strata.items():
            for q in spec["queries"]:
                if not self.stratum_room(s):
                    break
                for item in search(q):
                    if not self.stratum_room(s):
                        break
                    p = naver.item_to_post("blog", item, q)
                    ids = naver.parse_blog_url(p.url)
                    if p.post_id in self.posts or not ids:
                        continue
                    try:
                        body, blog_no = naver.fetch_blog_page(*ids)
                    except Exception as e:
                        log.warning("본문 실패 %s: %s", p.url, e)
                        continue
                    if not body:
                        continue
                    p.text, p.extra["snippet_only"] = body, False
                    if not self.add_stratum_post(p, s):
                        continue
                    if blog_no and not self.comments_done():
                        try:
                            self.add_comments(naver.fetch_blog_comments(*ids, blog_no, p.post_id))
                        except Exception as e:
                            log.warning("댓글 실패 %s: %s", p.url, e)
                log.info("[%s] %s -> %d/%d, 전체 posts=%d comments=%d", s, q,
                         self.n_stratum[("naver_blog", s)], spec["quota"],
                         len(self.posts), len(self.comments))

    def run_youtube(self, videos_per_query: int = 10):
        """층별 검색어로 영상을 찾아 댓글을 보충한다. 영상은 게시물 할당량에 포함하지 않는다."""
        for s, spec in self.strata.items():
            for q in spec["queries"]:
                if self.comments_done():
                    return
                for p in youtube.fetch_videos(youtube.search_videos(q, videos_per_query), q):
                    if not self.add_stratum_post(p, s) and p.post_id not in self.posts:
                        continue
                    if (p.comment_count or 0) > 0 and not self.comments_done():
                        need = self.t_comments - len(self.comments)
                        self.add_comments(youtube.fetch_comments(p.post_id.split(":", 1)[1],
                                                                 limit=min(300, need)))
                log.info("[youtube/%s] %s -> comments=%d", s, q, len(self.comments))

    def run_instagram(self):
        log.warning("[instagram] 층화 수집에서는 사용하지 않음")


def export(raw_dir: Path = DATA_DIR / "raw", out_dir: Path = DATA_DIR / "processed"):
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in ("posts", "comments"):
        f = raw_dir / f"{name}.jsonl"
        if f.exists():
            df = pd.read_json(f, lines=True)
            if "extra" in df:   # 분석에 쓰는 층/협찬 여부는 별도 컬럼으로 꺼내 둔다
                for k in ("stratum", "sponsored"):
                    df[k] = df["extra"].map(lambda e, k=k: e.get(k) if isinstance(e, dict) else None)
            df.to_csv(out_dir / f"{name}.csv", index=False, encoding="utf-8-sig")
            print(f"{name}: {len(df)}건", df["source"].value_counts().to_dict())


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", nargs="+", default=["naver", "youtube"])
    ap.add_argument("--export", action="store_true", help="수집 없이 CSV 내보내기만")
    ap.add_argument("--cap", nargs="+", default=[], metavar="SOURCE=N",
                    help="source_post_caps 덮어쓰기 (예: naver_blog=600)")
    ap.add_argument("--design", choices=["default", "strata"], default="default",
                    help="strata: 큐커 중심 층화 수집 (data/strata/)")
    a = ap.parse_args()
    base = DATA_DIR / "strata" if a.design == "strata" else DATA_DIR
    if not a.export:
        c = StrataCollector() if a.design == "strata" else Collector()
        for kv in a.cap:
            src, n = kv.split("=")
            c.caps[src] = int(n)
        for s in a.sources:
            try:
                getattr(c, f"run_{s}")()
            except RuntimeError as e:   # 키 누락 등은 해당 소스만 건너뜀
                log.error("[%s] 건너뜀: %s", s, e)
        log.info("완료: posts=%d/%d comments=%d/%d", len(c.posts), c.t_posts,
                 len(c.comments), c.t_comments)
    export(base / "raw", base / "processed")


if __name__ == "__main__":
    main()
