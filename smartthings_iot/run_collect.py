"""수집 실행: 게시물 ~600 / 댓글 ~5,000 목표.

    python run_collect.py                 # 키가 설정된 모든 소스
    python run_collect.py --sources naver youtube
    python run_collect.py --export        # data/raw/*.jsonl -> data/processed/*.csv
    python run_collect.py --cap naver_blog=600   # 키 없는 소스 몫을 블로그로 채울 때 등 상한 덮어쓰기

재실행하면 이미 저장된 id는 건너뛰고 목표치까지 이어서 수집한다.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd
import yaml
from dotenv import load_dotenv

from collect import instagram, naver, youtube
from collect.common import DATA_DIR, JsonlStore, is_relevant

log = logging.getLogger("collect")
CFG = yaml.safe_load((Path(__file__).parent / "config/settings.yaml").read_text(encoding="utf-8"))
TERMS = CFG["relevance_terms"]


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


def export(raw_dir: Path = DATA_DIR / "raw", out_dir: Path = DATA_DIR / "processed"):
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in ("posts", "comments"):
        f = raw_dir / f"{name}.jsonl"
        if f.exists():
            df = pd.read_json(f, lines=True)
            df.to_csv(out_dir / f"{name}.csv", index=False, encoding="utf-8-sig")
            print(f"{name}: {len(df)}건", df["source"].value_counts().to_dict())


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", nargs="+", default=["naver", "youtube", "instagram"])
    ap.add_argument("--export", action="store_true", help="수집 없이 CSV 내보내기만")
    ap.add_argument("--cap", nargs="+", default=[], metavar="SOURCE=N",
                    help="source_post_caps 덮어쓰기 (예: naver_blog=600)")
    a = ap.parse_args()
    if not a.export:
        c = Collector()
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
    export()


if __name__ == "__main__":
    main()
