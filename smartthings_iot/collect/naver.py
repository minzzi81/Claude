"""네이버 검색 Open API(블로그/카페/뉴스) + 블로그 본문/댓글 수집.

- 검색: https://developers.naver.com/docs/serviceapi/search/blog/blog.md (일 25,000회, 쿼리당 최대 1,000건)
- 검색 API는 요약(description)만 주므로, 블로그는 본문 페이지를 추가로 가져온다.
- 카페 글은 대부분 로그인이 필요해 본문 대신 제목+요약만 저장한다.
- 블로그 댓글(fetch_blog_comments)은 공식 API가 없어 모바일 댓글 페이지를 파싱한다(실험적).
  네이버 페이지 구조가 바뀌면 0건을 반환할 수 있으며, 이 경우 수집기는 게시물만 저장하고 계속 진행한다.
"""
from __future__ import annotations

import os
import re
from email.utils import parsedate_to_datetime
from urllib.parse import parse_qs, urlparse

from bs4 import BeautifulSoup

from .common import Comment, Post, clean_text, hash_author, http_get

API = "https://openapi.naver.com/v1/search/{kind}.json"
KIND_TO_SOURCE = {"blog": "naver_blog", "cafearticle": "naver_cafe", "news": "naver_news"}


def _headers() -> dict:
    cid, sec = os.getenv("NAVER_CLIENT_ID"), os.getenv("NAVER_CLIENT_SECRET")
    if not cid or not sec:
        raise RuntimeError("NAVER_CLIENT_ID / NAVER_CLIENT_SECRET 환경변수가 필요합니다 (.env)")
    return {"X-Naver-Client-Id": cid, "X-Naver-Client-Secret": sec}


def search(kind: str, query: str, max_items: int = 1000, sort: str = "sim"):
    """검색 결과 item(dict)을 순서대로 yield. kind: blog | cafearticle | news"""
    start = 1
    while start <= min(max_items, 1000):
        r = http_get(API.format(kind=kind), headers=_headers(),
                     params={"query": query, "display": 100, "start": start, "sort": sort})
        r.raise_for_status()
        items = r.json().get("items", [])
        if not items:
            return
        yield from items
        if len(items) < 100:
            return
        start += 100


def parse_blog_url(link: str) -> tuple[str, str] | None:
    """blog.naver.com/{blogId}/{logNo} 또는 PostView?blogId=&logNo= 형태에서 (blogId, logNo) 추출."""
    u = urlparse(link)
    if "blog.naver.com" not in u.netloc:
        return None
    qs = parse_qs(u.query)
    if "blogId" in qs and "logNo" in qs:
        return qs["blogId"][0], qs["logNo"][0]
    parts = [p for p in u.path.split("/") if p]
    if len(parts) >= 2 and parts[1].isdigit():
        return parts[0], parts[1]
    return None


def fetch_blog_body(blog_id: str, log_no: str) -> str:
    r = http_get("https://m.blog.naver.com/PostView.naver",
                 params={"blogId": blog_id, "logNo": log_no})
    if r.status_code != 200:
        return ""
    soup = BeautifulSoup(r.text, "html.parser")
    node = (soup.select_one("div.se-main-container")      # 스마트에디터 ONE
            or soup.select_one("div#viewTypeSelector")     # 구 에디터
            or soup.select_one("div#postViewArea"))
    return clean_text(node.get_text(" ")) if node else ""


def _fmt_date(s: str) -> str:
    if re.fullmatch(r"\d{8}", s or ""):
        return f"{s[:4]}-{s[4:6]}-{s[6:]}"
    try:
        return parsedate_to_datetime(s).date().isoformat()
    except Exception:
        return s or ""


def item_to_post(kind: str, item: dict, query: str) -> Post:
    src = KIND_TO_SOURCE[kind]
    link = item.get("link", "")
    if kind == "blog":
        ids = parse_blog_url(link)
        pid = f"{ids[0]}_{ids[1]}" if ids else link
        author = item.get("bloggername")
        date = item.get("postdate", "")
    elif kind == "cafearticle":
        pid, author, date = link, item.get("cafename"), ""
    else:
        pid, author, date = item.get("originallink") or link, None, item.get("pubDate", "")
    return Post(post_id=f"{src}:{pid}", source=src, url=link,
                title=clean_text(item.get("title")), text=clean_text(item.get("description")),
                created_at=_fmt_date(date), author_hash=hash_author(author), query=query,
                extra={"snippet_only": True})


def fetch_blog_comments(blog_id: str, log_no: str, post_id: str, limit: int = 200) -> list[Comment]:
    """[실험적] 모바일 댓글 페이지(u_cbox 마크업)에서 댓글 파싱."""
    r = http_get("https://m.blog.naver.com/CommentList.naver",
                 params={"blogId": blog_id, "logNo": log_no},
                 headers={"Referer": f"https://m.blog.naver.com/{blog_id}/{log_no}"})
    if r.status_code != 200:
        return []
    soup = BeautifulSoup(r.text, "html.parser")
    out: list[Comment] = []
    for i, li in enumerate(soup.select("li.u_cbox_comment")[:limit]):
        txt = li.select_one(".u_cbox_contents")
        if not txt:
            continue
        nick = li.select_one(".u_cbox_nick")
        date = li.select_one(".u_cbox_date")
        cid = li.get("data-info") or f"{i}"
        out.append(Comment(
            comment_id=f"naver_blog:{blog_id}_{log_no}:{cid}", post_id=post_id,
            source="naver_blog", text=clean_text(txt.get_text(" ")),
            created_at=clean_text(date.get_text()) if date else "",
            author_hash=hash_author(nick.get_text() if nick else None)))
    return out
