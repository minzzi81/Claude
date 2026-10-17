"""네이버 검색 Open API(블로그/카페/뉴스) + 블로그 본문/댓글 수집.

- 검색: https://developers.naver.com/docs/serviceapi/search/blog/blog.md (일 25,000회, 쿼리당 최대 1,000건)
- 검색 API는 요약(description)만 주므로, 블로그는 본문 페이지를 추가로 가져온다.
- 카페 글은 대부분 로그인이 필요해 본문 대신 제목+요약만 저장한다.
- API 키가 없으면 search_web()으로 블로그 섹션 검색(section.blog.naver.com)을 대신 쓴다(블로그만, 비공식).
- 블로그 댓글(fetch_blog_comments)은 공식 API가 없어 웹 댓글 위젯이 쓰는 commentBox JSON을 호출한다(실험적).
  응답 구조가 바뀌면 0건을 반환할 수 있으며, 이 경우 수집기는 게시물만 저장하고 계속 진행한다.
"""
from __future__ import annotations

import json
import os
import re
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from urllib.parse import parse_qs, urlparse

from bs4 import BeautifulSoup

from .common import Comment, Post, clean_text, hash_author, http_get

API = "https://openapi.naver.com/v1/search/{kind}.json"
KIND_TO_SOURCE = {"blog": "naver_blog", "cafearticle": "naver_cafe", "news": "naver_news"}
WEB_SEARCH = "https://section.blog.naver.com/ajax/SearchList.naver"
CBOX = "https://apis.naver.com/commentBox/cbox/web_naver_list_jsonp.json"
KST = timezone(timedelta(hours=9))


def has_api_key() -> bool:
    return bool(os.getenv("NAVER_CLIENT_ID") and os.getenv("NAVER_CLIENT_SECRET"))


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


def search_web(query: str, max_items: int = 1000):
    """[키 불필요] 블로그 섹션 검색 JSON을 API item 형식(dict)으로 변환해 yield. 블로그 전용."""
    page, per, seen = 1, 7, 0          # 이 엔드포인트는 페이지당 7건 고정, 최대 1,000건
    while seen < max_items:
        r = http_get(WEB_SEARCH, params={"countPerPage": per, "currentPage": page, "keyword": query,
                                         "orderBy": "sim", "type": "post"},
                     headers={"Referer": "https://section.blog.naver.com/Search/Post.naver"})
        r.raise_for_status()
        res = json.loads(r.text.split("\n", 1)[1])["result"]   # 첫 줄은 XSSI 방지 접두어 ")]}',"
        items = res.get("searchList") or []
        if not items:
            return
        for it in items:
            yield web_item_to_api(it)
        seen += len(items)
        if len(items) < per or seen >= (res.get("totalCount") or 0):
            return
        page += 1


def web_item_to_api(it: dict) -> dict:
    ts = it.get("addDate")
    date = datetime.fromtimestamp(ts / 1000, KST).strftime("%Y%m%d") if ts else ""
    return {"title": it.get("title", ""), "link": it.get("postUrl", ""),
            "description": it.get("contents", ""), "bloggername": it.get("domainIdOrBlogId"),
            "postdate": date}


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


def fetch_blog_page(blog_id: str, log_no: str) -> tuple[str, str]:
    """모바일 본문 페이지에서 (본문, blogNo) 반환. blogNo는 댓글 조회에 필요한 숫자 ID."""
    r = http_get("https://m.blog.naver.com/PostView.naver",
                 params={"blogId": blog_id, "logNo": log_no})
    if r.status_code != 200:
        return "", ""
    return parse_blog_page(r.text)


def parse_blog_page(page: str) -> tuple[str, str]:
    soup = BeautifulSoup(page, "html.parser")
    node = (soup.select_one("div.se-main-container")      # 스마트에디터 ONE
            or soup.select_one("div#viewTypeSelector")     # 구 에디터
            or soup.select_one("div#postViewArea"))
    m = re.search(r"var blogNo = '(\d+)'", page)
    if not node:
        return "", (m.group(1) if m else "")
    # 문단 경계는 "\n"으로 보존한다 (BERTopic을 문단 단위로 돌리기 위해)
    paras = [clean_text(x.get_text(" ")) for x in node.select("p.se-text-paragraph")]
    body = "\n".join(x for x in paras if x) or clean_text(node.get_text(" "))
    return body, (m.group(1) if m else "")


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


def fetch_blog_comments(blog_id: str, log_no: str, blog_no: str, post_id: str,
                        limit: int = 300) -> list[Comment]:
    """[실험적] 댓글 위젯용 commentBox JSON에서 댓글+답글 수집 (삭제/비밀 댓글 제외)."""
    out: list[Comment] = []
    page = 1
    while len(out) < limit:
        r = http_get(CBOX, params={
            "ticket": "blog", "templateId": "default", "pool": "blogid", "lang": "ko",
            "country": "KR", "objectId": f"{blog_no}_201_{log_no}", "groupId": blog_no,
            "pageSize": 50, "indexSize": 10, "listType": "OBJECT", "pageType": "default",
            "page": page, "initialize": "true", "useAltSort": "true", "replyPageSize": 100,
            "showReply": "true"},
            headers={"Referer": f"https://m.blog.naver.com/{blog_id}/{log_no}"})
        if r.status_code != 200:
            break
        res = r.json().get("result") or {}
        out += parse_cbox_comments(res.get("commentList") or [], blog_id, log_no, post_id)
        if page >= (res.get("pageModel") or {}).get("totalPages", 1):
            break
        page += 1
    return out[:limit]


def parse_cbox_comments(items: list[dict], blog_id: str, log_no: str, post_id: str) -> list[Comment]:
    prefix = f"naver_blog:{blog_id}_{log_no}:"
    out = []
    for c in items:
        text = clean_text(c.get("contents"))
        if c.get("deleted") or c.get("secret") or not text:
            continue
        no, parent = c["commentNo"], c.get("parentCommentNo")
        out.append(Comment(
            comment_id=prefix + no, post_id=post_id, source="naver_blog", text=text,
            created_at=(c.get("regTime") or "")[:10],
            author_hash=hash_author(c.get("profileUserId") or c.get("userName")),
            like_count=c.get("sympathyCount"),
            parent_comment_id=prefix + parent if parent and parent != no else ""))
    return out
