"""Instagram Graph API 해시태그 검색 (공식 API).

요건: Instagram 비즈니스/크리에이터 계정 + 연결된 Facebook 앱 + instagram_basic 권한 토큰.
제약: 7일간 고유 해시태그 30개까지 조회 가능. 타인 게시물의 '댓글 본문'은 Graph API로 조회할 수 없어
      캡션(게시물)과 comments_count만 수집한다. (비공식 스크래핑은 Instagram 약관 위반이라 지원하지 않음)
"""
from __future__ import annotations

import os

from .common import Post, clean_text, http_get

GRAPH = "https://graph.facebook.com/v21.0/"
FIELDS = "id,caption,comments_count,like_count,timestamp,permalink,media_type"


def _auth() -> tuple[str, str]:
    uid, tok = os.getenv("IG_USER_ID"), os.getenv("IG_ACCESS_TOKEN")
    if not uid or not tok:
        raise RuntimeError("IG_USER_ID / IG_ACCESS_TOKEN 환경변수가 필요합니다 (.env)")
    return uid, tok


def hashtag_posts(tag: str, max_items: int = 100) -> list[Post]:
    uid, tok = _auth()
    r = http_get(GRAPH + "ig_hashtag_search", params={"user_id": uid, "q": tag, "access_token": tok})
    r.raise_for_status()
    data = r.json().get("data", [])
    if not data:
        return []
    hid = data[0]["id"]
    posts: list[Post] = []
    for edge in ("top_media", "recent_media"):
        url, params = GRAPH + f"{hid}/{edge}", {"user_id": uid, "fields": FIELDS,
                                               "limit": 50, "access_token": tok}
        while url and len(posts) < max_items:
            r = http_get(url, params=params)
            r.raise_for_status()
            d = r.json()
            for m in d.get("data", []):
                cap = clean_text(m.get("caption"))
                posts.append(Post(post_id=f"instagram:{m['id']}", source="instagram",
                                  url=m.get("permalink", ""), title="", text=cap,
                                  created_at=m.get("timestamp", "")[:10], query=f"#{tag}",
                                  like_count=m.get("like_count"),
                                  comment_count=m.get("comments_count"),
                                  extra={"media_type": m.get("media_type"), "edge": edge}))
            url, params = d.get("paging", {}).get("next"), None
    return posts[:max_items]
