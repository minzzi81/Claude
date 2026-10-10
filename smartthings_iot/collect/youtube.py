"""YouTube Data API v3: 영상 검색 + 영상 설명 + 댓글(commentThreads) 수집.

할당량(기본 10,000 units/일): search.list=100, videos.list=1, commentThreads.list=1.
"""
from __future__ import annotations

import os

from .common import Comment, Post, clean_text, hash_author, http_get

BASE = "https://www.googleapis.com/youtube/v3/"


def _key() -> str:
    k = os.getenv("YOUTUBE_API_KEY")
    if not k:
        raise RuntimeError("YOUTUBE_API_KEY 환경변수가 필요합니다 (.env)")
    return k


def _get(endpoint: str, **params) -> dict:
    r = http_get(BASE + endpoint, params={"key": _key(), **params})
    if r.status_code == 403 and "commentsDisabled" in r.text:
        return {"items": []}
    r.raise_for_status()
    return r.json()


def search_videos(query: str, max_items: int = 100) -> list[str]:
    ids, token = [], None
    while len(ids) < max_items:
        d = _get("search", part="id", q=query, type="video", maxResults=50,
                 regionCode="KR", relevanceLanguage="ko", pageToken=token or "")
        ids += [it["id"]["videoId"] for it in d.get("items", [])]
        token = d.get("nextPageToken")
        if not token:
            break
    return ids[:max_items]


def fetch_videos(video_ids: list[str], query: str) -> list[Post]:
    posts = []
    for i in range(0, len(video_ids), 50):
        d = _get("videos", part="snippet,statistics", id=",".join(video_ids[i:i + 50]))
        for it in d.get("items", []):
            sn, st = it["snippet"], it.get("statistics", {})
            posts.append(Post(
                post_id=f"youtube:{it['id']}", source="youtube",
                url=f"https://www.youtube.com/watch?v={it['id']}",
                title=clean_text(sn.get("title")), text=clean_text(sn.get("description")),
                created_at=sn.get("publishedAt", "")[:10],
                author_hash=hash_author(sn.get("channelId")), query=query,
                like_count=int(st["likeCount"]) if "likeCount" in st else None,
                comment_count=int(st["commentCount"]) if "commentCount" in st else None,
                extra={"tags": sn.get("tags", [])}))
    return posts


def fetch_comments(video_id: str, limit: int = 300) -> list[Comment]:
    """최상위 댓글 + (응답에 포함된) 대댓글. relevance 순."""
    out, token = [], None
    pid = f"youtube:{video_id}"
    while len(out) < limit:
        d = _get("commentThreads", part="snippet,replies", videoId=video_id, maxResults=100,
                 order="relevance", textFormat="plainText", pageToken=token or "")
        for th in d.get("items", []):
            top = th["snippet"]["topLevelComment"]
            out.append(_to_comment(top, pid))
            for rep in th.get("replies", {}).get("comments", []):
                out.append(_to_comment(rep, pid, parent=top["id"]))
        token = d.get("nextPageToken")
        if not token:
            break
    return out[:limit]


def _to_comment(c: dict, post_id: str, parent: str = "") -> Comment:
    sn = c["snippet"]
    return Comment(comment_id=f"youtube:{c['id']}", post_id=post_id, source="youtube",
                   text=clean_text(sn.get("textDisplay") or sn.get("textOriginal")),
                   created_at=sn.get("publishedAt", "")[:10],
                   author_hash=hash_author(sn.get("authorChannelId", {}).get("value")),
                   like_count=sn.get("likeCount"),
                   parent_comment_id=f"youtube:{parent}" if parent else "")
