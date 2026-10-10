"""공통 유틸: 레코드 스키마, JSONL 저장/중복제거, HTTP 재시도, 텍스트 정리."""
from __future__ import annotations

import hashlib
import html
import json
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

import requests

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")


@dataclass
class Post:
    post_id: str          # "{source}:{원본 ID}"
    source: str           # naver_blog | naver_cafe | naver_news | youtube | instagram
    url: str
    title: str
    text: str
    created_at: str = ""
    author_hash: str = ""
    query: str = ""
    like_count: int | None = None
    comment_count: int | None = None
    extra: dict = field(default_factory=dict)


@dataclass
class Comment:
    comment_id: str
    post_id: str
    source: str
    text: str
    created_at: str = ""
    author_hash: str = ""
    like_count: int | None = None
    parent_comment_id: str = ""


def hash_author(name: str | None) -> str:
    """작성자 식별자는 원문 대신 해시로만 저장 (개인정보 최소화)."""
    if not name:
        return ""
    return hashlib.sha256(name.encode("utf-8")).hexdigest()[:16]


_TAG = re.compile(r"<[^>]+>")
_WS = re.compile(r"\s+")


def clean_text(s: str | None) -> str:
    if not s:
        return ""
    s = html.unescape(_TAG.sub(" ", s))
    return _WS.sub(" ", s).strip()


def is_relevant(text: str, terms: Iterable[str]) -> bool:
    t = text.lower()
    return any(term.lower() in t for term in terms)


class JsonlStore:
    """id 기준 중복 제거를 하는 append-only JSONL 저장소. 재실행 시 이어서 수집된다."""

    def __init__(self, path: Path, id_key: str):
        self.path = path
        self.id_key = id_key
        path.parent.mkdir(parents=True, exist_ok=True)
        self.ids: set[str] = set()
        self.count_by_source: dict[str, int] = {}
        if path.exists():
            for line in path.open(encoding="utf-8"):
                if line.strip():
                    rec = json.loads(line)
                    self.ids.add(rec[id_key])
                    src = rec.get("source", "")
                    self.count_by_source[src] = self.count_by_source.get(src, 0) + 1

    def __len__(self) -> int:
        return len(self.ids)

    def __contains__(self, rid: str) -> bool:
        return rid in self.ids

    def add(self, rec) -> bool:
        d = asdict(rec) if not isinstance(rec, dict) else rec
        rid = d[self.id_key]
        if rid in self.ids:
            return False
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
        self.ids.add(rid)
        src = d.get("source", "")
        self.count_by_source[src] = self.count_by_source.get(src, 0) + 1
        return True

    def n_source(self, source: str) -> int:
        return self.count_by_source.get(source, 0)


def http_get(url: str, *, params=None, headers=None, retries: int = 4,
             delay: float = 0.3, timeout: int = 20) -> requests.Response:
    """429/5xx/네트워크 오류 시 지수 백오프 재시도. 요청 간 delay 초 대기."""
    h = {"User-Agent": UA}
    if headers:
        h.update(headers)
    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, headers=h, timeout=timeout)
            if r.status_code == 429 or r.status_code >= 500:
                raise requests.HTTPError(f"{r.status_code}", response=r)
            time.sleep(delay)
            return r
        except requests.RequestException as e:
            last_exc = e
            time.sleep(2 ** (attempt + 1))
    raise RuntimeError(f"GET 실패: {url} ({last_exc})")
