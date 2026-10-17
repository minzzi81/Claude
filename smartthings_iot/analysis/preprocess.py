"""문서 구성 + 형태소 분석(Kiwi) + 제품 태깅."""
from __future__ import annotations

import re

import pandas as pd
from kiwipiepy import Kiwi

KEEP_TAGS = {"NNG", "NNP", "SL", "XR", "VA"}   # 일반/고유명사, 외국어, 어근, 형용사
# 스마트싱스·연결·기능 같은 공통어는 일부러 남겨 둔다: 제품과 무관한 단어는 log-odds에서 z≈0으로 걸러진다
STOPWORDS = {
    "삼성", "samsung", "사용", "제품", "이번", "정도",
    "오늘", "때문", "생각", "하나", "부분", "경우", "사람", "포스팅", "블로그", "영상", "구독",
    "좋아요", "댓글", "링크", "http", "https", "www", "com", "the", "and", "to", "of",
}
_URL = re.compile(r"https?://\S+|www\.\S+")


def build_docs(posts: pd.DataFrame, comments: pd.DataFrame) -> pd.DataFrame:
    """게시물(제목+본문)과 댓글을 하나의 문서 테이블로 합친다."""
    p = posts.assign(doc_id=posts["post_id"], kind="post", parent_id=posts["post_id"],
                     text=(posts["title"].fillna("") + " " + posts["text"].fillna("")))
    c = comments.assign(doc_id=comments["comment_id"], kind="comment",
                        parent_id=comments["post_id"], text=comments["text"].fillna(""))
    cols = ["doc_id", "kind", "parent_id", "source", "text", "created_at"]
    return pd.concat([p[cols], c[cols]], ignore_index=True)


def tag_products(docs: pd.DataFrame, products: dict[str, list[str]],
                 inherit_from_post: bool = True) -> pd.DataFrame:
    """문서별 언급 제품 리스트. 제품 언급이 없는 댓글은 (옵션) 부모 게시물의 제품을 상속."""
    low = docs["text"].fillna("").str.lower()
    regs = {p: re.compile("|".join(f"(?:{x})" for x in pats)) for p, pats in products.items()}
    docs = docs.copy()
    docs["products"] = [[p for p, rg in regs.items() if rg.search(t)] for t in low]
    if inherit_from_post:
        post_prod = dict(zip(docs.loc[docs.kind == "post", "doc_id"],
                             docs.loc[docs.kind == "post", "products"]))
        mask = (docs.kind == "comment") & (docs["products"].str.len() == 0)
        docs.loc[mask, "products"] = docs.loc[mask, "parent_id"].map(
            lambda pid: post_prod.get(pid, [])).values
    return docs


class Tokenizer:
    def __init__(self, products: dict[str, list[str]] | None = None, extra_stop=(), user_words=()):
        self.kiwi = Kiwi()
        for pats in (products or {}).values():
            for w in pats:
                if " " not in w and re.fullmatch(r"[가-힣]+", w):
                    self.kiwi.add_user_word(w, "NNP")
        for w in ("스마트싱스", "스마트홈", "자동화", "루틴", "매터", "지그비", *user_words):
            self.kiwi.add_user_word(w, "NNP")
        self.stop = STOPWORDS | set(extra_stop)

    def __call__(self, text: str) -> list[str]:
        text = _URL.sub(" ", text or "")
        out = []
        for tok in self.kiwi.tokenize(text):
            if tok.tag not in KEEP_TAGS:
                continue
            w = tok.form.lower()
            if tok.tag == "VA":
                w += "다"
            if len(w) < 2 or w in self.stop or w.isdigit():
                continue
            out.append(w)
        return out
