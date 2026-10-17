"""큐커(분석 대상) vs 비교군(다른 스마트싱스 제품) 분석.

Step 1 (기본) Weighted log-odds ratio + informative Dirichlet prior (Monroe et al., 2008)
    비교군 대비 큐커 글에서 유난히 많이 쓰인 단어. 제품과 무관한 공통어는 z≈0으로 걸러지고,
    희소 단어는 사전분포(전체 빈도)로 보정된다.
Step 2 (보완) BERTopic
    전체 글을 문단 단위로 넣어 의미가 비슷한 문단끼리 주제로 묶고, 큐커/비교군 문단의 주제별 비율을 계산.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from .preprocess import build_docs
from .product_keywords import doc_term, log_odds_dirichlet

Z_SIG = 1.96   # 양측 95%


def focus_docs(posts: pd.DataFrame, comments: pd.DataFrame, focus: str) -> pd.DataFrame:
    """층(stratum)이 있는 게시물과 그 댓글을 문서로 만들고 group(focus/other)을 붙인다."""
    posts = posts[posts["stratum"].notna() & (posts["stratum"] != "")]
    stratum = dict(zip(posts["post_id"], posts["stratum"]))
    comments = comments[comments["post_id"].isin(stratum)]
    docs = build_docs(posts, comments)
    docs["stratum"] = docs["parent_id"].map(stratum)
    docs["group"] = np.where(docs["stratum"] == focus, "focus", "other")
    return docs.reset_index(drop=True)


def self_terms(products: dict[str, list[str]]) -> re.Pattern:
    """제품명 자체(큐커, 냉장고 …)는 키워드에서 뺀다. 그룹을 정의하는 단어라 당연히 튄다."""
    return re.compile("|".join(f"(?:{x})" for pats in products.values() for x in pats))


def focus_log_odds(tokens: list[list[str]], is_focus: np.ndarray, exclude: re.Pattern | None = None,
                   min_df: int = 3, prior_scale: float = 1.0) -> pd.DataFrame:
    """단어별 log-odds z (양수 = 큐커 쪽, 음수 = 비교군 쪽). 전 어휘를 반환한다."""
    X, vocab = doc_term(tokens, min_df=min_df)
    B = (X > 0).astype(np.int8).tocsr()
    total = np.asarray(X.sum(axis=0)).ravel().astype(float)
    y_f = np.asarray(X[is_focus].sum(axis=0)).ravel().astype(float)
    y_o = total - y_f
    delta, z = log_odds_dirichlet(y_f, y_o, total, prior_scale)
    n_f, n_o = y_f.sum(), y_o.sum()
    df = pd.DataFrame({
        "keyword": vocab, "z_score": z.round(3), "log_odds": delta.round(4),
        "freq_focus": y_f.astype(int), "freq_other": y_o.astype(int),
        "docs_focus": np.asarray(B[is_focus].sum(axis=0)).ravel(),
        "docs_other": np.asarray(B[~is_focus].sum(axis=0)).ravel(),
        "per10k_focus": (y_f / n_f * 1e4).round(1), "per10k_other": (y_o / n_o * 1e4).round(1),
    })
    if exclude is not None:
        df = df[~df["keyword"].map(lambda w: bool(exclude.fullmatch(w)))]
    return df.sort_values("z_score", ascending=False).reset_index(drop=True)


def split_lo(lo: pd.DataFrame, top_n: int = 30, n_common: int = 20):
    """(큐커 고유, 비교군 고유, 공통어=자주 나오지만 |z|<1.96이라 걸러진 단어) 세 표로 나눈다."""
    focus = lo[lo["z_score"] > Z_SIG].head(top_n)
    other = lo[lo["z_score"] < -Z_SIG].sort_values("z_score").head(top_n)
    common = (lo[lo["z_score"].abs() < Z_SIG]
              .assign(freq=lambda d: d["freq_focus"] + d["freq_other"])
              .sort_values("freq", ascending=False).head(n_common))
    return focus, other, common


def paragraphs(docs: pd.DataFrame, min_chars: int = 80) -> pd.DataFrame:
    """게시물 본문을 문단으로 나눈다. 블로그 에디터는 줄마다 문단을 만들어서 짧은 줄은 이어 붙인다."""
    rows = []
    for d in docs[docs["kind"] == "post"].itertuples():
        paras, buf = [], ""
        for line in str(d.text).split("\n"):
            buf = f"{buf} {line}".strip()
            if len(buf) >= min_chars:
                paras.append(buf)
                buf = ""
        if buf:   # 남은 짧은 꼬리는 마지막 문단에 붙인다
            if paras and len(buf) < min_chars / 2:
                paras[-1] += " " + buf
            else:
                paras.append(buf)
        rows += [(d.doc_id, d.stratum, d.group, p) for p in paras]
    out = pd.DataFrame(rows, columns=["post_id", "stratum", "group", "text"])
    out.insert(0, "para_id", [f"{p}#{i}" for i, p in enumerate(out["post_id"])])
    return out


def fit_topics(texts: list[str], tokenizer, embedding_model: str, min_topic_size: int = 15,
               seed: int = 42, reduce_outliers: bool = True, nr_topics: int | None = None):
    """BERTopic: 한국어 문장 임베딩 → UMAP → HDBSCAN → 주제별 대표 단어(c-TF-IDF, Kiwi 토큰)."""
    from bertopic import BERTopic
    from hdbscan import HDBSCAN
    from sentence_transformers import SentenceTransformer
    from sklearn.feature_extraction.text import CountVectorizer
    from umap import UMAP

    emb_model = SentenceTransformer(embedding_model)
    emb = emb_model.encode(texts, batch_size=64, show_progress_bar=False)
    model = BERTopic(
        embedding_model=emb_model,
        umap_model=UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric="cosine",
                        random_state=seed),
        hdbscan_model=HDBSCAN(min_cluster_size=min_topic_size, metric="euclidean",
                              cluster_selection_method="eom", prediction_data=True),
        vectorizer_model=CountVectorizer(tokenizer=tokenizer, lowercase=False, token_pattern=None,
                                         min_df=2),
        top_n_words=10, nr_topics=nr_topics)   # nr_topics: 비슷한 주제끼리 병합해 이 개수로 줄임
    topics, _ = model.fit_transform(texts, emb)
    if reduce_outliers and -1 in topics:   # 어느 주제에도 안 묶인 문단(-1)은 가장 가까운 주제로
        topics = model.reduce_outliers(texts, topics, strategy="embeddings", embeddings=emb)
        model.update_topics(texts, topics=topics, vectorizer_model=model.vectorizer_model)
    return model, np.asarray(topics)


def topic_words(model, topic: int, n: int = 8) -> list[str]:
    return [w for w, _ in (model.get_topic(topic) or []) if w][:n]


def topic_share(topics: np.ndarray, group: np.ndarray, labels: dict[int, str]) -> pd.DataFrame:
    """주제별 큐커/비교군 문단 비율(%)과 차이. z는 주제 단위 log-odds (|z|>1.96이면 쏠림이 유의)."""
    ids = sorted(set(topics))
    f = np.array([((topics == t) & (group == "focus")).sum() for t in ids], dtype=float)
    o = np.array([((topics == t) & (group == "other")).sum() for t in ids], dtype=float)
    _, z = log_odds_dirichlet(f, o, f + o)
    df = pd.DataFrame({"topic": ids, "label": [labels.get(t, "") for t in ids],
                       "n_focus": f.astype(int), "n_other": o.astype(int),
                       "pct_focus": (f / f.sum() * 100).round(1),
                       "pct_other": (o / o.sum() * 100).round(1)})
    df["diff_pp"] = (df["pct_focus"] - df["pct_other"]).round(1)
    df["z_score"] = z.round(2)
    return df.sort_values("diff_pp", ascending=False).reset_index(drop=True)
