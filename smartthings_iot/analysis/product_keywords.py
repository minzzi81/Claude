"""제품별 연관 키워드 통계 모델링.

1) Weighted log-odds ratio with informative Dirichlet prior (Monroe, Colaresi & Quinn, 2008)
   제품 문서 vs 나머지 문서의 단어 빈도 차이를 z-score로. 희소 단어의 과대평가를 사전분포로 보정.
2) 카이제곱 검정 (문서 단위 2x2: 단어 출현 × 제품 언급) + Benjamini-Hochberg FDR 보정
3) PMI (문서 단위 제품-단어 동시출현), 최소 동시출현 빈도 적용
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.stats import chi2 as chi2_dist
from sklearn.feature_extraction.text import CountVectorizer


def doc_term(tokens: list[list[str]], min_df: int = 3):
    cv = CountVectorizer(analyzer=lambda x: x, min_df=min_df)
    X = cv.fit_transform(tokens)
    return csr_matrix(X), np.array(cv.get_feature_names_out())


def bh_fdr(p: np.ndarray) -> np.ndarray:
    n = len(p)
    order = np.argsort(p)
    q = p[order] * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(q, 0, 1)
    return out


def log_odds_dirichlet(y_i: np.ndarray, y_j: np.ndarray, prior: np.ndarray,
                       prior_scale: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """y_i: 그룹 i 단어 빈도, y_j: 비교 그룹 빈도, prior: 배경(전체) 빈도. (delta, z) 반환."""
    a = prior * prior_scale + 1e-9
    a0 = a.sum()
    n_i, n_j = y_i.sum(), y_j.sum()
    li = np.log((y_i + a) / (n_i + a0 - y_i - a))
    lj = np.log((y_j + a) / (n_j + a0 - y_j - a))
    delta = li - lj
    var = 1.0 / (y_i + a) + 1.0 / (y_j + a)
    return delta, delta / np.sqrt(var)


def chi2_assoc(B: csr_matrix, g: np.ndarray):
    """B: 이진 문서-단어 행렬, g: 제품 문서 bool. (chi2, p, a=제품 내 출현 문서 수, 양의연관 여부)"""
    N, ng = B.shape[0], g.sum()
    a = np.asarray(B[g].sum(axis=0)).ravel().astype(float)
    b = np.asarray(B[~g].sum(axis=0)).ravel().astype(float)
    c, d = ng - a, (N - ng) - b
    den = (a + b) * (c + d) * (a + c) * (b + d)
    chi = np.where(den > 0, N * (a * d - b * c) ** 2 / np.where(den > 0, den, 1), 0.0)
    return chi, chi2_dist.sf(chi, 1), a, (a * d - b * c) > 0


def product_keywords(docs: pd.DataFrame, tokens: list[list[str]], products: dict[str, list[str]],
                     top_n: int = 30, min_df: int = 3, min_cooc: int = 3,
                     prior_scale: float = 1.0) -> pd.DataFrame:
    """docs['products'](list) 와 tokens(문서별 토큰)로 제품별 연관 키워드 표를 만든다.
    products: 제품명 -> 매칭 패턴. 해당 제품을 정의하는 패턴과 일치하는 토큰은 결과에서 제외한다."""
    X, vocab = doc_term(tokens, min_df=min_df)
    B = (X > 0).astype(np.int8).tocsr()
    total = np.asarray(X.sum(axis=0)).ravel().astype(float)
    df_w = np.asarray(B.sum(axis=0)).ravel().astype(float)
    N = X.shape[0]
    rows = []
    for prod in products:
        g = docs["products"].apply(lambda ps: prod in ps).to_numpy()
        if g.sum() < 5 or (~g).sum() < 5:
            continue
        y_i = np.asarray(X[g].sum(axis=0)).ravel().astype(float)
        y_j = total - y_i
        delta, z = log_odds_dirichlet(y_i, y_j, total, prior_scale)
        chi, p, a, pos = chi2_assoc(B, g)
        q = bh_fdr(p)
        with np.errstate(divide="ignore", invalid="ignore"):
            pmi = np.log2((a / N) / ((df_w / N) * (g.sum() / N)))
        own = re.compile("|".join(f"(?:{x})" for x in products[prod]))
        self_term = np.array([bool(own.fullmatch(w)) for w in vocab])
        ok = (a >= min_cooc) & pos & ~self_term
        idx = np.where(ok)[0]
        idx = idx[np.argsort(-z[idx])][:top_n]
        for rank, k in enumerate(idx, 1):
            rows.append({"product": prod, "rank": rank, "keyword": vocab[k],
                         "n_docs_product": int(g.sum()), "doc_freq_in_product": int(a[k]),
                         "freq_in_product": int(y_i[k]), "log_odds": round(delta[k], 4),
                         "z_score": round(z[k], 3), "chi2": round(chi[k], 3),
                         "p_value": p[k], "q_value_bh": q[k], "pmi": round(pmi[k], 3)})
    return pd.DataFrame(rows)


def cooccurrence_edges(tokens: list[list[str]], min_df: int = 5, min_cooc: int = 5,
                       top_k: int = 300) -> pd.DataFrame:
    """키워드 네트워크용 문서 단위 동시출현 엣지 (NPMI 포함)."""
    X, vocab = doc_term(tokens, min_df=min_df)
    B = (X > 0).astype(np.int32).tocsc()
    C = (B.T @ B).tocoo()
    N, df_w = B.shape[0], np.asarray(B.sum(axis=0)).ravel()
    m = (C.row < C.col) & (C.data >= min_cooc)
    r, c, v = C.row[m], C.col[m], C.data[m].astype(float)
    pmi = np.log((v / N) / ((df_w[r] / N) * (df_w[c] / N)))
    npmi = pmi / -np.log(v / N)
    e = pd.DataFrame({"source": vocab[r], "target": vocab[c], "cooc": v.astype(int),
                      "npmi": npmi.round(4)})
    return e.sort_values(["cooc", "npmi"], ascending=False).head(top_k)
