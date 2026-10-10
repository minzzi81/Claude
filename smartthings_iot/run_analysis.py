"""제품별 연관 키워드 분석 실행.

    python run_analysis.py                    # data/processed/{posts,comments}.csv 사용
    python run_analysis.py --no-inherit       # 댓글이 부모 게시물의 제품 태그를 상속하지 않음

출력(data/results/):
    product_keywords.csv   제품별 상위 키워드 (log-odds z, chi2, BH q-value, PMI)
    product_doc_counts.csv 제품별 언급 문서 수 (게시물/댓글)
    cooccurrence_edges.csv 키워드 동시출현 네트워크 엣지
    tokens.jsonl           문서별 토큰 (추가 모델링(LDA/STM 등) 재사용용)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml

from analysis.preprocess import Tokenizer, build_docs, tag_products
from analysis.product_keywords import cooccurrence_edges, product_keywords

ROOT = Path(__file__).parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", default=str(ROOT / "data/processed"))
    ap.add_argument("--out-dir", default=str(ROOT / "data/results"))
    ap.add_argument("--no-inherit", action="store_true")
    ap.add_argument("--top-n", type=int, default=30)
    ap.add_argument("--min-df", type=int, default=3)
    a = ap.parse_args()

    products = yaml.safe_load((ROOT / "config/settings.yaml").read_text(encoding="utf-8"))["products"]
    ind, out = Path(a.in_dir), Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    posts = pd.read_csv(ind / "posts.csv")
    comments = pd.read_csv(ind / "comments.csv") if (ind / "comments.csv").exists() else \
        pd.DataFrame(columns=["comment_id", "post_id", "source", "text", "created_at"])

    docs = tag_products(build_docs(posts, comments), products, inherit_from_post=not a.no_inherit)
    tok = Tokenizer(products)
    tokens = [tok(t) for t in docs["text"]]

    with (out / "tokens.jsonl").open("w", encoding="utf-8") as f:
        for did, kind, ps, ts in zip(docs.doc_id, docs.kind, docs.products, tokens):
            f.write(json.dumps({"doc_id": did, "kind": kind, "products": ps, "tokens": ts},
                               ensure_ascii=False) + "\n")

    counts = (docs.explode("products").dropna(subset=["products"])
              .groupby(["products", "kind"]).size().unstack(fill_value=0))
    counts.to_csv(out / "product_doc_counts.csv", encoding="utf-8-sig")

    kw = product_keywords(docs, tokens, products, top_n=a.top_n, min_df=a.min_df)
    kw.to_csv(out / "product_keywords.csv", index=False, encoding="utf-8-sig")
    cooccurrence_edges(tokens).to_csv(out / "cooccurrence_edges.csv", index=False,
                                      encoding="utf-8-sig")

    print(f"문서 {len(docs)}개 (게시물 {(docs.kind == 'post').sum()}, 댓글 {(docs.kind == 'comment').sum()})")
    print(counts)
    if not kw.empty:
        for p, g in kw.groupby("product", sort=False):
            print(f"[{p}] " + ", ".join(g.head(10)["keyword"]))


if __name__ == "__main__":
    main()
