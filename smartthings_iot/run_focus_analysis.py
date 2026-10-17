"""큐커 vs 비교군 분석 실행 (Step 1 log-odds → Step 2 BERTopic → 인사이트 요약).

    python run_focus_analysis.py                          # data/strata/processed 사용
    python run_focus_analysis.py --exclude-sponsored      # 협찬 고지 글 제외하고 재확인
    python run_focus_analysis.py --no-topics              # BERTopic 생략 (Step 1만)
    python run_focus_analysis.py --in-dir data/demo/processed --out-dir data/demo/results   # 가상 데이터 데모

출력(out-dir):
    step1_focus_keywords.csv   큐커에서 유난히 많이 쓰인 단어 (z > 1.96)
    step1_other_keywords.csv   비교군에서 유난히 많이 쓰인 단어 (z < -1.96)
    step1_common_words.csv     자주 나오지만 제품과 무관해 걸러진 단어 (|z| < 1.96)
    step2_topic_info.csv       주제별 대표 단어·문단 수·대표 문단
    step2_topic_share.csv      주제별 큐커/비교군 문단 비율(%)과 차이
    step2_paragraph_topics.csv 문단별 주제
    summary.md                 결과 요약
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml

from analysis.focus import (fit_topics, focus_docs, focus_log_odds, paragraphs, self_terms,
                            split_lo, topic_share, topic_words)
from analysis.preprocess import Tokenizer

ROOT = Path(__file__).parent
CFG = yaml.safe_load((ROOT / "config/settings.yaml").read_text(encoding="utf-8"))


def md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", default=str(ROOT / "data/strata/processed"))
    ap.add_argument("--out-dir", default=str(ROOT / "data/strata/results"))
    ap.add_argument("--focus", default=CFG["focus_product"])
    ap.add_argument("--exclude-sponsored", action="store_true")
    ap.add_argument("--no-topics", action="store_true")
    ap.add_argument("--top-n", type=int, default=30)
    ap.add_argument("--min-df", type=int, default=3)
    ap.add_argument("--min-topic-size", type=int, default=15)
    ap.add_argument("--nr-topics", type=int, default=None, help="비슷한 주제를 병합해 이 개수로 줄임")
    ap.add_argument("--embedding-model", default="snunlp/KR-SBERT-V40K-klueNLI-augSTS")
    ap.add_argument("--labels", default=str(ROOT / "config/topic_labels.yaml"),
                    help="주제 번호 -> 이름 (사람이 대표 단어를 보고 붙인다). 없으면 대표 단어로 표시")
    a = ap.parse_args()

    ind, out = Path(a.in_dir), Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    posts = pd.read_csv(ind / "posts.csv")
    comments = pd.read_csv(ind / "comments.csv")
    synthetic = (posts["source"] == "synthetic").any()
    if a.exclude_sponsored:
        posts = posts[posts["sponsored"] != True]  # noqa: E712  (NaN은 비협찬으로 둔다)

    docs = focus_docs(posts, comments, a.focus)
    is_focus = (docs["group"] == "focus").to_numpy()
    tok = Tokenizer(CFG["products"], user_words=CFG.get("user_words", []))
    tokens = [tok(t) for t in docs["text"]]
    banner = ("> ⚠️ **가상 데이터(합성) 데모 결과입니다. 실제 수집 데이터가 아니므로 결과로 인용하면 안 됩니다.**\n\n"
              if synthetic else "")

    # ---------------- Step 1: log-odds ----------------
    lo = focus_log_odds(tokens, is_focus, exclude=self_terms(CFG["products"]), min_df=a.min_df)
    f_kw, o_kw, common = split_lo(lo, a.top_n)
    f_kw.to_csv(out / "step1_focus_keywords.csv", index=False, encoding="utf-8-sig")
    o_kw.to_csv(out / "step1_other_keywords.csv", index=False, encoding="utf-8-sig")
    common.to_csv(out / "step1_common_words.csv", index=False, encoding="utf-8-sig")

    n_posts = docs.query("kind == 'post'").groupby("group").size().to_dict()
    n_cmts = docs.query("kind == 'comment'").groupby("group").size().to_dict()
    md = [f"# {a.focus} vs 비교군 분석 요약\n", banner,
          f"- 게시물: {a.focus} {n_posts.get('focus', 0)}건 / 비교군 {n_posts.get('other', 0)}건",
          f"- 댓글: {a.focus} {n_cmts.get('focus', 0)}건 / 비교군 {n_cmts.get('other', 0)}건",
          f"- 협찬 고지 글 제외: {'예' if a.exclude_sponsored else '아니오'}\n",
          "## Step 1. Log-odds ratio (비교군 대비 큐커에서 유난히 많이 쓰인 단어)\n",
          f"### {a.focus} 고유 키워드 (z > 1.96)\n",
          md_table(f_kw.head(15)[["keyword", "z_score", "per10k_focus", "per10k_other"]]),
          "\n### 비교군 고유 키워드 (z < -1.96)\n",
          md_table(o_kw.head(15)[["keyword", "z_score", "per10k_focus", "per10k_other"]]),
          "\n### 걸러진 공통어 (자주 나오지만 |z| < 1.96)\n",
          ", ".join(f"{r.keyword}(z={r.z_score})" for r in common.itertuples()), ""]

    # ---------------- Step 2: BERTopic ----------------
    if not a.no_topics:
        paras = paragraphs(docs)
        model, topics = fit_topics(paras["text"].tolist(), tok, a.embedding_model,
                                   a.min_topic_size, nr_topics=a.nr_topics)
        paras["topic"] = topics
        lp = Path(a.labels)
        labels = (yaml.safe_load(lp.read_text(encoding="utf-8")) or {}) if lp.exists() and not synthetic else {}
        words = {t: topic_words(model, t) for t in set(topics)}
        labels = {t: labels.get(t) or "·".join(words[t][:4]) for t in words}
        paras["topic_label"] = paras["topic"].map(labels)
        paras.to_csv(out / "step2_paragraph_topics.csv", index=False, encoding="utf-8-sig")

        info = pd.DataFrame([{
            "topic": t, "label": labels[t], "n_paragraphs": int((topics == t).sum()),
            "top_words": ", ".join(words[t]),
            "example": paras.loc[paras.topic == t, "text"].iloc[0][:120]} for t in sorted(words)])
        info.to_csv(out / "step2_topic_info.csv", index=False, encoding="utf-8-sig")
        share = topic_share(topics, paras["group"].to_numpy(), labels)
        share.to_csv(out / "step2_topic_share.csv", index=False, encoding="utf-8-sig")

        md += ["## Step 2. BERTopic (문단 단위 주제 → 큐커/비교군 비율)\n",
               f"- 문단 {len(paras)}개 → 주제 {len(words)}개 "
               f"(임베딩: `{a.embedding_model}`, min_topic_size={a.min_topic_size}, "
               f"nr_topics={a.nr_topics}, seed=42)\n",
               md_table(share.merge(info[["topic", "top_words"]], on="topic")
                        [["topic", "label", "pct_focus", "pct_other", "diff_pp", "z_score", "top_words"]]),
               "\n> 주제 이름은 대표 단어와 대표 문단을 보고 사람이 `config/topic_labels.yaml`에 붙인다.\n"]
        top, bottom = share.iloc[0], share.iloc[-1]
        md += ["## 인사이트 초안 (수치 확인 후 문장 확정)\n",
               f"- {a.focus} 문단이 가장 많이 몰린 주제: **{top.label}** "
               f"({a.focus} {top.pct_focus}% vs 비교군 {top.pct_other}%, z={top.z_score})",
               f"- 비교군 쪽으로 가장 쏠린 주제: **{bottom.label}** "
               f"({a.focus} {bottom.pct_focus}% vs 비교군 {bottom.pct_other}%, z={bottom.z_score})",
               f"- Step 1 {a.focus} 키워드: " + ", ".join(f_kw.head(10)["keyword"]),
               f"- Step 1 비교군 키워드: " + ", ".join(o_kw.head(10)["keyword"])]

    (out / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
