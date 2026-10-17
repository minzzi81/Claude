# 삼성 SmartThings IoT 소셜 데이터 수집 · 제품별 연관 키워드 분석

목표: 게시물 **~600건**, 댓글 **~5,000건** 수집 → 제품별 연관 키워드 통계 분석

## 1. 설치 / 키 설정

```bash
pip install -r requirements.txt
cp .env.example .env   # 키 입력
```

| 소스 | 필요한 키 | 수집 대상 | 비고 |
|---|---|---|---|
| 네이버 블로그/카페/뉴스 | `NAVER_CLIENT_ID`, `NAVER_CLIENT_SECRET` ([developers.naver.com](https://developers.naver.com) → 검색 API) | 게시물 (블로그는 본문 전문) | 일 25,000회, 쿼리당 최대 1,000건. 카페는 로그인 필요로 요약문만 |
| 네이버 블로그 (키 없음) | — | 게시물 (본문 전문) | 키가 없으면 블로그 섹션 검색(`section.blog.naver.com`)으로 자동 대체. 블로그만, 쿼리당 최대 1,000건 |
| 네이버 블로그 댓글 | — | 댓글 (답글 포함) | **실험적**: 공식 API 없음, 댓글 위젯용 commentBox JSON 호출. 구조 변경 시 0건. 글당 평균 2~3건 |
| YouTube | `YOUTUBE_API_KEY` (Google Cloud → YouTube Data API v3) | 영상(게시물) + **댓글 주 소스** | 일 10,000 units. 이 설정 기준 약 1,500~2,000 units 소모 |
| Instagram | `IG_USER_ID`, `IG_ACCESS_TOKEN` (비즈니스 계정 + Graph API) | 해시태그 게시물 캡션 | **기본 수집에서 제외.** 타인 게시물 댓글 본문은 Graph API로 조회 불가, 비공식 스크래핑은 약관 위반이라 미지원 |

댓글 5,000건은 주로 YouTube에서 채워진다. 키가 없는 소스는 자동으로 건너뛴다.
YouTube는 키 없이 접근하면 클라우드/데이터센터 IP에서 봇 확인 페이지가 떠서 영상·댓글을 가져올 수 없다. 그래서 키 없는 대체 경로는 만들지 않았다.

## 2. 수집

```bash
python run_collect.py                       # 전체
python run_collect.py --sources naver youtube
python run_collect.py --export              # jsonl -> csv 만
python run_collect.py --sources naver --cap naver_blog=600   # 키가 없어 비는 몫을 블로그로 채우기
```

- 목표치·소스별 상한·검색어·관련성 필터·제품 사전: `config/settings.yaml`
- 결과: `data/raw/{posts,comments}.jsonl` (재실행 시 이어서 수집, id 중복 제거) → `data/processed/*.csv`
- 작성자는 원문 대신 SHA-256 해시(16자)만 저장한다.
- 관련성 필터: 제목/본문에 `스마트싱스|smartthings` 포함 글만 저장 (인스타 해시태그 글 제외)

### 스키마
- posts: `post_id, source, url, title, text, created_at, author_hash, query, like_count, comment_count, extra`
- comments: `comment_id, post_id, source, text, created_at, author_hash, like_count, parent_comment_id`

## 3. 분석

```bash
python run_analysis.py              # 댓글은 제품 언급이 없으면 부모 게시물의 제품 태그 상속
python run_analysis.py --no-inherit
```

1. **전처리**: Kiwi 형태소 분석 (명사/외국어/어근/형용사), 불용어 제거, 제품명 사용자 사전 등록
2. **제품 태깅**: `config/settings.yaml`의 정규식 사전 (오탐 방지 lookaround 포함, 예: 패밀리허브 ≠ 허브)
3. **통계 모델** (제품 언급 문서 vs 나머지):
   - **Weighted log-odds ratio + informative Dirichlet prior** (Monroe et al., 2008) → `z_score` 순 정렬 (|z|>1.96 ≈ 유의)
   - **χ² 검정** (문서 단위 2×2) + **Benjamini–Hochberg FDR** → `q_value_bh`
   - **PMI** (제품–단어 문서 동시출현)
   - 제품 자신을 정의하는 단어(예: 냉장고 → 냉장고·패밀리허브)는 결과에서 제외
4. **출력** `data/results/`: `product_keywords.csv`, `product_doc_counts.csv`, `cooccurrence_edges.csv`(NPMI, 네트워크 시각화용), `tokens.jsonl`(LDA/STM 등 추가 모델링용)

## 4. 큐커 vs 비교군 분석 (층화 수집 → Step 1 log-odds → Step 2 BERTopic)

```bash
python run_collect.py --design strata --sources naver youtube   # -> data/strata/{raw,processed}
python run_focus_analysis.py --nr-topics 15                     # -> data/strata/results/
python run_focus_analysis.py --exclude-sponsored --out-dir data/strata/results_nosponsor   # 협찬 글 제외 재확인
```

**수집 설계** (`config/settings.yaml`의 `strata`)
- 큐커(분석 대상) 200건 + 비교군 8개 제품 × 50건 = 네이버 블로그 게시물 600건. 댓글은 블로그 + 유튜브.
- 모든 층에 같은 포함 기준을 적용: 스마트싱스 언급 + 해당 제품 언급.
- 검색어는 "스마트싱스 + 제품명"까지만 넣는다. 다이어트·레시피 같은 맥락어는 넣지 않는다(결론을 수집 단계에서 만들지 않기 위해).
- 본문의 협찬 고지 문구("소정의 원고료", "제공받아" 등)를 `sponsored`로 표시.

**Step 1. Log-odds ratio** (Monroe et al., 2008, informative Dirichlet prior)
- 비교군 글 대비 큐커 글(게시물 + 댓글)에서 유난히 많이 쓰인 단어. z > 1.96 = 큐커 쪽, z < -1.96 = 비교군 쪽.
- 스마트싱스·앱·연결처럼 제품과 무관한 단어는 |z| < 1.96으로 걸러진다(`step1_common_words.csv`). 제품명 자체는 제외.
- 사전분포(전체 빈도)로 몇 번 안 나온 단어가 우연히 튀지 않게 보정.

**Step 2. BERTopic**
- 게시물 본문을 문단 단위로 나눠(짧은 줄은 80자 이상이 되도록 이어 붙임) 한국어 문장 임베딩(KR-SBERT) → UMAP → HDBSCAN.
- 주제별 대표 단어는 Kiwi 토큰 기반 c-TF-IDF. 어느 주제에도 안 묶인 문단은 가장 가까운 주제로 재배정. seed 고정.
- `step2_topic_share.csv`: 주제별 큐커/비교군 문단 비율(%)과 차이, 주제 단위 log-odds z.
- 주제 이름은 대표 단어·대표 문단(`step2_topic_info.csv`)을 보고 `config/topic_labels.yaml`에 붙인다(`{번호: 이름}`).
- `--nr-topics`는 판단이 필요한 값이다. 너무 크면 같은 맥락이 여러 주제로 쪼개지고, 너무 작으면 다른 맥락이 하나로 합쳐진다.

**파이프라인 데모 (가상 데이터)**

```bash
python demo/make_synthetic.py
python run_focus_analysis.py --in-dir data/demo/processed --out-dir data/demo/results --min-topic-size 10 --nr-topics 10
```

⚠️ 가설을 일부러 심어 둔 합성 데이터다. 출력 형태를 미리 보는 용도이며, 결과로 인용하면 안 된다(요약에 경고가 찍힌다).

## 테스트

```bash
python -m pytest -q tests   # 합성 픽스처로 파서·저장소·통계 모듈 검증 (네트워크 불필요)
```
