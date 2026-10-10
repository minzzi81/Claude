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
| 네이버 블로그 댓글 | — | 댓글 | **실험적**: 공식 API 없음, 댓글 위젯(cbox) JSONP 응답 사용. 구조 변경 시 0건 |
| YouTube | `YOUTUBE_API_KEY` (Google Cloud → YouTube Data API v3) | 영상(게시물) + **댓글 주 소스** | 일 10,000 units. 이 설정 기준 약 1,500~2,000 units 소모 |
| Instagram | `IG_USER_ID`, `IG_ACCESS_TOKEN` (비즈니스 계정 + Graph API) | 해시태그 게시물 캡션 | 타인 게시물 댓글 본문은 Graph API로 조회 불가. 비공식 스크래핑은 약관 위반이라 미지원 |

댓글 5,000건은 주로 YouTube에서 채워진다. 키가 없는 소스는 자동으로 건너뛴다.

## 2. 수집

```bash
python run_collect.py                       # 전체
python run_collect.py --sources naver youtube
python run_collect.py --export              # jsonl -> csv 만
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

## 테스트

```bash
python -m pytest -q tests   # 합성 픽스처로 파서·저장소·통계 모듈 검증 (네트워크 불필요)
```
