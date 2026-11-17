# 글로벌 · 로컬라이징 분리, 코스 기반 대화형 RAG, 웹 분석 이관 (2026-11-18)

"KRAFTON Raon Zena × premium-discount — 글로벌/로컬라이징 분리, 코스 기반 대화형 RAG 및 웹 분석(`logis-center`) 통합 마스터 기획서"를 반영한 기록입니다.
- raon-zena 쪽 일(Phase 1 대화 · 코스 · RAG 배선, Phase 3 언어팩)이 이 문서입니다.
- premium-discount 쪽 일(Phase 2 웹 분석 이관과 tracker-logis-center 이관)은 그 저장소 README 24절입니다.
- 앞의 기록은 [`MASTER_PLAN_ENGINE_UNIFICATION_AND_VERSION_SYNC.md`](MASTER_PLAN_ENGINE_UNIFICATION_AND_VERSION_SYNC.md) 입니다.

## 기획서 1.2 의 미연결 지점: 지금 상태

| 지점 | 전 | 지금 |
| --- | --- | --- |
| `apps/raon-zena` ↔ `premium-discount-mcp` | 루트에 등록만 되어 있고 부르는 곳 없음 | `pd_bridge` 가 `find_mcp_server()` 로 서버를 띄움(그 서버가 premium-discount-app 을 띄움). `PdSyncClient` 와 `SyncLoop` 를 함께 씀(검색 · 표 · 동기화). 서버가 없으면 제나는 맥락 없이 답함(`--no-pd` 로 끔) |
| `SessionContext.verdicts` · `Situation.context` | 슬롯만 있고 늘 비어 있음 | 질문마다 `[추천]`(최대 3줄) · `[참고]` 를 채움. `[코스]` 줄을 새로 둠 |
| `premium-discount-relay` | 부르는 곳 없음 | 턴마다 Relay Key(최근 4턴, 생략된 주어 되살리기). 질문 검색 결과끼리 신구 대체(`Supersedes`) → 옛 기록은 discount |
| `premium-discount-window` | 스타일 비트 · 알파 · 프레임 간격만 | 앞 창 찾기(`foreground_window` · `exclude_self`)로 질문 검색의 범위(`cc`, `app://<게임>.exe`)를 정함. 누르기와 끌기 · 제자리 돌아가기 · 점 아래 창은 아직 오버레이에 연결하지 않음 |
| 사이드카 | — | premium-discount 쪽 `tauri.sidecar.conf.json` 이 `premium-discount-app` · `premium-discount-mcp` 를 함께 설치(2026-11-17). raon-zena 는 `$PD_MCP` → raon-zena 옆 → 설치 폴더 순서로 찾음 |

## 대화 · 코스 · RAG 배선 (Phase 1)

### 세션 (`zena-core::session`, `prompt`)
- `SessionTracker` 에 두 가지를 더했습니다.
  - `push_turn_keys` · `active_keys`: 최근 4턴의 Relay Key 입니다. 키가 없는 턴도 한 칸을 차지해, 4턴 동안 나오지 않은 주제는 빠집니다.
  - `set_course` · `course`: 지금 코스입니다.
- `CourseBrief` 는 기획서 3.3 (1)의 모양이고 언어와 상관없는 데이터입니다.
  - 필드: 코스 id, 체크포인트 이름, 고스트와의 차이(음수 = 앞섬), 다음 추천(이름 · 로컬 · 전역 등급), 마찰 큼.
  - 언어팩이 `[코스]` 줄로 만듭니다: `[코스] 체크포인트 '다리' / 고스트보다 10.0초 빠름 / 다음 추천 '보스방' / 이 구간에서 막혀 있다 (마찰 큼)`
  - 줄의 자리는 `[추천]` 뒤, 사용자 턴 안입니다. 시스템 머리(캐시)는 그대로입니다.

### 게임 코스 (`apps/raon-zena/src/course.rs`)
- premium-discount-nav 의 타임트라이얼(`trial/1`)을 게임에 씁니다.
  - 체크포인트: 화면 인코더가 알아본 장면(`VisionKind::Scene`)이 바뀔 때마다 하나를 둡니다. 해시는 `screen_hash(app://<게임>, "scene:<장면 키>")` 입니다.
  - 결승: 화면의 승리(`vision.game_victory`)입니다.
  - 패배: 그 구간의 마찰입니다. 한 체크포인트에서 두 번 지면 "막혀 있다"가 됩니다.
- 승리하면 런이 기기의 기록실(`%LOCALAPPDATA%\RaonZena\courses.json`, 게임마다 코스 10개)에 들어갑니다.
- 다음 판에서는 `Collection::guide` 가 내 최고 기록과 구간 시간을 견주고, 다음 체크포인트를 알려 줍니다.
- 다른 사용자의 코스(공개 고스트, `/nav/trials`)는 아직 받지 않습니다. 지금 고스트는 내 기록뿐입니다.

### 질문 하나의 흐름
1. 표정: 사전의 질문 노드(`ask_*`)가 이미 THINK · FOCUSED 표정을 즉시 냅니다(전부터 있던 동작). 검색 지연은 이 표정이 가립니다.
2. 맥락 단계: Lv3 작업에 맥락 단계(`LlmJob::prepare`)를 붙입니다.
   - LLM 스레드가 생성 전에 다리를 부르고, 2.5초(`ASK_CONTEXT_BUDGET`) 안에 오지 않으면 맥락 없이 답합니다.
   - 파이프라인의 프레임 루프는 기다리지 않습니다.
3. 다리 `resolve_context_for_ask(cc, active_keys, query)`:
   1. 생략된 주어를 되살립니다: 질의에 없는 활성 키를 앞에 붙입니다("그럼 2페이즈는?" → "말레니아 그럼 2페이즈는?").
   2. `pd_search_merged` 로 찾습니다. 가장 최근 주제를 대상(`to_key`)으로 먼저 찾고, 없으면 대상 없이 찾습니다. 범위는 앞에 있던 게임(`cc`)입니다.
   3. 원문과 시각(UUIDv7)이 있는 기록끼리 `premium_discount_relay::compare` 로 견줍니다. 늦은 기록이 옛 기록의 수치 · 단위 · 형용사를 바꿨으면 대체입니다. 옛 기록은 기기 장부에서 `superseded` 가 되어 discount 로 내려가고, `[참고]` 에 "더 새 기록이 있다"고 남습니다.
   4. 남은 기록마다 로컬 판정과 서버 판정을 섞지 않고 `zena_verdict::stance(local, global).line(label)` 로 `[추천]` 줄을 만듭니다.
      - 로컬 판정: 기기 장부의 판정입니다. 없으면 이 기기의 표(`my_vote`)를 씁니다.
      - 서버 판정: 검색 결과의 `verdict`.
4. 브리핑한 기록은 5분 동안 피드백 · 결과의 크레딧 후보로 남습니다.

### 피드백 · 결과 → 기기 장부 · 표
- **피드백 찾기**: 언어팩의 `app.feedback_recommend` · `app.feedback_not_recommend` 글이 사용자의 말에 들어 있는지 봅니다.
  - 그 낱말이 분석기의 부정 범위 안이면 부정입니다("안 통했어").
  - `zena_verdict::feedback_obs` 가 부호와 무게를 정합니다. 이중 부정은 반만큼 반대쪽입니다.
- **크레딧**: 피드백 한 마디를 최근에 브리핑한 기록 k 개에 나눕니다.
  - 말과 기록의 코사인 1위가 2위보다 0.03 이상 앞서면 1위에 몰아주고, 아니면 1위 무리가 똑같이 나눕니다.
  - 결과(화면의 승리 · "이겼다")는 브리핑한 기록에 똑같이 나눕니다.
- **기기 장부** (`LocalLedger`, `%LOCALAPPDATA%\RaonZena\premium-discount\zena_ledger.json`): 기록마다 관측과 대체 여부를 둡니다.
  - 판정은 PD-Verdict v1 의 `chat` 트랙이고, 기준선은 **내 다른 기록들의 값**입니다(또래와 견줌). 기록이 모자라면 판정하지 않습니다.
  - 시험: 기록 12개 중 계속 칭찬받은 것 → premium, 계속 혹평받은 것 → discount, 대체된 것 → discount.
- **표**: `--pd-share` 로 동의했을 때만 `pd_vote` 로 보냅니다.
  - 피드백은 한 기록에 몰렸을 때만 `vote` 로, 결과는 `vote_outcome` 으로 보냅니다.
  - 보내지 못한 표는 premium-discount-app 의 보관함이 들고 있다가 `SyncLoop` 가 다시 보냅니다.

## 언어팩 (Phase 3): `crates/zena-locale`, `data/locales/<언어>.json`

- 국가별로 갈아 끼우는 것은 모델(`LlmBackend`) · 표현 사전(`data/zena_lexicon.json`)과 **말의 틀**입니다.
- 말의 틀을 언어팩 파일 하나로 모았습니다.
  - 시스템 페르소나
  - 프롬프트의 블록 이름 · 세션 요약 · 말투 힌트 · 거리 · 흐름 · 방향 이름 · 코스 줄
  - Lv2 즉답 큐 대사
  - 추천 · 비추천 말투(`speak.rs`)
  - 앱 문구(화면 맥락, 게임 사건 이름, 대체 알림, 템플릿 백엔드의 대사, 피드백 낱말)
- 한국어(`ko`) · 영어(`en`)를 함께 실었습니다. 다른 언어는 같은 모양의 파일을 `--locale-dir` 로 읽고 `--lang` 으로 고릅니다(코드 수정 없음).
  - 빠진 열쇠가 있는 팩은 읽지 않습니다. 반쯤 번역된 팩이 조용히 빈 줄을 내지 않게 하려는 것입니다.
- **한국어 결과는 바이트까지 그대로입니다.** 기존 시험이 그대로 통과합니다.
  - 프롬프트 머리가 바이트까지 같은지 보는 시험
  - `[세션]` · `[말투]` · `[추천]` · 방향 · 큐 순서 · 추천 말투 문장
- 페르소나는 Raon-1-nano 가 캐시하는 시스템 머리에 들어갑니다. 그래서 언어는 시작할 때 한 번 정합니다(`main` 이 백엔드를 띄우기 전에 `set_language`).
- 렌더 함수는 언어팩을 인자로도 받습니다(`render_in`). 시험이 전역 언어를 바꾸지 않고 영어 결과를 봅니다.

## 기획서와 다른 점

- `zena_verdict::Stance::from_tiers` 는 없습니다. `zena_verdict::stance(Some(local), Some(global)).line(label)` 입니다.
- `zena_verdict::feedback::extract` 도 없습니다.
  - 있는 것은 `feedback_obs(intent, negated, credit, at, params)` 이고, 글을 분석하지 않습니다.
  - 사전에는 `feedback.*` 노드가 없고, 라우터는 부정된 구간을 버립니다.
  - 그래서 의도는 언어팩 낱말로, 부정은 분석기의 부정 범위로 정합니다(`pd_bridge::detect_feedback`).
- raon-zena 에는 로컬 판정 장부가 없었습니다(투표 · 판정은 premium-discount-app 안에만 있었음). `LocalLedger` 를 새로 만들었습니다.
- 피드백 한 마디를 기록 k 개에 나누는 1×k 행렬은 이중 센터링하면 모두 0 이 됩니다. 그래서 차이(margin) 1:1 규칙을 씁니다. 행렬 배정은 피드백이 여럿 쌓일 때의 일입니다.
- `premium_discount_schema::address::normalize_key` 는 없습니다. MCP 의 `to_key`(서버가 `hash_id`)를 씁니다.
- 서버 검색 결과에는 원문이 없습니다(서버는 벡터 · 주소만 둠). 신구 대체는 원문이 있는 기록(내 글)끼리만 판정합니다.
- 기록의 시각은 id(UUIDv7)에서 읽습니다. 검색 결과에 시각 칸이 없기 때문입니다.
- Relay 문장은 nms 낱말(Stanza 또는 규칙 분석기)에서 만듭니다. nms 낱말에는 머리 번호가 없습니다. 그래서 수치 뒤의 명사 단위는 붙지 않고, `%` · 기호 단위만 붙습니다. 규칙 분석기에서는 주장(주어 · 목적어 · 뿌리)을 뽑지 못해 대체를 판정하지 않습니다.
- THINK 표정 선행 송출은 이미 있던 동작입니다(사전의 질문 노드). 새로 만든 것은 맥락 단계와 그 시간 상한입니다.
- 코스의 마찰은 "한 체크포인트에서 두 번 패배"로 정합니다. premium-discount-nav 의 `FrictionBaselines`(SDS 띠)는 웹 · 앱 화면의 사건 신호용이라 게임 패배에는 아직 쓰지 않았습니다.

## 검증

| 묶음 | 결과 |
| --- | --- |
| `cargo test --workspace` | 전부 통과, 경고 없음 |
| `zena-locale` | 3: 함께 실은 두 팩의 열쇠가 같음, 열쇠가 빠진 팩은 거절, 폴더에서 읽어 고르기 |
| `zena-core` | 33 (기존 30 + 3). 기존의 한국어 문장 · 프롬프트 머리 바이트 동일 시험이 그대로 통과. 코스 줄(앞섬 · 뒤짐 · 같음 · 등급 다름 · 판정 전), 영어 팩 렌더(한글이 하나도 남지 않음), 활성 키 4턴 |
| `zena-verdict` | 7 (추천 말투가 언어팩에서 와도 같은 문장) |
| `apps/raon-zena` | 26 (기존 17 + 9). `pd_bridge` 7: 주어 되살리기 · UUIDv7 시각, 크레딧 배분, `[추천]` 줄(로컬 · 서버 · 내 표), 실제 신구 대체(20% → 15%, 옛 기록 discount · `[참고]`), 피드백 · 결과 크레딧과 동의한 표만, 장부의 또래 기준 판정 · 저장, 피드백 낱말 · 부정. `course` 2: 이긴 코스가 다음 판의 고스트(다리에서 10초 앞섬 · 다음 '보스방'), 다른 게임 · 짧은 판 |
| `nlp-nms-router` `eval_ko` | 3 통과 (라우터가 모델을 다리와 나눠 쓰도록 바꾼 뒤에도 같음) |

## 남은 것

- 기획서 3.2 의 1단계 중 둘을 넣지 않았습니다.
  - 체크포인트를 통과할 때의 Lv2 즉답("고스트보다 0.8초 빨라!"): 큐 종류와 언어팩 대사가 새로 필요합니다.
  - 마찰이 급증할 때 `ProactiveTimer` 로 먼저 말 걸기.
  - 지금 코스 상태는 질문 · 능동 발화의 Lv3 프롬프트(`[코스]`)에만 들어갑니다.
- 공개 고스트: 다른 사용자의 코스(`/nav/trials`)를 받아 `Collection::guide` 의 `others` 로 넘기는 것. 지금은 내 기록뿐입니다.
- 오버레이의 누르기와 끌기(5 px) · 제자리 돌아가기 · 점 아래 창(`premium-discount-window`).
- 실제 premium-discount-mcp · premium-discount-app 을 띄운 대화 E2E. 이 PC 의 설치본으로 확인하지 않았고, 다리는 가짜 도구로 시험했습니다.
- 웹 분석 리포트(`cmd_analytic_report` 의 `lines`)를 제나의 `[참고]` 로 잇는 MCP 도구(기획서 4.2 의 3).
