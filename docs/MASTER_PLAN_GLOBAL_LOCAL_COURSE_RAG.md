# 글로벌 · 로컬라이징 분리, 코스 기반 대화형 RAG, 웹 분석 이관 (2026-11-18)

"KRAFTON Raon Zena × premium-discount — 글로벌/로컬라이징 분리, 코스 기반 대화형 RAG 및 웹 분석(`logis-center`) 통합 마스터 기획서"를 반영한 기록입니다.
- raon-zena 쪽 일(Phase 1 대화 · 코스 · RAG 배선, Phase 3 언어팩)이 이 문서입니다.
- premium-discount 쪽 일(Phase 2 웹 분석 이관과 tracker-logis-center 이관)은 그 저장소 README 24절입니다.
- 앞의 기록은 [`MASTER_PLAN_ENGINE_UNIFICATION_AND_VERSION_SYNC.md`](MASTER_PLAN_ENGINE_UNIFICATION_AND_VERSION_SYNC.md) 입니다.

## 기획서 1.2 의 미연결 지점: 지금 상태

| 지점 | 전 | 지금 |
| --- | --- | --- |
| `apps/raon-zena` ↔ `premium-discount-mcp` | 루트에 등록만 되어 있고 부르는 곳 없음 | `pd_bridge` 가 `find_mcp_server()` 로 서버를 띄움(그 서버가 premium-discount-app 을 띄움). `PdSyncClient` 와 `SyncLoop` 를 함께 씀(검색 · 표 · 동기화 · 신구 대체 보고). 서버가 없거나 검색이 실패하면 기기 장부(`zena-verdict` 판정 + 예전에 브리핑한 기록)로 답함(`--no-pd` 로 끔) |
| `SessionContext.verdicts` · `Situation.context` | 슬롯만 있고 늘 비어 있음 | 질문마다 `[추천]`(최대 3줄) · `[참고]` 를 채움. `[코스]` 줄을 새로 둠 |
| `premium-discount-relay` | 부르는 곳 없음 | 턴마다 Relay Key(최근 4턴, 생략된 주어 되살리기). 질문 검색 결과끼리 신구 대체(`Supersedes`) → 옛 기록은 discount |
| `premium-discount-window` | 스타일 비트 · 알파 · 프레임 간격만 | 앞 창 찾기(`foreground_window` · `exclude_self`)와 점 아래 창(`window_at`)으로 질문 검색의 범위를 정하고 `normalize_cc` 로 premium-discount 와 같은 cc(`eldenring.exe`)로 맞춤. 오버레이가 누르기와 끌기(5 px) · 제자리 돌아가기 · 쉬는 동안 82 % 를 씀(`apps/raon-zena/src/pet.rs`) |
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
- 체크포인트 판정(기기): 이긴 런이 지난 체크포인트마다 premium 하나, 진 자리마다 discount 하나를 쌓습니다(`CheckpointTally`, 결정 3개부터 등급, `course_marks.json`). 다음 추천 체크포인트의 로컬 등급이 이것입니다.
- 마찰이 급증하면(한 체크포인트에서 두 번 패배) `ProactiveTimer` 가 정해진 침묵(기본 90초, `--proactive`)을 기다리지 않고 8초 뒤 코칭 한 마디(`proactive.coach`, 사전 노드)를 먼저 건넵니다. 마찰 하나에 한 번입니다.
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
      - 공유에 동의했으면(`--pd-share`) 판정한 대체를 `pd_supersede` 로 premium-discount 에 알립니다(서버의 옛 평가도 `superseded` → discount, 이유 `superseded`).
   4. 남은 기록마다 로컬 판정과 서버 판정을 섞지 않고 `zena_verdict::Stance::from_tiers(local, global).line(label)` 로 `[추천]` 줄을 만듭니다.
      - 로컬 판정: 기기 장부의 판정입니다. 없으면 이 기기의 표(`my_vote`)를 씁니다.
      - 서버 판정: 검색 결과의 `verdict`.
4. 브리핑한 기록은 5분 동안 피드백 · 결과의 크레딧 후보로 남습니다. 원문 · 이름은 기기 장부에도 남습니다(최근 500개).
5. **폴백**: premium-discount-mcp 를 찾지 못했거나, 서버 · 앱이 응답하지 않아 검색이 실패하면 기기 장부에 남은 기록을 질의와 코사인(0.3 이상)으로 골라 같은 모양의 `[추천]` 줄(로컬 판정만, "서버 판정 전")을 만듭니다. 이때는 대체 보고 · 표 보내기를 하지 않고 피드백은 장부에만 쌓습니다.
6. 모든 검색 결과와 코스 · 체크포인트 맥락은 사용자 턴의 Situation(`[추천]` · `[참고]` · `[코스]`)에만 들어갑니다. 시스템 머리는 바뀌지 않아 Raon-1-nano 의 프리픽스 캐시(`Generator::prefix_state`)가 그대로입니다.

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
- 호출어(`WAKE_WORDS`)를 갈아 끼울 수 있습니다: `$ZENA_WAKE_WORDS`(쉼표로 나눔) → 사전의 `meta.wake_words` → 기본값(제나야 · 제나 · 라온 제나 …) 순서입니다. 함께 실은 사전에는 `meta.wake_words` 가 없어 기본값 그대로입니다.

## 기획서와 다른 점

- `zena_verdict::Stance::from_tiers` 는 처음에 없었습니다(`stance(Some(local), Some(global))`). 기획서 이름대로 `Stance::from_tiers(local, global)` 를 더했고 결과는 `stance` 와 같습니다.
- `zena_verdict::feedback::extract` 도 없습니다.
  - 있는 것은 `feedback_obs(intent, negated, credit, at, params)` 이고, 글을 분석하지 않습니다.
  - 사전에는 `feedback.*` 노드가 없고, 라우터는 부정된 구간을 버립니다.
  - 그래서 의도는 언어팩 낱말로, 부정은 분석기의 부정 범위로 정합니다(`pd_bridge::detect_feedback`).
- raon-zena 에는 로컬 판정 장부가 없었습니다(투표 · 판정은 premium-discount-app 안에만 있었음). `LocalLedger` 를 새로 만들었습니다.
- 피드백 한 마디를 기록 k 개에 나누는 1×k 행렬에 기획서의 이중 센터링 + 1:1 배정을 쓰면 애매한 경우를 가리지 못합니다.
  - 센터링은 `v − 평균` 이 됩니다(관측이 하나뿐인 열은 전체 평균을 씁니다).
  - 1:1 배정의 차이는 같은 기록을 두고 다투는 다른 행(피드백)과의 차이입니다. 그래서 행이 하나면 1위 기록과 2위 기록이 아무리 가까워도 1위를 고릅니다.
  - 그래서 기록 사이의 차이를 직접 보는 규칙을 씁니다(1위가 2위보다 0.03 이상 앞서면 1위, 아니면 나눔). 행렬 배정은 피드백이 여럿 쌓일 때의 일입니다.
  - (2026-11-18 고침: 처음에는 "1×k 는 이중 센터링하면 모두 0" 이라고 적었는데, 관측 하나인 행 · 열은 전체 평균을 써서 0 이 되지 않습니다.)
  - 2차 지시문도 이중 센터링 + 1:1 배정(`premium_discount_sds::matrix`)을 다시 요구했지만 위 이유로 바꾸지 않았습니다. 대신 코스 결과의 크레딧은 체크포인트 판정(`CheckpointTally`)으로 더했습니다.
- `premium_discount_schema::address::normalize_key` 는 없습니다. MCP 의 `to_key`(서버가 `hash_id`)를 씁니다.
- 서버 검색 결과에는 원문이 없습니다(서버는 벡터 · 주소만 둠). 신구 대체는 원문이 있는 기록(내 글)끼리만 판정합니다.
- 기록의 시각은 id(UUIDv7)에서 읽습니다. 검색 결과에 시각 칸이 없기 때문입니다.
- Relay 문장은 nms 낱말(Stanza 또는 규칙 분석기)에서 만듭니다. nms 낱말에는 머리 번호가 없습니다. 그래서 수치 뒤의 명사 단위는 붙지 않고, `%` · 기호 단위만 붙습니다. 규칙 분석기에서는 주장(주어 · 목적어 · 뿌리)을 뽑지 못해 대체를 판정하지 않습니다.
- THINK 표정 선행 송출은 이미 있던 동작입니다(사전의 질문 노드). 새로 만든 것은 맥락 단계와 그 시간 상한입니다.
- 코스의 마찰은 "한 체크포인트에서 두 번 패배"로 정합니다. premium-discount-nav 의 `FrictionBaselines`(SDS 띠)는 웹 · 앱 화면의 사건 신호용이라 게임 패배에는 아직 쓰지 않았습니다.

## 검증

| 묶음 | 결과 |
| --- | --- |
| `cargo test --workspace` (2차, PD bb51311) | 140 통과 · 0 실패 · 4 무시, 경고 없음 |
| 프리픽스 캐시 | `prompt::tests::history_and_session_go_into_the_user_turn_so_the_cached_prefix_never_changes` 통과 (`[추천]` · `[참고]` · `[코스]` 를 채워도 시스템 머리 바이트 동일) |
| `zena-locale` | 3: 함께 실은 두 팩의 열쇠가 같음, 열쇠가 빠진 팩은 거절, 폴더에서 읽어 고르기 |
| `zena-core` | 34 (1차 33 + 1). 마찰이 급증하면 침묵이 차기 전 코칭 한 번 |
| `zena-verdict` | 7 (추천 말투가 언어팩에서 와도 같은 문장, `from_tiers` 가 `stance` 와 같음) |
| `zena-lexicon` | 8 (`proactive.coach` 노드) |
| `nlp-nms-router` | 12 (+1 호출어: 기본값 · 사전 `meta.wake_words`) · `eval_ko` 3 |
| `apps/raon-zena` | 33 + 무시 1 (1차 26 + 7). `pd_bridge` 10: 1차 7 + 앱 URI → cc 정규화, premium-discount 없이 장부가 답하고 피드백을 쌓음, 검색이 실패하면 장부로 폴백(브리핑한 기록을 기억). `course` 3: + 이긴 길 premium · 막힌 곳 discount. `pet` 3: 누르기 / 5 px 끌기, 기다렸다 제자리로 미끄러지기(말하면 멈춤), 쉬는 동안 82 % |
| 실제 MCP 왕복 (`--ignored`) | `a_real_mcp_server_round_trip_falls_back_to_the_ledger_without_the_app` 통과: `PD_MCP=<premium-discount-mcp (bb51311 release)>`, 도구 목록에 `pd_search_merged` · `pd_vote` · `pd_supersede` · `pd_verdicts`, premium-discount-app 이 없어 검색은 unavailable → 장부로 답함 |

## 남은 것

- 체크포인트를 통과할 때의 Lv2 즉답("고스트보다 0.8초 빨라!"): 큐 종류와 언어팩 대사가 새로 필요합니다. 지금 코스 상태는 질문 · 능동 발화의 Lv3 프롬프트(`[코스]`)와 마찰 코칭에만 쓰입니다.
- 공개 고스트: 다른 사용자의 코스(`/nav/trials`)를 받아 `Collection::guide` 의 `others` 로 넘기는 것. 지금은 내 기록뿐입니다.
- 오버레이의 누르기 · 끌기 · 제자리 돌아가기는 규칙(`pet.rs`)만 시험했습니다. 실제 창에서 눈으로 본 것은 아닙니다.
- premium-discount-app 까지 띄운 대화 E2E: MCP 서버와의 왕복과 앱이 없을 때의 폴백은 확인했고, 앱이 서버 검색 결과를 돌려주는 경로는 가짜 도구로만 시험했습니다.
- 웹 분석 리포트(`cmd_analytic_report` 의 `lines`)를 제나의 `[참고]` 로 잇는 MCP 도구(기획서 4.2 의 3).
