# STT — whisper-large-v3 (Docker + GPU)

`input/` 에 넣은 음성 파일을 [openai/whisper-large-v3](https://huggingface.co/openai/whisper-large-v3)
로 전사해 `output/` 에 txt로 저장합니다. Docker 이미지 하나로 끝나므로 호스트에 Python이나 CUDA를 설치할 필요가 없습니다.

기본값은 **한국어 회의·통화 녹음**(수십 분~수 시간)에 맞춰 조정되어 있습니다. 모델 카드
권장값을 그대로 쓰면 실제로 겪었던 두 가지 실패(문장 반복 붕괴, 발화 구간 누락)가 나기
때문입니다. 근거와 실측은 아래 "모델 카드 권장값에서 의도적으로 벗어난 두 가지"에 있습니다.

## 요구 사항

- Docker / Docker Compose
- NVIDIA GPU + 최신 드라이버, NVIDIA Container Toolkit

GPU가 없으면 CPU로 동작하지만 실사용이 어려울 만큼 느립니다. 실행 로그 첫 줄의 `GPU:` 표시로
GPU 인식 여부를 확인하세요.

## 실행

전사할 음성 파일을 `input/` 에 넣고 실행합니다.

```powershell
docker compose build          # 최초 1회
docker compose run --rm stt
```

최초 실행에서 모델 가중치 약 3GB를 내려받습니다. 이후로는 `hf-cache` 볼륨에 캐시되어
다시 받지 않습니다.

## 출력

입력 파일 하나당 두 개의 txt가 `output/_inbox/` 에 생성됩니다.

| 파일 | 내용 |
| --- | --- |
| `<이름>.txt` | 전사 결과 전문 (타임스탬프 없음) |
| `<이름>.segments.txt` | 문장별 `[hh:mm:ss -> hh:mm:ss]` 타임스탬프 포함 |

`_inbox/` 는 **방금 나온 결과**가 쌓이는 곳입니다. 중복 판정은 `output/` 전체를 재귀로 훑기
때문에, 결과를 `output/` 아래 어떤 폴더로 옮겨 정리해도 다시 전사되지 않습니다. 다만 그러려면
`.segments.txt` 의 **파일명을 바꾸지 않아야** 합니다.

이미 전사된 파일은 건너뜁니다. 무시하고 다시 전사하려면 `--force` 를 붙입니다.

## 옵션

```powershell
docker compose run --rm stt --help
```

| 옵션 | 기본값 | 설명 |
| --- | --- | --- |
| `--language` | `ko` | 언어 코드. `auto` 면 모델이 자동 감지 |
| `--mode` | `sequential` | `sequential` 은 Whisper 고유 long-form 알고리즘(정확). `chunked` 는 30초 창을 병렬 처리(약 3배 빠르나 타임스탬프가 거칠고 드물게 글자가 깨짐) |
| `--only` | – | 파일명에 이 문자열이 포함된 파일만 처리 |
| `--force` | – | `output/` 에 결과가 있어도 다시 전사 |
| `--no-speech-threshold` | 없음(비활성) | 무음으로 판정된 30초 창을 버리는 임계값. 기본적으로 끕니다(아래 참고) |
| `--collapse-repeats` | `3` | 동일한 문장이 이만큼 연속되면 하나로 합칩니다. `0` 이면 비활성 |
| `--batch-size` | `8` | `--mode chunked` 에서만 사용 |
| `--beams` | `1` | 1보다 크면 beam search (느리지만 약간 정확) |
| `--model` | `openai/whisper-large-v3` | 다른 Whisper 계열 모델로 교체 가능 |

예시 — 영어 파일 하나만 빠르게:

```powershell
docker compose run --rm stt --only interview --language en --mode chunked
```

## HTTP API

배치 실행 말고 **상주 서버**로도 쓸 수 있습니다. 모델을 한 번만 올려 두고 업로드를 받으므로,
파일 하나씩 산발적으로 처리할 때 매번 3GB를 다시 읽지 않습니다.

```powershell
copy .env.example .env      # 토큰·공개범위·보존기한 설정
docker compose up -d api
```

브라우저에서 `http://localhost:17860/docs` 를 열면 스키마와 시험용 폼이 나옵니다.

### 쓰는 법

긴 녹음은 수십 분이 걸려 동기 응답이 불가능합니다. 업로드하면 `job_id` 가 돌아오고,
그걸로 진행률을 확인한 뒤 결과를 가져갑니다.

```bash
# 1. 올린다
curl -X POST http://localhost:17860/jobs -F "file=@회의.m4a"
# → {"id":"a1b2...", "status":"queued", "queue_position":2, "estimated_seconds":740.5, ...}

# 2. 확인한다
curl http://localhost:17860/jobs/a1b2...
# → {"status":"running", "progress":{"done":41,"total":88,"fraction":0.466,"eta":612.3}, ...}

# 3. 가져간다
curl http://localhost:17860/jobs/a1b2...            # JSON (소수점 타임스탬프 포함)
curl http://localhost:17860/jobs/a1b2.../text       # CLI 의 <이름>.txt 와 같은 내용
curl http://localhost:17860/jobs/a1b2.../segments   # CLI 의 <이름>.segments.txt 와 같은 내용
```

`STT_API_TOKEN` 을 설정했다면 모든 요청에 `-H "Authorization: Bearer <토큰>"` 이 필요합니다
(`/healthz` 는 모니터링용이라 예외).

| 엔드포인트 | 설명 |
| --- | --- |
| `POST /jobs` | 업로드 → job 생성. 폼 필드로 `language` `mode` `beams` `no_speech_threshold` `collapse_repeats` `force` |
| `GET /jobs` | 최근 목록 (결과 본문 제외) |
| `GET /jobs/{id}` | 상태·진행률·대기순번·예상시간·결과 |
| `GET /jobs/{id}/text` | 타임스탬프 없는 전문 |
| `GET /jobs/{id}/segments` | 타임스탬프 포함 전문 |
| `DELETE /jobs/{id}` | job 과 결과 삭제 |
| `GET /healthz` | 모델 적재 여부, 큐 길이, 실시간 배속 |

### 알아 둘 것

- **작업은 차례로 처리됩니다.** GPU가 한 장인 것도 있지만, 진행률 계측이 모델 메서드를
  교체하는 방식이라 동시 실행이 애초에 불가능합니다. 워커는 하나뿐이고 나머지는 큐에서 기다립니다.
- **같은 파일을 같은 옵션으로 다시 올리면 전사하지 않고 기존 결과를 돌려줍니다.** 파일명이
  아니라 내용 해시로 판정하므로, 이름이 달라도 같은 녹음이면 GPU를 다시 쓰지 않습니다.
  무시하려면 `force=true`.
- **JSON 결과의 타임스탬프는 초 단위 실수입니다.** txt 로 받으면 `hh:mm:ss` 로 잘립니다.
- **업로드 원본은 전사 직후 삭제됩니다.** job 기록과 결과는 `STT_RETENTION_DAYS`(기본 7일)
  뒤에 자동으로 지워집니다.
- **기본 공개 범위는 그 서버 자신뿐입니다.** `STT_BIND=127.0.0.1` 로 묶여 있어 다른 PC 에서
  `http://<서버IP>:<포트>` 로 접속하면 연결이 거부됩니다. 사내에 열려면 `.env` 에서
  `STT_BIND=0.0.0.0` 으로 바꾸고 포트를 `STT_HOST_PORT` 로 정하십시오.
- **여는 순간 반드시 토큰을 설정하십시오.** 지금 구조에는 사용자별 격리가 없습니다.
  토큰을 가진 사람은 `GET /jobs` 로 **남이 올린 전사본 목록까지 전부** 볼 수 있습니다.
  회의·면담 녹음이 오가는 서비스라 이 점을 먼저 합의하고 여는 편이 좋습니다.
- **GPU 를 고를 수 있습니다.** `.env` 의 `STT_GPU` 에 장치 번호를 씁니다. 인덱스는 0부터라
  7장짜리 장비의 "7번째"는 `6` 입니다.
- **재시작하면 진행 중이던 작업은 실패로 표시됩니다.** 완료된 결과는 볼륨에 남아 유지됩니다.

## 결과 검증

긴 녹음은 조용히 실패할 수 있습니다. 두 가지만 확인하면 아래 두 실패를 잡을 수 있습니다.

- `.segments.txt` 의 **마지막 타임스탬프**가 헤더의 `duration` 에 가까운가 — 멀면 뒷부분이 누락된 것
- **고유 세그먼트 비율**이 지나치게 낮지 않은가 — 낮으면 반복 붕괴가 일어난 것

## 모델 카드 권장값에서 의도적으로 벗어난 두 가지

whisper-large-v3 모델 카드는 long-form 전사에 `condition_on_prev_tokens=True` 와
`no_speech_threshold=0.6` 을 권장하지만, 실제 녹음에서는 두 값 모두 문제를 일으켜 껐습니다.

- **`condition_on_prev_tokens=False`** — `True` 로 두면 반복 붕괴가 일어났습니다.
  14분 통화에서 세그먼트 810개가 생성됐지만 고유한 문장은 35개뿐이었고,
  1분 40초 지점 이후로는 `응 응 응 …` 이 파일 끝까지 반복됐습니다.
- **`no_speech_threshold` 비활성** — `0.6` 으로 두면 실제 발화가 삭제됐습니다.
  36분 37초 회의 녹음에서 마지막 68초(회의 마무리 대화)가 통째로 누락됐고,
  해당 구간만 따로 전사해 보니 정상적인 대화가 담겨 있었습니다.
  이 억제는 파일 끝뿐 아니라 중간 구간에서도 조용히 일어날 수 있어 기본적으로 끕니다.

그 대신 `no_speech_threshold` 를 끄면 조용한 구간에서 필러가 반복되는 짧은 환각이
생깁니다(예: `아..` 가 2초 간격으로 열두 번). 실제 발화는 그렇게 규칙적으로 반복되지
않으므로, 동일 문장이 3회 이상 연속되면 하나로 합치는 후처리(`--collapse-repeats`)로
정리합니다. 발화를 버리는 방식이 아니라 중복만 접기 때문에 내용이 사라지지 않습니다.

## 구성

- `app/core.py` — 전사 엔진. 화면에 출력하지도, 파일을 만들지도 않고 결과를 값으로
  돌려줍니다. 모델 카드에서 벗어난 튜닝값이 전부 여기 한 곳에 있습니다.
- `app/cli.py` — 배치 실행기. 폴더 스캔, 중복 판정, txt 저장, 진행률 표시처럼
  "부르는 쪽"의 몫만 맡습니다.
- `app/server.py` — HTTP API. 같은 core를 쓰되 업로드로 받고 JSON으로 돌려줍니다.
  두 진입점이 엔진을 공유하므로 CLI로 돌린 결과와 API로 돌린 결과가 같습니다.
- `Dockerfile` — python:3.12-slim + ffmpeg + PyTorch 2.8.0 **cu128**.
  RTX 50 시리즈(Blackwell, sm_120)는 CUDA 12.8 휠이 필요하므로 이 버전을 고정했습니다.
- `docker-compose.yml` — 서비스 둘. `stt`(배치)는 `input/`·`output/` 과 `app/` 을 마운트해
  스크립트를 고쳐도 재빌드가 필요 없고, `api`(상주)는 재현성을 위해 `app/` 을 마운트하지
  않습니다. 두 서비스 모두 `STT_GPU` 로 지정한 GPU 한 장만 컨테이너에 넘깁니다.
- `.env.example` — `api` 서비스 설정. `.env` 로 복사해 씁니다(`.env` 는 커밋되지 않습니다).
- 모델 가중치는 `hf-cache` 볼륨에 캐시되어 최초 실행에서만 내려받습니다.

## 참고

- 지원 확장자: m4a, mp3, wav, flac, ogg, opus, aac, wma, mp4, webm (ffmpeg가 읽을 수 있는 형식).
- `input/` 과 `output/` 의 내용, 원본 음성은 저장소에 커밋되지 않습니다(`.gitignore` 참고).
