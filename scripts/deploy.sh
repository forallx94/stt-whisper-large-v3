#!/bin/sh
# GPU 서버에 배포한다. GitLab CI 의 deploy 잡이 부르지만 손으로 실행해도 된다.
#
# 배포 대상이 하나뿐이라 레지스트리를 거치지 않는다. 러너가 GPU 서버 위에서 돌고
# 여기서 바로 빌드한다. 12.6GB 이미지를 망으로 밀어 넣을 이유가 없고, 빌드 캐시도
# 대상 서버에 그대로 쌓인다.
#
# ── 이 스크립트가 도는 환경에서 조심할 것 ───────────────────────────────
#
# CI 잡은 docker:24-cli 컨테이너(Alpine/BusyBox ash) 안에서 돈다. 그래서
#
#   · bash 가 아니다. $SECONDS, [[ ]] 같은 bash 전용 기능을 쓸 수 없다.
#   · python 이 없다. JSON 파싱을 여기서 할 수 없다.
#   · 127.0.0.1 은 잡 컨테이너 자신이지 호스트가 아니다. 호스트에 공개된
#     API 포트로 curl 하면 연결되지 않는다 (실측 확인).
#
# 그래서 상태 확인은 전부 api 컨테이너 안에서 한다(docker compose exec).
# 네트워크 구성에 의존하지 않고, 파싱도 python 이 있는 쪽에서 처리된다.
#
# compose 프로젝트 이름을 -p 로 고정하는 것도 중요하다. CI 작업 디렉터리는 매번
# 새로 체크아웃되지만 이름이 같으면 같은 컨테이너와 같은 볼륨(모델 캐시 hf-cache,
# job 기록 api-data)을 이어받는다. 안 그러면 배포할 때마다 모델 3GB를 다시 받는다.
set -eu

PROJECT="${COMPOSE_PROJECT:-stt}"
# 전사 중인 작업이 끝나기를 최대 얼마나 기다릴지(초). 0이면 기다리지 않고 바로 교체한다.
DRAIN_TIMEOUT="${DRAIN_TIMEOUT:-5400}"
# 교체 후 모델 적재를 기다리는 한도(초). 첫 배포는 가중치 3GB 내려받기를 포함한다.
READY_TIMEOUT="${READY_TIMEOUT:-1800}"
INTERVAL=10

log() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }
dc()  { docker compose -p "$PROJECT" "$@"; }

# api 컨테이너 안에서 /healthz 를 읽어 "ready queued running" 세 값을 한 줄로 뱉는다.
# 컨테이너가 없거나 아직 안 떴으면 0이 아닌 값으로 끝난다.
api_state() {
  dc exec -T api python -c '
import json, urllib.request
h = json.load(urllib.request.urlopen("http://localhost:17860/healthz"))
print("%s %s %s" % (h["ready"], h["queued"], h["running"]))
' 2>/dev/null
}

# ---------------------------------------------------------------- 1. 설정 파일
#
# .env 는 저장소에 없다(토큰이 들어가므로). CI 변수에서 만들어 쓴다.
# 설정의 출처가 GitLab 한 곳으로 모이고, 서버에 평문 비밀이 방치되지 않는다.
write_env() {
  # STT_UI_USERS 는 쉼표가 들어가면 GitLab 이 masked 를 거부한다(실측). 그래서
  # base64 로 넣고 여기서 푼다. base64 알파벳은 masked 허용 문자와 일치한다.
  if [ -n "${STT_UI_USERS_B64:-}" ]; then
    STT_UI_USERS=$(printf '%s' "$STT_UI_USERS_B64" | base64 -d)
  fi

  cat > .env <<EOF
STT_BIND=${STT_BIND:-127.0.0.1}
STT_HOST_PORT=${STT_HOST_PORT:-17860}
STT_API_TOKEN=${STT_API_TOKEN:-}
STT_GPU=${STT_GPU:-0}
STT_DEVICE=${STT_DEVICE:-}
STT_RETENTION_DAYS=${STT_RETENTION_DAYS:-7}
STT_MAX_UPLOAD_MB=${STT_MAX_UPLOAD_MB:-1024}
STT_BATCH_SIZE=${STT_BATCH_SIZE:-8}
STT_MODEL=${STT_MODEL:-openai/whisper-large-v3}
STT_UI_BIND=${STT_UI_BIND:-127.0.0.1}
STT_UI_PORT=${STT_UI_PORT:-17861}
STT_UI_USERS=${STT_UI_USERS:-}
EOF
  log ".env 작성 (GPU=${STT_GPU:-0}, api=${STT_BIND:-127.0.0.1}:${STT_HOST_PORT:-17860}, ui=${STT_UI_BIND:-127.0.0.1}:${STT_UI_PORT:-17861})"

  # 사내에 여는데 인증이 없으면 망 안의 누구나 전사본을 전부 읽고 지울 수 있다.
  # 설정 실수로 그런 상태가 배포되는 것을 여기서 막는다.
  if [ "${STT_BIND:-127.0.0.1}" = "0.0.0.0" ] && [ -z "${STT_API_TOKEN:-}" ]; then
    log "ERROR: STT_BIND=0.0.0.0 인데 STT_API_TOKEN 이 비어 있다."
    log "       CI/CD 변수에 STT_API_TOKEN 을 등록해야 한다 (protected + masked)."
    return 1
  fi
  if [ "${STT_UI_BIND:-127.0.0.1}" = "0.0.0.0" ] && [ -z "${STT_UI_USERS:-}" ]; then
    log "ERROR: STT_UI_BIND=0.0.0.0 인데 STT_UI_USERS 가 비어 있다."
    log "       UI 는 API 토큰을 대신 들고 있으므로, 로그인 없이 열면 토큰을 공개한 것과 같다."
    return 1
  fi
}

# ---------------------------------------------------------------- 2. 대기열 비우기
#
# server.py 는 재시작하면 진행 중이던 작업을 "interrupted by server restart" 로
# 실패 처리한다. 1시간 40분짜리 회의를 40분째 전사하는 중이라면 그대로 날아가고
# 올린 사람은 이유도 모른다. 그래서 큐가 빌 때까지 기다린 뒤에 교체한다.
drain() {
  if [ "$DRAIN_TIMEOUT" -le 0 ]; then
    log "drain 생략 (DRAIN_TIMEOUT=0) — 진행 중인 전사가 있으면 버려진다"
    return 0
  fi

  waited=0
  while [ "$waited" -lt "$DRAIN_TIMEOUT" ]; do
    if ! state=$(api_state); then
      log "api 가 응답하지 않는다. 처리 중인 작업이 없다고 보고 진행한다."
      return 0
    fi
    queued=$(echo "$state" | cut -d' ' -f2)
    running=$(echo "$state" | cut -d' ' -f3)
    if [ "$queued" = "0" ] && [ "$running" = "None" ]; then
      log "대기열 비어 있음. 교체를 진행한다."
      return 0
    fi
    log "전사 진행 중 (대기 ${queued}건, 실행 ${running}) — ${waited}/${DRAIN_TIMEOUT}초 대기"
    sleep "$INTERVAL"
    waited=$((waited + INTERVAL))
  done

  log "ERROR: ${DRAIN_TIMEOUT}초를 기다렸는데 아직 처리 중이다. 배포를 중단한다."
  log "       진행 중인 전사를 버리고 배포하려면 DRAIN_TIMEOUT=0 으로 다시 실행한다."
  return 1
}

# ---------------------------------------------------------------- 3. 기동 확인
#
# up -d 는 컨테이너가 떴다는 것만 알려 준다. 모델 적재는 그 뒤에 일어나므로
# ready 가 true 가 될 때까지 봐야 실제로 쓸 수 있는 상태인지 알 수 있다.
wait_ready() {
  waited=0
  while [ "$waited" -lt "$READY_TIMEOUT" ]; do
    if state=$(api_state); then
      ready=$(echo "$state" | cut -d' ' -f1)
      if [ "$ready" = "True" ]; then
        log "API 준비 완료 (${waited}초)"
        return 0
      fi
      log "모델 적재 중… ${waited}/${READY_TIMEOUT}초"
    else
      log "api 기동 대기… ${waited}/${READY_TIMEOUT}초"
    fi
    sleep "$INTERVAL"
    waited=$((waited + INTERVAL))
  done

  log "ERROR: 모델 적재가 끝나지 않았다. 로그:"
  dc logs --tail 60 api || true
  return 1
}

# ---------------------------------------------------------------- 실행

log "=== 배포 시작 (compose project=${PROJECT}) ==="
write_env
drain

log "이미지 빌드 (첫 배포는 cu128 휠 2.5GB 내려받기를 포함해 오래 걸린다)"
dc build api ui

log "컨테이너 교체"
dc up -d api ui

wait_ready

log "=== 배포 완료 ==="
dc ps
log "상태: $(api_state)"
