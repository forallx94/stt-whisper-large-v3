"""전사 HTTP API. core.py 를 그대로 쓰고, "부르는 쪽"의 몫만 CLI 와 다르게 한다.

  - 무엇을 처리할지: 폴더 스캔이 아니라 업로드
  - 옵션을 받는 방법: 명령줄 인자가 아니라 폼 필드
  - 결과를 어디에 둘지: txt 파일이 아니라 JSON 응답 (+ 다운로드용 txt)
  - 진행 상황을 어디에 보일지: 터미널이 아니라 job 상태

전사 설정은 core.py 에만 있다. 그래야 CLI 로 돌린 결과와 API 로 돌린 결과가 같다.

## 왜 작업 큐인가

GPU 가 한 장인 것도 이유지만, 그보다 core._progress_hook 이 모델 메서드를 갈아끼우기
때문이다. 같은 엔진으로 두 전사를 동시에 돌리면 진행률 카운터가 섞이고 원복 순서가
꼬인다. 그래서 워커 스레드는 반드시 하나다. 요청은 큐에 쌓였다가 차례로 처리된다.

## 엔드포인트

  POST   /jobs              음성 업로드 → job 생성 (202)
  GET    /jobs              최근 job 목록
  GET    /jobs/{id}         상태·진행률·대기순번·결과
  GET    /jobs/{id}/text            전사 전문 (text/plain)
  GET    /jobs/{id}/segments        타임스탬프 포함 전문 (text/plain)
  DELETE /jobs/{id}         job 과 결과 삭제
  GET    /healthz           모델 적재 여부, 큐 길이
  GET    /docs              자동 생성 API 문서

## 환경 변수

  STT_DATA             작업 데이터 경로 (기본 /data)
  STT_MODEL            모델 id (기본 openai/whisper-large-v3)
  STT_BATCH_SIZE       chunked 모드 배치 크기 (기본 8)
  STT_API_TOKEN        설정하면 Authorization: Bearer <토큰> 을 요구 (기본 없음=인증 없음)
  STT_RETENTION_DAYS   업로드와 결과를 며칠 뒤 지울지 (기본 7, 0이면 지우지 않음)
  STT_MAX_UPLOAD_MB    업로드 크기 상한 (기본 1024)
  STT_DEVICE           쓸 장치를 직접 지정 (예: cuda:6). 비우면 자동(cuda:0 또는 cpu)
  STT_PORT             컨테이너가 듣는 포트 (기본 17860). 호스트 쪽 포트는 compose 에서 정한다
"""

import hashlib
import json
import os
import queue
import subprocess
import threading
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import PlainTextResponse

import core

DATA = Path(os.environ.get("STT_DATA", "/data"))
MODEL = os.environ.get("STT_MODEL", core.DEFAULT_MODEL)
BATCH_SIZE = int(os.environ.get("STT_BATCH_SIZE", "8"))
API_TOKEN = os.environ.get("STT_API_TOKEN", "").strip()
RETENTION_DAYS = float(os.environ.get("STT_RETENTION_DAYS", "7"))
MAX_UPLOAD_BYTES = int(float(os.environ.get("STT_MAX_UPLOAD_MB", "1024")) * 1024 * 1024)
# 컨테이너 안에서 듣는 포트. 호스트에 어떤 포트로 내보낼지는 docker-compose 가 정한다.
# 8000 은 흔해서 충돌이 잦으므로 덜 쓰이는 번호를 기본값으로 둔다.
PORT = int(os.environ.get("STT_PORT", "17860"))
# GPU 가 여러 장인 장비에서 특정 장치를 쓰고 싶을 때 지정한다 (예: cuda:6).
# 컨테이너에 GPU 를 한 장만 넘기면 그게 cuda:0 으로 보이므로 보통은 비워 둬도 된다.
DEVICE = os.environ.get("STT_DEVICE", "").strip() or None

JOBS_DIR = DATA / "jobs"
UPLOAD_DIR = DATA / "uploads"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------- job


@dataclass
class Job:
    """작업 하나. 그대로 JSON 으로 저장되고 그대로 응답에 실린다."""

    id: str
    filename: str
    sha256: str
    options: dict
    status: str = "queued"          # queued | running | done | failed
    created: float = 0.0
    started: Optional[float] = None
    finished: Optional[float] = None
    duration: Optional[float] = None    # 원본 길이(초). 업로드 직후 ffprobe 로 잰다
    progress: Optional[dict] = None     # {done, total, fraction, eta}
    error: Optional[str] = None
    result: Optional[dict] = None       # core.TranscriptResult 를 dict 로


def result_to_dict(r):
    """TranscriptResult 를 JSON 으로 내보낼 형태로 바꾼다.

    타임스탬프를 초 단위 실수 그대로 싣는다. txt 로 내보내면 fmt_hms 가 정수로
    잘라 버려 소수점이 사라지는데, API 는 그걸 잃을 이유가 없다.
    """
    return {
        "text": r.text,
        "segments": [asdict(s) for s in r.segments],
        "source": r.source,
        "duration": r.duration,
        "elapsed": r.elapsed,
        "collapsed": r.collapsed,
        "covered": r.covered,
        "language": r.language,
        "model": r.model,
        "mode": r.mode,
    }


def dedup_key(sha256, options):
    """같은 파일이라도 옵션이 다르면 결과가 다르므로 둘을 묶어 열쇠로 쓴다."""
    return sha256 + "|" + json.dumps(options, sort_keys=True, ensure_ascii=False)


# ---------------------------------------------------------------- 저장소


class Store:
    """job 을 메모리에 들고, 상태가 바뀔 때마다 JSON 으로 남긴다.

    진행률은 초당 여러 번 바뀌므로 디스크에 쓰지 않고 메모리에만 둔다.
    재시작하면 진행 중이던 작업은 복구할 수 없으므로 failed 로 표시한다.
    """

    def __init__(self):
        JOBS_DIR.mkdir(parents=True, exist_ok=True)
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._jobs: dict[str, Job] = {}
        self._by_key: dict[str, str] = {}
        self._rates: list[float] = []   # 최근 실시간 배속 (duration / elapsed)
        self._load()

    def _load(self):
        restored = interrupted = 0
        for path in sorted(JOBS_DIR.glob("*.json")):
            try:
                job = Job(**json.loads(path.read_text(encoding="utf-8")))
            except Exception as exc:
                log(f"WARNING: could not read {path.name}: {exc}")
                continue
            if job.status in ("queued", "running"):
                # 서버가 내려가면서 끊긴 작업이다. 다시 올릴 방법이 없으니 실패로 남긴다.
                job.status = "failed"
                job.error = "interrupted by server restart"
                job.finished = time.time()
                self._write(job)
                interrupted += 1
            self._jobs[job.id] = job
            if job.status == "done":
                self._by_key[dedup_key(job.sha256, job.options)] = job.id
                if job.result and job.result.get("elapsed"):
                    self._rates.append(job.result["duration"] / job.result["elapsed"])
            restored += 1
        self._rates = self._rates[-20:]
        if restored:
            log(f"restored {restored} job(s) from disk ({interrupted} marked interrupted)")

    def _write(self, job):
        """원자적으로 쓴다. 쓰다가 죽어도 반쪽짜리 JSON 이 남지 않게."""
        path = JOBS_DIR / f"{job.id}.json"
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(asdict(job), ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(path)

    def add(self, job):
        with self._lock:
            self._jobs[job.id] = job
            self._write(job)

    def get(self, job_id):
        with self._lock:
            return self._jobs.get(job_id)

    def find_done(self, sha256, options):
        """같은 파일을 같은 옵션으로 이미 전사한 적이 있으면 그 job 을 돌려준다."""
        with self._lock:
            job_id = self._by_key.get(dedup_key(sha256, options))
            return self._jobs.get(job_id) if job_id else None

    def all(self):
        with self._lock:
            return sorted(self._jobs.values(), key=lambda j: j.created, reverse=True)

    def update(self, job, persist=True, **fields):
        with self._lock:
            for k, v in fields.items():
                setattr(job, k, v)
            if job.status == "done":
                self._by_key[dedup_key(job.sha256, job.options)] = job.id
                if job.result and job.result.get("elapsed"):
                    self._rates.append(job.result["duration"] / job.result["elapsed"])
                    self._rates = self._rates[-20:]
            if persist:
                self._write(job)

    def remove(self, job_id):
        with self._lock:
            job = self._jobs.pop(job_id, None)
            if job is None:
                return False
            self._by_key.pop(dedup_key(job.sha256, job.options), None)
            (JOBS_DIR / f"{job_id}.json").unlink(missing_ok=True)
        for leftover in UPLOAD_DIR.glob(f"{job_id}.*"):
            leftover.unlink(missing_ok=True)
        return True

    def rate(self):
        """최근 실시간 배속의 중앙값. 표본이 없으면 None."""
        with self._lock:
            if not self._rates:
                return None
            ordered = sorted(self._rates)
            return ordered[len(ordered) // 2]


STORE = Store()
WORK: "queue.Queue[str]" = queue.Queue()
READY = threading.Event()
RUNNING_ID: Optional[str] = None
ENGINE: Optional[core.SttEngine] = None


# ---------------------------------------------------------------- 워커


def worker_loop():
    """큐에서 하나씩 꺼내 전사한다. 이 스레드는 하나뿐이다(위 '왜 작업 큐인가' 참고)."""
    global ENGINE, RUNNING_ID

    device, dtype = core.pick_device()
    if DEVICE:
        device = DEVICE
    name = core.gpu_name()
    if name:
        log(f"GPU: {name}")
    else:
        log("WARNING: no CUDA device visible, falling back to CPU (very slow)")
    log(f"loading {MODEL} (device={device}, dtype={dtype})")
    ENGINE = core.SttEngine(model_id=MODEL, batch_size=BATCH_SIZE, device=device, dtype=dtype)
    READY.set()
    log("model ready, waiting for jobs")

    while True:
        job_id = WORK.get()
        job = STORE.get(job_id)
        if job is None or job.status != "queued":
            WORK.task_done()
            continue

        RUNNING_ID = job_id
        STORE.update(job, status="running", started=time.time())
        upload = next(UPLOAD_DIR.glob(f"{job_id}.*"), None)
        try:
            if upload is None:
                raise FileNotFoundError("uploaded file is gone")
            opts = job.options
            result = ENGINE.transcribe(
                upload,
                language=opts["language"],
                beams=opts["beams"],
                mode=opts["mode"],
                no_speech=opts["no_speech_threshold"],
                run_len=opts["collapse_repeats"],
                # 임시 파일 이름이 아니라 사용자가 올린 원본 이름이 결과에 남아야 한다
                source=job.filename,
                on_decoded=lambda d: STORE.update(job, persist=False, duration=d),
                # 진행률은 초당 여러 번 바뀐다. 메모리에만 반영하고 디스크에는 쓰지 않는다.
                on_progress=lambda p: STORE.update(
                    job,
                    persist=False,
                    progress={"done": p.done, "total": p.total,
                              "fraction": round(p.fraction, 4), "eta": round(p.eta, 1)},
                ),
            )
            STORE.update(job, status="done", finished=time.time(), progress=None,
                         duration=result.duration, result=result_to_dict(result))
            log(f"{job_id} done: {job.filename} "
                f"({core.fmt_hms(result.duration)} in {core.fmt_hms(result.elapsed)})")
        except Exception as exc:
            STORE.update(job, status="failed", finished=time.time(), progress=None,
                         error=f"{type(exc).__name__}: {exc}")
            log(f"{job_id} FAILED: {job.filename}: {type(exc).__name__}: {exc}")
        finally:
            RUNNING_ID = None
            # 결과를 얻었으므로 업로드 원본은 바로 지운다. 남겨 둘 이유가 없고,
            # 남의 녹음이 서버에 쌓이는 것 자체가 위험이다.
            if upload is not None:
                upload.unlink(missing_ok=True)
            WORK.task_done()


def retention_loop():
    """보존 기한이 지난 job 을 지운다. 남의 녹음과 전사본이 무한히 쌓이지 않게 한다."""
    if RETENTION_DAYS <= 0:
        log("retention disabled (STT_RETENTION_DAYS=0)")
        return
    while True:
        cutoff = time.time() - RETENTION_DAYS * 86400
        removed = [j.id for j in STORE.all()
                   if j.status in ("done", "failed") and (j.finished or j.created) < cutoff]
        for job_id in removed:
            STORE.remove(job_id)
        if removed:
            log(f"retention: removed {len(removed)} job(s) older than {RETENTION_DAYS} day(s)")
        time.sleep(3600)


# ---------------------------------------------------------------- 응답 만들기


def queue_position(job):
    """대기열에서 몇 번째인지. 실행 중이면 0, 대기 중이 아니면 None."""
    if job.status == "running":
        return 0
    if job.status != "queued":
        return None
    waiting = [j for j in STORE.all() if j.status == "queued"]
    waiting.sort(key=lambda j: j.created)
    return next((i + 1 for i, j in enumerate(waiting) if j.id == job.id), None)


def estimate_wait(job):
    """앞에 밀린 것까지 더해 대기 시간을 초 단위로 어림한다. 근거가 없으면 None.

    최근 완료된 작업들의 실시간 배속 중앙값을 쓴다. 표본이 없으면(첫 작업이면)
    추정하지 않는다 -- 틀린 숫자를 보여주느니 모른다고 하는 편이 낫다.
    """
    rate = STORE.rate()
    if rate is None or rate <= 0:
        return None
    total = 0.0
    if job.status == "running":
        return round(job.progress["eta"], 1) if job.progress else None
    running = STORE.get(RUNNING_ID) if RUNNING_ID else None
    if running is not None and running.progress:
        total += running.progress["eta"]
    for other in sorted((j for j in STORE.all() if j.status == "queued"), key=lambda j: j.created):
        if other.id == job.id:
            break
        if other.duration:
            total += other.duration / rate
    if job.duration:
        total += job.duration / rate
    return round(total, 1)


def public(job, with_result=True):
    body = asdict(job)
    if not with_result:
        body.pop("result", None)
    body["queue_position"] = queue_position(job)
    if job.status in ("queued", "running"):
        body["estimated_seconds"] = estimate_wait(job)
    return body


# ---------------------------------------------------------------- 앱


def require_token(authorization: Optional[str] = Header(None)):
    """STT_API_TOKEN 이 설정된 경우에만 Bearer 토큰을 요구한다."""
    if not API_TOKEN:
        return
    if authorization != f"Bearer {API_TOKEN}":
        raise HTTPException(status_code=401, detail="invalid or missing bearer token")


@asynccontextmanager
async def lifespan(_app):
    """서버가 뜨자마자 워커를 띄운다. 모델 적재는 워커 스레드에서 background 로 진행되므로
    요청은 바로 받을 수 있고, 아직 준비 전이면 job 이 큐에서 기다린다."""
    threading.Thread(target=worker_loop, daemon=True, name="stt-worker").start()
    threading.Thread(target=retention_loop, daemon=True, name="stt-retention").start()
    if not API_TOKEN:
        log("WARNING: STT_API_TOKEN is not set -- the API is open to anyone who can reach it")
    yield


app = FastAPI(
    lifespan=lifespan,
    title="STT — whisper-large-v3",
    description="한국어 회의·통화 녹음에 맞춰 조정된 Whisper 전사 API. "
                "GPU 한 장을 큐로 나눠 쓰므로 작업은 차례로 처리된다.",
    version="1.0.0",
)


@app.get("/healthz")
def healthz():
    return {
        "ready": READY.is_set(),
        "model": MODEL,
        "gpu": core.gpu_name(),
        "queued": sum(1 for j in STORE.all() if j.status == "queued"),
        "running": RUNNING_ID,
        "realtime_factor": STORE.rate(),
        "retention_days": RETENTION_DAYS,
        "auth": bool(API_TOKEN),
    }


@app.post("/jobs", status_code=202, dependencies=[Depends(require_token)])
async def create_job(
    file: UploadFile = File(..., description="음성 파일"),
    language: str = Form("ko", description='ISO 코드, "auto" 면 자동 감지'),
    mode: str = Form("sequential", description="sequential(정확) | chunked(약 3배 빠름)"),
    beams: int = Form(1, description="1보다 크면 beam search"),
    no_speech_threshold: Optional[float] = Form(None, description="기본 비활성. 실제 발화를 지울 수 있음"),
    collapse_repeats: int = Form(3, description="같은 문장이 이만큼 연속되면 하나로 접음. 0이면 비활성"),
    force: bool = Form(False, description="같은 파일·같은 옵션의 기존 결과를 무시하고 다시 전사"),
):
    """음성을 올려 전사 작업을 만든다. 바로 결과가 오지 않고 job 이 돌아온다.

    긴 녹음은 수십 분이 걸리므로 동기 응답이 불가능하다. job_id 를 받아
    GET /jobs/{id} 로 진행률을 확인하고, done 이 되면 결과를 가져간다.
    """
    if mode not in ("sequential", "chunked"):
        raise HTTPException(400, "mode must be 'sequential' or 'chunked'")

    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in core.AUDIO_EXTS:
        raise HTTPException(
            415, f"unsupported extension {suffix!r}; supported: {sorted(core.AUDIO_EXTS)}")

    job_id = uuid.uuid4().hex[:16]
    # 업로드는 반드시 파일로 떨군다. 표준입력으로 흘리면 moov atom 이 끝에 있는
    # m4a/mp4 에서 ffmpeg 가 seek 하지 못해 실패한다 (core.decode_audio 주석 참고).
    target = UPLOAD_DIR / f"{job_id}{suffix}"
    digest = hashlib.sha256()
    size = 0
    try:
        with target.open("wb") as fh:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    raise HTTPException(413, f"upload exceeds {MAX_UPLOAD_BYTES // 1024 // 1024} MB")
                digest.update(chunk)
                fh.write(chunk)
    except Exception:
        target.unlink(missing_ok=True)
        raise
    if size == 0:
        target.unlink(missing_ok=True)
        raise HTTPException(400, "uploaded file is empty")

    sha256 = digest.hexdigest()
    options = {
        "language": language,
        "mode": mode,
        "beams": beams,
        "no_speech_threshold": no_speech_threshold,
        "collapse_repeats": collapse_repeats,
        "model": MODEL,
    }

    # 파일명이 아니라 내용 해시로 판정한다. 업로드 파일명은 믿을 수 없고,
    # 같은 녹음을 두 사람이 올려도 GPU 를 두 번 쓸 이유가 없다.
    if not force:
        existing = STORE.find_done(sha256, options)
        if existing is not None:
            target.unlink(missing_ok=True)
            body = public(existing)
            body["deduplicated"] = True
            return body

    job = Job(id=job_id, filename=file.filename or f"{job_id}{suffix}", sha256=sha256,
              options=options, created=time.time(), duration=probe_duration(target))
    STORE.add(job)
    WORK.put(job_id)
    log(f"{job_id} queued: {job.filename} ({size / 1024 / 1024:.1f} MB, {mode})")
    return public(job)


def probe_duration(path):
    """ffprobe 로 길이만 빠르게 잰다. 대기열 예상 시간을 내려면 미리 알아야 한다.

    실패해도 치명적이지 않다. 길이를 모르면 예상 시간만 안 나온다.
    """
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=nw=1:nk=1", str(path)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
        return float(out.stdout.strip())
    except Exception:
        return None


@app.get("/jobs", dependencies=[Depends(require_token)])
def list_jobs(limit: int = 50):
    """최근 job 목록. 응답이 커지지 않도록 결과 본문은 빼고 준다."""
    return {"jobs": [public(j, with_result=False) for j in STORE.all()[:limit]]}


@app.get("/jobs/{job_id}", dependencies=[Depends(require_token)])
def get_job(job_id: str):
    job = STORE.get(job_id)
    if job is None:
        raise HTTPException(404, "no such job")
    return public(job)


def _done_or_404(job_id):
    job = STORE.get(job_id)
    if job is None:
        raise HTTPException(404, "no such job")
    if job.status != "done":
        raise HTTPException(409, f"job is {job.status}, not done")
    return job


def _as_result(job):
    """저장된 dict 를 core 의 렌더러가 받을 수 있는 형태로 되돌린다."""
    r = job.result
    return core.TranscriptResult(
        segments=[core.Segment(**s) for s in r["segments"]],
        source=r["source"], duration=r["duration"], elapsed=r["elapsed"],
        collapsed=r["collapsed"], language=r["language"], model=r["model"], mode=r["mode"],
    )


@app.get("/jobs/{job_id}/text", response_class=PlainTextResponse,
         dependencies=[Depends(require_token)])
def get_text(job_id: str):
    """타임스탬프 없는 전문. CLI 가 만드는 <이름>.txt 와 같은 내용이다."""
    return core.render_plain(_as_result(_done_or_404(job_id)))


@app.get("/jobs/{job_id}/segments", response_class=PlainTextResponse,
         dependencies=[Depends(require_token)])
def get_segments(job_id: str):
    """타임스탬프가 붙은 전문. CLI 가 만드는 <이름>.segments.txt 와 같은 내용이다."""
    return core.render_segments(_as_result(_done_or_404(job_id)))


@app.delete("/jobs/{job_id}", dependencies=[Depends(require_token)])
def delete_job(job_id: str):
    job = STORE.get(job_id)
    if job is None:
        raise HTTPException(404, "no such job")
    if job.status in ("queued", "running"):
        raise HTTPException(409, f"job is {job.status}; wait for it to finish")
    STORE.remove(job_id)
    return {"deleted": job_id}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")
