"""전사 웹 UI. server.py 의 HTTP API 를 호출하는 클라이언트일 뿐이다.

전사도, 모델도 여기 없다. 이 컨테이너에는 torch 조차 설치되지 않는다.
API 와 같은 프로세스에 끼워넣지 않은 이유가 두 가지 있다.

  1. UI 가 죽어도 API 는 살아 있어야 한다. 반대도 마찬가지다.
  2. UI 가 API 의 첫 사용자가 되면 API 설계가 실사용으로 검증된다.
     여기서 쓰기 불편한 API 는 다른 프로젝트에서도 불편하다.

## 환경 변수

  STT_API_URL     API 주소 (기본 http://api:17860 -- compose 네트워크 안의 서비스명)
  STT_API_TOKEN   API 토큰. UI 가 대신 들고 있으므로 브라우저 사용자는 몰라도 된다
  STT_UI_USERS    UI 로그인 계정. "user:pass,user2:pass2" 형식. 비우면 로그인 없음
  STT_UI_PORT     수신 포트 (기본 7860)
"""

import html
import os
import tempfile
import time
from pathlib import Path

import gradio as gr
import requests

API_URL = os.environ.get("STT_API_URL", "http://api:17860").rstrip("/")
API_TOKEN = os.environ.get("STT_API_TOKEN", "").strip()
UI_PORT = int(os.environ.get("STT_UI_PORT", "7860"))
POLL_SECONDS = 2
# 원본 길이와 마지막 타임스탬프가 이만큼 벌어지면 누락을 의심한다. 아래 verdict() 참고.
GAP_WARN_SECONDS = 30
# 고유 문장 비율이 이보다 낮으면 반복 붕괴를 의심한다.
UNIQUE_WARN_RATIO = 0.5

HEADERS = {"Authorization": f"Bearer {API_TOKEN}"} if API_TOKEN else {}


def fmt_hms(seconds):
    if seconds is None:
        return "??:??:??"
    seconds = int(seconds)
    return f"{seconds // 3600:02d}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"


def api_get(path, **kw):
    r = requests.get(f"{API_URL}{path}", headers=HEADERS, timeout=30, **kw)
    r.raise_for_status()
    return r


# ---------------------------------------------------------------- 결과 표시


def verdict(result):
    """전사가 조용히 실패하지 않았는지 확인해 한 줄로 알려준다.

    README 의 "결과 검증"을 사람이 눈으로 하던 것을 UI 가 대신한다. 이 검사가
    가능한 이유는 API 가 duration 과 covered 를 함께 돌려주기 때문이다 --
    txt 만 받으면 정수로 잘린 값밖에 없어 판정이 거칠어진다.
    """
    duration = result.get("duration") or 0
    covered = result.get("covered") or 0
    gap = duration - covered
    lines = []

    # 마지막 타임스탬프가 원본 길이에 한참 못 미치면 뒷부분이 통째로 빠진 것이다.
    #
    # 기준을 비율(예: 5%)로 잡으면 안 된다. 긴 파일일수록 기준이 느슨해지는데,
    # 실제로 겪은 누락은 36분 37초 회의에서 마지막 68초였다(README 참고). 비율이면
    # 그 사례가 기준을 통과해 버린다. 누락된 길이는 원본 길이와 무관하게 나쁘므로
    # 고정값으로 본다.
    if duration > 0 and gap > GAP_WARN_SECONDS:
        lines.append(
            f"⚠️ **뒷부분이 누락됐을 수 있습니다** — 원본 {fmt_hms(duration)} 인데 "
            f"마지막 타임스탬프가 {fmt_hms(covered)} 입니다 ({fmt_hms(gap)} 차이). "
            f"끝에 긴 무음이 있었다면 정상일 수 있으니 그 구간을 직접 확인해 보세요."
        )
    else:
        lines.append(f"✅ 끝까지 전사됨 — 원본 {fmt_hms(duration)}, 마지막 {fmt_hms(covered)}")

    # 고유 세그먼트 비율이 낮으면 반복 붕괴다.
    segments = result.get("segments") or []
    if segments:
        unique = len({s["text"] for s in segments})
        ratio = unique / len(segments)
        if ratio < UNIQUE_WARN_RATIO:
            lines.append(
                f"⚠️ **반복이 많습니다** — 세그먼트 {len(segments)}개 중 고유 문장이 "
                f"{unique}개뿐입니다. 반복 붕괴일 수 있습니다."
            )

    collapsed = result.get("collapsed") or 0
    if collapsed:
        lines.append(f"ℹ️ 반복되던 필러 세그먼트 {collapsed}개를 접었습니다(내용 손실 없음).")

    return "\n\n".join(lines)


def save_temp(name, content):
    """다운로드 버튼에 물릴 파일을 만든다."""
    path = Path(tempfile.gettempdir()) / name
    path.write_text(content, encoding="utf-8")
    return str(path)


def render_segments(result):
    """내려받기·복사용 원문. CLI 가 만드는 .segments.txt 와 같은 형식이다."""
    lines = []
    for s in result.get("segments", []):
        lines.append(f"[{fmt_hms(s['start'])} -> {fmt_hms(s['end'])}] {s['text']}")
    return "\n".join(lines)


def segments_html(result):
    """타임스탬프본을 읽기 좋게 그린다.

    textarea 로 보여 주면 두 가지가 불편하다. 내용만큼 높이가 늘어나 박스가 아니라
    페이지가 스크롤되고, 긴 문장이 줄바꿈되면서 줄머리의 타임스탬프가 본문에 묻힌다.
    한 시간짜리 회의는 세그먼트가 천 줄이 넘어 둘 다 심해진다.

    그래서 타임스탬프를 고정폭 열로 빼고, 본문만 접히게 하고, 전체를 고정 높이
    스크롤 영역에 담는다.
    """
    segments = result.get("segments") or []
    if not segments:
        return ""
    rows = []
    for s in segments:
        stamp = fmt_hms(s["start"])
        # 접힌 세그먼트에는 몇 개가 접혔는지 표시한다. 환각 구간을 눈으로 찾을 수 있다.
        runs = s.get("runs") or 1
        badge = f'<span class="stt-runs" title="같은 문장 {runs}개를 접었습니다">×{runs}</span>' if runs > 1 else ""
        rows.append(
            f'<div class="stt-row">'
            f'<span class="stt-time">{stamp}</span>'
            f'<span class="stt-text">{html.escape(s["text"])}{badge}</span>'
            f'</div>'
        )
    return f'<div class="stt-seg">{"".join(rows)}</div>'


def download(path=None, label=None):
    """다운로드 버튼의 상태를 만든다.

    gr.File 출력은 파일명이 링크가 아니라 오른쪽 크기 표시("3.1 KB")가 링크라서,
    파일명을 눌러도 아무 일이 없다. gr.DownloadButton 은 버튼 전체가 눌리므로
    파일명을 버튼 글자로 넣으면 보이는 그대로 누를 수 있다.
    """
    if path is None:
        return gr.DownloadButton(value=None, label=label or "내려받기", interactive=False)
    return gr.DownloadButton(value=path, label=f"⬇  {label}", interactive=True)


def safe_stem(filename):
    """다운로드 파일명에 쓸 수 있도록 경로 구분자 등을 걷어낸다."""
    stem = Path(filename or "transcript").stem
    return "".join(c for c in stem if c not in '\\/:*?"<>|').strip() or "transcript"


# ---------------------------------------------------------------- 전사 실행


def transcribe(file_path, language, mode, beams, no_speech, collapse, force):
    """업로드하고, 끝날 때까지 상태를 받아 화면을 갱신한다.

    제너레이터라서 yield 할 때마다 화면이 바뀐다. 긴 녹음은 수십 분이 걸리므로
    대기순번과 진행률을 계속 보여 주지 않으면 사람들은 창을 닫아 버린다.
    """
    blank = ("", "", "", download(), download(), "")
    if not file_path:
        yield "음성 파일을 선택하세요.", *blank
        return

    name = Path(file_path).name
    try:
        with open(file_path, "rb") as fh:
            data = {
                "language": language,
                "mode": mode,
                "beams": int(beams),
                "collapse_repeats": int(collapse),
                "force": bool(force),
            }
            if no_speech is not None and no_speech > 0:
                data["no_speech_threshold"] = float(no_speech)
            # 업로드는 오래 걸릴 수 있으므로 넉넉히 기다린다.
            r = requests.post(f"{API_URL}/jobs", headers=HEADERS,
                              files={"file": (name, fh)}, data=data, timeout=1800)
        if r.status_code >= 400:
            yield f"❌ 업로드 실패 ({r.status_code}) — {r.text[:300]}", *blank
            return
        job = r.json()
    except requests.RequestException as exc:
        yield f"❌ API 에 연결할 수 없습니다 — {exc}", *blank
        return

    job_id = job["id"]
    if job.get("deduplicated"):
        yield (f"♻️ 같은 파일을 같은 옵션으로 이미 전사한 기록이 있어 그 결과를 가져왔습니다 "
               f"(`{job_id}`). 다시 전사하려면 **강제 재전사**를 켜세요."), *blank

    while job["status"] in ("queued", "running"):
        if job["status"] == "queued":
            pos = job.get("queue_position")
            est = job.get("estimated_seconds")
            msg = f"⏳ 대기 중 — 앞에 {pos - 1 if pos else 0}건"
            if est:
                msg += f", 예상 {fmt_hms(est)} 후 시작"
        else:
            p = job.get("progress")
            if p:
                msg = (f"🎧 전사 중 — {p['fraction'] * 100:.1f}%  "
                       f"({p['done']}/{p['total']} 구간)  남은 시간 약 {fmt_hms(p['eta'])}")
            else:
                msg = "🎧 오디오를 읽는 중…"
        yield msg, *blank
        time.sleep(POLL_SECONDS)
        try:
            job = api_get(f"/jobs/{job_id}").json()
        except requests.RequestException as exc:
            yield f"❌ 상태를 확인할 수 없습니다 — {exc}", *blank
            return

    if job["status"] == "failed":
        yield f"❌ 전사 실패 — {job.get('error')}", *blank
        return

    result = job["result"]
    stem = safe_stem(job["filename"])
    plain = result["text"] + "\n"
    segs = render_segments(result)

    speed = result["duration"] / result["elapsed"] if result["elapsed"] else 0
    header = (f"### 완료 — {job['filename']}\n\n"
              f"길이 {fmt_hms(result['duration'])} · 전사 {fmt_hms(result['elapsed'])} "
              f"({speed:.1f}배속) · {result['mode']} · `{job_id}`\n\n"
              + verdict(result))

    yield (header, plain, segments_html(result), segs,
           download(save_temp(f"{stem}.txt", plain), f"{stem}.txt"),
           download(save_temp(f"{stem}.segments.txt", segs + "\n"), f"{stem}.segments.txt"),
           job_id)


# ---------------------------------------------------------------- 최근 작업


def recent_jobs():
    try:
        jobs = api_get("/jobs", params={"limit": 30}).json()["jobs"]
    except requests.RequestException as exc:
        return [[f"API 에 연결할 수 없습니다: {exc}", "", "", ""]]
    rows = []
    for j in jobs:
        when = time.strftime("%m-%d %H:%M", time.localtime(j["created"]))
        rows.append([j["id"], when, j["filename"],
                     j["status"], fmt_hms(j.get("duration"))])
    return rows or [["(아직 작업이 없습니다)", "", "", "", ""]]


def load_job(job_id):
    job_id = (job_id or "").strip()
    if not job_id:
        return "작업 ID 를 입력하세요.", "", "", "", download(), download()
    try:
        job = api_get(f"/jobs/{job_id}").json()
    except requests.RequestException as exc:
        return f"❌ 불러올 수 없습니다 — {exc}", "", "", "", download(), download()
    if job["status"] != "done":
        return f"이 작업은 아직 `{job['status']}` 상태입니다.", "", "", "", download(), download()

    result = job["result"]
    stem = safe_stem(job["filename"])
    plain = result["text"] + "\n"
    segs = render_segments(result)
    header = (f"### {job['filename']}\n\n길이 {fmt_hms(result['duration'])} · "
              f"{result['mode']} · `{job_id}`\n\n" + verdict(result))
    return (header, plain, segments_html(result), segs,
            download(save_temp(f"{stem}.txt", plain), f"{stem}.txt"),
            download(save_temp(f"{stem}.segments.txt", segs + "\n"), f"{stem}.segments.txt"))


def health():
    try:
        h = api_get("/healthz").json()
    except requests.RequestException as exc:
        return f"❌ API 에 연결할 수 없습니다 ({API_URL}) — {exc}"
    if not h["ready"]:
        return "⏳ 모델을 올리는 중입니다. 잠시 뒤 다시 확인하세요. (올려 둔 작업은 큐에서 기다립니다)"
    bits = [f"✅ 준비됨 · {h['gpu'] or 'CPU'}", f"대기 {h['queued']}건"]
    if h.get("realtime_factor"):
        bits.append(f"최근 {h['realtime_factor']:.0f}배속")
    return " · ".join(bits)


# ---------------------------------------------------------------- 화면


# 색은 전부 gradio 변수로 쓴다. 밝은 테마와 어두운 테마 양쪽에서 읽혀야 한다.
CSS = """
.stt-seg {
  max-height: 60vh;          /* 화면 높이에 맞춘다. 내용이 길어도 페이지가 아니라 여기가 스크롤된다 */
  overflow-y: auto;
  border: 1px solid var(--border-color-primary);
  border-radius: var(--radius-lg);
  background: var(--background-fill-primary);
  padding: 4px 0;
  font-size: var(--text-md);
  line-height: 1.7;
}
.stt-row {
  display: flex;             /* 타임스탬프를 별도 열로 고정한다. 본문이 줄바꿈돼도 묻히지 않는다 */
  gap: 12px;
  padding: 3px 14px;
  align-items: baseline;
}
.stt-row:nth-child(even) { background: var(--background-fill-secondary); }
.stt-time {
  flex: 0 0 auto;            /* 절대 줄어들거나 줄바꿈되지 않게 한다 */
  white-space: nowrap;
  font-family: var(--font-mono);
  font-size: var(--text-sm);
  color: var(--body-text-color-subdued);
  user-select: all;          /* 한 번 클릭으로 시각 전체가 선택된다 */
}
.stt-text { flex: 1 1 auto; color: var(--body-text-color); white-space: pre-wrap; }
.stt-runs {
  margin-left: 6px;
  padding: 0 5px;
  border-radius: var(--radius-sm);
  background: var(--background-fill-secondary);
  border: 1px solid var(--border-color-primary);
  font-size: var(--text-xs);
  color: var(--body-text-color-subdued);
}
"""

with gr.Blocks(title="STT — whisper-large-v3", theme=gr.themes.Soft(), css=CSS) as demo:
    gr.Markdown(
        "# 음성 전사\n"
        "한국어 회의·통화 녹음에 맞춰 조정된 whisper-large-v3 로 전사합니다. "
        "GPU 한 장을 나눠 쓰므로 작업은 올린 순서대로 처리됩니다."
    )
    status_bar = gr.Markdown(value=health)

    with gr.Tab("전사"):
        with gr.Row():
            with gr.Column(scale=1):
                # gr.Audio 가 아니라 gr.File 을 쓴다. Audio 는 파일을 다시 인코딩해서
                # 내용 해시가 달라지고, 그러면 서버의 중복 판정이 매번 빗나간다.
                upload = gr.File(label="음성 파일", type="filepath",
                                 file_types=[".m4a", ".mp3", ".wav", ".flac", ".ogg",
                                             ".opus", ".aac", ".wma", ".mp4", ".webm"])
                language = gr.Dropdown(["ko", "en", "ja", "zh", "auto"], value="ko",
                                       label="언어", info="auto 면 모델이 직접 감지합니다")
                mode = gr.Radio(["sequential", "chunked"], value="sequential", label="모드",
                                info="sequential 은 정확하고 느립니다. chunked 는 약 3배 빠르지만 "
                                     "타임스탬프가 거칠고 드물게 글자가 깨집니다")
                with gr.Accordion("고급 설정", open=False):
                    beams = gr.Slider(1, 5, value=1, step=1, label="beam 수",
                                      info="1보다 크면 조금 정확해지고 그만큼 느려집니다")
                    collapse = gr.Slider(0, 10, value=3, step=1, label="반복 접기",
                                         info="같은 문장이 이만큼 연속되면 하나로 접습니다. 0이면 끕니다")
                    no_speech = gr.Slider(0, 1, value=0, step=0.05, label="무음 임계값",
                                          info="0이면 비활성(권장). 켜면 실제 발화가 지워질 수 있습니다")
                    force = gr.Checkbox(False, label="강제 재전사",
                                        info="같은 파일의 기존 결과를 무시하고 다시 전사합니다")
                run = gr.Button("전사 시작", variant="primary")

            with gr.Column(scale=2):
                out_status = gr.Markdown()
                with gr.Tabs():
                    with gr.Tab("전문"):
                        # max_lines 를 주지 않으면 내용만큼 높이가 늘어나 페이지가 스크롤된다.
                        # lines 와 같은 값을 줘야 상자 안에서 스크롤된다.
                        out_text = gr.Textbox(lines=20, max_lines=20,
                                              show_copy_button=True, label=None)
                    with gr.Tab("타임스탬프"):
                        out_seg_html = gr.HTML()
                        with gr.Accordion("원문 그대로 보기 (복사용)", open=False):
                            out_segments = gr.Textbox(lines=14, max_lines=14,
                                                      show_copy_button=True, label=None)
                with gr.Row():
                    # gr.File 은 파일명이 아니라 옆의 크기 표시가 링크라 누를 곳을 찾기
                    # 어렵다. DownloadButton 은 버튼 전체가 눌리고, 글자에 파일명을 넣어
                    # 무엇이 받아지는지 보이게 한다.
                    dl_text = gr.DownloadButton("전문 (.txt)", interactive=False, size="md")
                    dl_segments = gr.DownloadButton("타임스탬프본 (.segments.txt)",
                                                    interactive=False, size="md")
                out_id = gr.Textbox(visible=False)

    with gr.Tab("최근 작업"):
        gr.Markdown("서버에 남아 있는 작업입니다. 보존 기한이 지나면 자동으로 사라집니다.")
        refresh = gr.Button("새로 고침")
        table = gr.Dataframe(headers=["작업 ID", "시각", "파일명", "상태", "길이"],
                             datatype=["str"] * 5, interactive=False, wrap=True)
        with gr.Row():
            job_input = gr.Textbox(label="작업 ID", placeholder="위 표에서 ID 를 복사해 넣으세요", scale=3)
            load = gr.Button("결과 불러오기", scale=1)
        hist_status = gr.Markdown()
        hist_text = gr.Textbox(lines=14, max_lines=14, show_copy_button=True, label="전문")
        hist_seg_html = gr.HTML()
        with gr.Accordion("타임스탬프 원문 (복사용)", open=False):
            hist_segments = gr.Textbox(lines=14, max_lines=14, show_copy_button=True, label=None)
        with gr.Row():
            hist_dl_text = gr.DownloadButton("전문 (.txt)", interactive=False, size="md")
            hist_dl_segments = gr.DownloadButton("타임스탬프본 (.segments.txt)",
                                                 interactive=False, size="md")

    run.click(
        transcribe,
        inputs=[upload, language, mode, beams, no_speech, collapse, force],
        outputs=[out_status, out_text, out_seg_html, out_segments,
                 dl_text, dl_segments, out_id],
        # 서버가 어차피 한 번에 하나씩 처리한다. UI 에서 동시 실행을 열어 두면
        # 큐에 쌓이는 것을 사용자가 더 헷갈려한다.
        concurrency_limit=4,
    )
    refresh.click(recent_jobs, outputs=table)
    load.click(load_job, inputs=job_input,
               outputs=[hist_status, hist_text, hist_seg_html, hist_segments,
                        hist_dl_text, hist_dl_segments])
    demo.load(recent_jobs, outputs=table)


def ui_auth():
    """STT_UI_USERS="user:pass,user2:pass2" 를 gradio 가 받는 형태로 바꾼다."""
    raw = os.environ.get("STT_UI_USERS", "").strip()
    if not raw:
        return None
    pairs = []
    for entry in raw.split(","):
        if ":" in entry:
            user, _, password = entry.partition(":")
            pairs.append((user.strip(), password.strip()))
    return pairs or None


if __name__ == "__main__":
    auth = ui_auth()
    if auth is None:
        print("WARNING: STT_UI_USERS is not set -- anyone who can reach this page can use it",
              flush=True)
    demo.queue(default_concurrency_limit=4).launch(
        server_name="0.0.0.0", server_port=UI_PORT, auth=auth, show_api=False,
    )
