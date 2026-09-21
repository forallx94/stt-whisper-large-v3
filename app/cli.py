"""input/ 폴더의 음성 파일을 일괄 전사해 output/_inbox/ 에 txt 로 저장한다.

전사 자체는 core.py 가 한다. 이 파일이 맡는 것은 "부르는 쪽"의 몫뿐이다.

  - 무엇을 처리할지 고르기 (폴더 스캔, 중복 판정)
  - 옵션을 받는 방법 (명령줄 인자)
  - 결과를 어디에 둘지 (output/_inbox/ 에 txt 두 개)
  - 진행 상황을 어디에 보일지 (터미널)

같은 core 를 HTTP 서버가 부르면 이 네 가지만 달라지고 전사 결과는 같다.

입력 파일 하나당 두 개의 txt 를 만든다.
  <이름>.txt            전사 결과 전문 (타임스탬프 없음)
  <이름>.segments.txt   문장별 [hh:mm:ss] 타임스탬프 포함
"""

import argparse
import sys
import time
from pathlib import Path

import core


def log(msg):
    """진행 상황을 시각과 함께 한 줄 출력한다. 컨테이너 로그로 바로 흘려보내려고 flush 한다."""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def show_progress(p):
    """진행률을 같은 줄에 덮어쓴다. core 에 넘기는 on_progress 콜백.

    캐리지 리턴으로 덮어쓰므로 로그를 파일로 받으면 한 줄에 뭉친다.
    서버는 이 자리에 job 상태를 갱신하는 콜백을 넣으면 된다.
    """
    print(
        f"    {p.fraction * 100:5.1f}%  ({p.done}/{p.total} windows)"
        f"  elapsed {core.fmt_hms(p.elapsed)}  eta {core.fmt_hms(p.eta)}",
        end="\r",
        flush=True,
    )


def find_audio(indir, only=None):
    """입력 폴더에서 처리 대상 음성 파일을 찾는다.

    하위 폴더는 보지 않는다(입력은 평평한 폴더 하나로 둔다).
    정렬해서 돌려주므로 배치 처리 순서가 실행할 때마다 같다.
    only 를 주면 파일명에 그 문자열이 포함된 것만 남긴다.

    서버에는 입력 폴더라는 개념이 없으므로 core 가 아니라 여기에 있다.
    """
    found = sorted(p for p in Path(indir).iterdir() if p.is_file() and p.suffix.lower() in core.AUDIO_EXTS)
    if only:
        found = [p for p in found if only.lower() in p.name.lower()]
    return found


def already_done(outdir):
    """이미 전사가 끝난 파일들의 이름(확장자 제외)을 모은다.

    output/ 전체를 재귀로 훑는다. 정리되어 하위 폴더로 옮겨진 결과도 "이미 완료"로 인식한다.
    Path.stem 을 쓰면 "이름.segments" 가 남으므로 접미사 길이만큼 직접 잘라낸다.

    파일명으로 판정하므로 이름을 바꾸면 다시 전사 대상이 된다. 서버에서는 업로드
    파일명을 믿을 수 없으므로 이 방식을 그대로 쓸 수 없고 내용 해시로 가야 한다.
    """
    return {p.name[: -len(".segments.txt")] for p in outdir.rglob("*.segments.txt")}


def write_result(result, outdir, stem):
    """전사 결과를 txt 두 개로 저장한다. core 가 만든 문자열을 그대로 쓴다."""
    plain = outdir / f"{stem}.txt"
    plain.write_text(core.render_plain(result), encoding="utf-8")

    segments = outdir / f"{stem}.segments.txt"
    segments.write_text(core.render_segments(result), encoding="utf-8")

    log(f"  wrote {plain.name} ({len(result.text)} chars) and {segments.name}")


def transcribe_one(engine, path, outdir, language, beams, mode, no_speech, run_len):
    """파일 하나를 전사하고 결과를 저장한다. 로그와 진행률 표시가 여기 붙는다."""
    log(f"decoding audio: {path.name}")
    try:
        result = engine.transcribe(
            path,
            language=language,
            beams=beams,
            mode=mode,
            no_speech=no_speech,
            run_len=run_len,
            on_decoded=lambda d: log(f"  duration {core.fmt_hms(d)}"),
            on_progress=show_progress,
        )
    finally:
        # 전사가 실패해도 진행률 줄을 닫아 다음 로그가 겹쳐 찍히지 않게 한다.
        print(flush=True)

    log(f"  transcribed in {core.fmt_hms(result.elapsed)}"
        f" ({result.duration / result.elapsed:.1f}x realtime)")
    if result.collapsed:
        log(f"  collapsed {result.collapsed} repeated filler segment(s)")
    write_result(result, outdir, path.stem)


def main():
    ap = argparse.ArgumentParser()
    # 기본 경로는 컨테이너 안의 마운트 지점이다. docker-compose.yml 에서
    # 호스트의 input/ 이 /audio 로, output/ 이 /output 으로 연결된다.
    ap.add_argument("--input", default="/audio")
    ap.add_argument("--output", default="/output")
    ap.add_argument("--model", default=core.DEFAULT_MODEL)
    ap.add_argument("--language", default="ko", help='ISO code, or "auto" to detect')
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--beams", type=int, default=1)
    ap.add_argument(
        "--mode",
        default="sequential",
        choices=["sequential", "chunked"],
        help="sequential = Whisper's native long-form algorithm (accurate); chunked = faster",
    )
    ap.add_argument(
        "--no-speech-threshold",
        type=float,
        default=None,
        help="drop windows whose no-speech probability exceeds this (sequential mode). "
        "Off by default: it discarded real speech at the end of a recording.",
    )
    ap.add_argument(
        "--collapse-repeats",
        type=int,
        default=3,
        help="fold runs of this many identical consecutive segments into one; 0 disables",
    )
    ap.add_argument("--only", default=None, help="only files whose name contains this substring")
    ap.add_argument("--force", action="store_true", help="re-transcribe files already in output/")
    args = ap.parse_args()

    outdir = Path(args.output)
    # 새 결과는 _inbox/ 에 쓴다. 정리해서 output/ 아래 다른 폴더로 옮겨도 재전사되지 않는다.
    inbox = outdir / "_inbox"
    inbox.mkdir(parents=True, exist_ok=True)

    files = find_audio(args.input, args.only)
    if not files:
        log(f"no audio files found in {args.input}")
        return 1

    done = already_done(outdir)
    pending = [f for f in files if args.force or f.stem not in done]
    log(f"found {len(files)} audio file(s), {len(pending)} to transcribe")
    for f in files:
        if f not in pending:
            log(f"  skip (already done): {f.name}")
    if not pending:
        return 0

    # GPU 가 보이면 fp16, 없으면 CPU + fp32 로 떨어진다. CPU 는 실사용이 어려울 만큼
    # 느리므로 경고를 남긴다. 실행 로그 첫 줄의 GPU 표시로 확인할 수 있다.
    device, dtype = core.pick_device()
    name = core.gpu_name()
    if name:
        log(f"GPU: {name}")
    else:
        log("WARNING: no CUDA device visible, falling back to CPU (very slow)")

    # 모델 로딩은 한 번만 하고 아래 루프에서 같은 엔진을 재사용한다.
    log(f"loading {args.model} (device={device}, dtype={dtype})")
    engine = core.SttEngine(
        model_id=args.model, batch_size=args.batch_size, device=device, dtype=dtype
    )

    failures = []
    for i, path in enumerate(pending, 1):
        log(f"=== [{i}/{len(pending)}] {path.name}")
        try:
            transcribe_one(
                engine, path, inbox, args.language, args.beams, args.mode,
                args.no_speech_threshold, args.collapse_repeats,
            )
        except Exception as exc:  # 파일 하나가 실패해도 배치 전체가 멈추지 않도록 계속 진행한다
            failures.append((path.name, exc))
            log(f"  FAILED: {type(exc).__name__}: {exc}")

    if failures:
        # 실패한 파일은 결과가 없으므로 다음 실행에서 자동으로 다시 대상이 된다.
        log(f"done with {len(failures)} failure(s):")
        for name, exc in failures:
            log(f"  {name}: {exc}")
        return 1
    log("all files transcribed")
    return 0


if __name__ == "__main__":
    # 실패 건수가 있으면 종료 코드 1. 셸이나 CI 에서 성공 여부를 판별할 수 있다.
    sys.exit(main())
