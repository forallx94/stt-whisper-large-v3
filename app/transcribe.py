"""input/ 폴더의 음성 파일을 openai/whisper-large-v3 로 일괄 전사한다.

입력 파일 하나당 두 개의 txt를 output/_inbox/ 에 만든다.
  <이름>.txt            전사 결과 전문 (타임스탬프 없음)
  <이름>.segments.txt   문장별 [hh:mm:ss] 타임스탬프 포함

기본값은 한국어 회의·통화 녹음(수십 분~수 시간)에 맞춰 조정되어 있다.
모델 카드 권장값에서 의도적으로 벗어난 곳이 두 군데 있으며, 그 근거는
README 의 "모델 카드 권장값에서 의도적으로 벗어난 두 가지"에 있다.
"""

import argparse
import itertools
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

# ffmpeg 가 읽을 수 있는 형식 중 실제로 들어오는 것들. 확장자로만 1차 선별한다.
AUDIO_EXTS = {".m4a", ".mp3", ".wav", ".flac", ".ogg", ".opus", ".aac", ".wma", ".mp4", ".webm"}
# Whisper 가 요구하는 입력 샘플레이트. 원본이 무엇이든 여기에 맞춰 리샘플링한다.
SAMPLE_RATE = 16000


def log(msg):
    """진행 상황을 시각과 함께 한 줄 출력한다. 컨테이너 로그로 바로 흘려보내려고 flush 한다."""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def fmt_hms(seconds):
    """초를 hh:mm:ss 로 바꾼다. 타임스탬프가 없는 세그먼트를 위해 None 도 받는다."""
    if seconds is None:
        return "??:??:??"
    seconds = int(seconds)
    return f"{seconds // 3600:02d}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"


def decode_audio(path):
    """ffmpeg 로 읽을 수 있는 파일을 16kHz 모노 float32 PCM 으로 디코딩한다.

    디코딩을 ffmpeg 에 맡기므로 파이썬 쪽에 형식별 코덱이 필요 없고,
    지원 형식은 컨테이너에 설치된 ffmpeg 가 읽을 수 있는 범위와 같아진다.
    결과는 임시 파일로 떨구지 않고 파이프(pipe:1)로 바로 받는다.
    """
    cmd = [
        # -nostdin: 배치 루프 도중 ffmpeg 가 표준입력을 가로채지 않도록 막는다.
        # -threads 0: 디코딩 스레드 수를 ffmpeg 가 알아서 정하게 한다.
        "ffmpeg", "-nostdin", "-threads", "0", "-i", str(path),
        # f32le / ac 1 / ar 16000: Whisper 입력 규격(32비트 부동소수 리틀엔디언, 모노, 16kHz).
        "-f", "f32le", "-ac", "1", "-ar", str(SAMPLE_RATE), "-v", "error", "pipe:1",
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        # 호출한 쪽(배치 루프)에서 잡아 이 파일만 건너뛰고 나머지를 계속 처리한다.
        raise RuntimeError(f"ffmpeg failed to decode {path.name}: {proc.stderr.decode(errors='replace')}")
    return np.frombuffer(proc.stdout, dtype=np.float32)


def find_audio(indir, only=None):
    """입력 폴더에서 처리 대상 음성 파일을 찾는다.

    하위 폴더는 보지 않는다(입력은 평평한 폴더 하나로 둔다).
    정렬해서 돌려주므로 배치 처리 순서가 실행할 때마다 같다.
    only 를 주면 파일명에 그 문자열이 포함된 것만 남긴다.
    """
    found = sorted(p for p in Path(indir).iterdir() if p.is_file() and p.suffix.lower() in AUDIO_EXTS)
    if only:
        found = [p for p in found if only.lower() in p.name.lower()]
    return found


def build_pipeline(model_id, batch_size, dtype, device, mode):
    """모델을 올리고 transformers ASR 파이프라인을 만든다. 배치 시작 때 한 번만 부른다.

    mode 가 파이프라인 생성 시점에 박히는 점에 주의한다. chunked 설정은 파이프라인
    구성 자체를 바꾸므로, 한 번 만든 뒤에는 전사 요청마다 모드를 바꿀 수 없다.
    """
    log(f"loading {model_id} (device={device}, dtype={dtype})")
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        # low_cpu_mem_usage: 가중치를 CPU 메모리에 통째로 펼치지 않고 올려 최초 로딩 부담을 줄인다.
        # sdpa: PyTorch 내장 스케일드 닷프로덕트 어텐션. 별도 패키지 설치 없이 쓸 수 있다.
        model_id, torch_dtype=dtype, low_cpu_mem_usage=True, attn_implementation="sdpa"
    ).to(device)
    processor = AutoProcessor.from_pretrained(model_id)
    kwargs = {}
    if mode == "chunked":
        # 빠른 쪽: 30초 창을 서로 독립적으로 병렬 디코딩하고 겹치는 부분으로 이어 붙인다.
        # 앞뒤 문맥을 보지 않으므로 타임스탬프가 거칠고 이음매에서 드물게 글자가 깨진다.
        kwargs = {"chunk_length_s": 30, "stride_length_s": 5, "batch_size": batch_size}
    return pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=dtype,
        device=device,
        **kwargs,
    )


def attach_progress(model, total_windows):
    """인코더 호출을 세어 진행률을 표시한다. 원래 forward 를 돌려주므로 호출한 쪽에서 복원해야 한다.

    두 모드 모두 30초 창 하나당 인코더가 정확히 한 번 돈다. 그래서 인코더 호출을
    세면 오디오를 미리 쪼개 보지 않고도 실제 진행률을 알 수 있다.

    주의: 모델 객체의 메서드를 갈아끼우는 방식이라, 같은 모델로 두 전사를 동시에
    돌릴 수 없다. 카운터가 서로 섞이고 복원 순서도 꼬인다. 서버로 올릴 때
    워커를 하나만 두고 큐로 직렬화해야 하는 이유 중 하나다.
    """
    encoder = model.model.encoder
    original = encoder.forward
    state = {"done": 0, "start": time.time()}

    def counting_forward(*args, **kwargs):
        out = original(*args, **kwargs)
        # 호출부에 따라 입력이 키워드로도 위치 인자로도 들어오므로 양쪽을 본다.
        feats = kwargs.get("input_features")
        if feats is None and args:
            feats = args[0]
        # chunked 모드는 창 여러 개를 한 배치로 넘기므로 배치 크기만큼 한 번에 진행한다.
        step = feats.shape[0] if hasattr(feats, "shape") else 1
        # total_windows 는 어림값이라 실제 호출이 더 많을 수 있다. 100%를 넘기지 않도록 자른다.
        state["done"] = min(state["done"] + step, total_windows)
        frac = state["done"] / total_windows
        elapsed = time.time() - state["start"]
        eta = elapsed / frac - elapsed if frac > 0 else 0
        print(
            f"    {frac * 100:5.1f}%  ({state['done']}/{total_windows} windows)"
            f"  elapsed {fmt_hms(elapsed)}  eta {fmt_hms(eta)}",
            end="\r",
            flush=True,
        )
        return out

    encoder.forward = counting_forward
    return original


def collapse_repeats(chunks, run_len):
    """같은 문장이 run_len 회 이상 연달아 나오면 하나로 접는다.

    Whisper 는 무음에 가까운 구간을 필러 토큰의 규칙적인 반복으로 채운다
    ("아.." 가 2초 간격으로 열두 번). 실제 발화는 그렇게 규칙적으로 반복되지
    않으므로, 연속 구간을 접으면 발화는 남기고 환각만 걷어낼 수 있다.
    짧은 반복은 건드리지 않는다 -- 사람은 같은 말을 두 번쯤 반복한다.

    접을 때 시작은 첫 세그먼트, 끝은 마지막 세그먼트의 시각을 쓴다. 반복이
    차지하던 시간 구간이 그대로 보존되므로 뒤쪽 타임스탬프가 밀리지 않는다.
    """
    out = []
    collapsed = 0
    # groupby 는 연속된 것만 묶는다. 떨어져서 나온 같은 문장은 각각 살아남는다.
    for _, group in itertools.groupby(chunks, key=lambda c: c["text"].strip()):
        run = list(group)
        if len(run) >= run_len:
            out.append(
                {
                    "text": run[0]["text"].strip(),
                    "start": run[0]["start"],
                    "end": run[-1]["end"],
                    # 몇 개를 접었는지 남긴다. 나중에 환각 구간을 되짚을 때 쓸 수 있다.
                    "runs": len(run),
                }
            )
            collapsed += len(run) - 1
        else:
            out.extend({**c, "text": c["text"].strip(), "runs": 1} for c in run)
    return out, collapsed


def transcribe(asr, path, outdir, language, beams, mode, no_speech, run_len):
    """파일 하나를 전사해 outdir 에 txt 두 개로 저장한다."""
    log(f"decoding audio: {path.name}")
    audio = decode_audio(path)
    # ffmpeg 가 16kHz 모노로 맞춰 줬으므로 샘플 수를 나누면 그대로 길이(초)가 된다.
    duration = len(audio) / SAMPLE_RATE
    log(f"  duration {fmt_hms(duration)}")

    generate_kwargs = {"task": "transcribe", "num_beams": beams}
    # auto 면 language 를 아예 넘기지 않는다. 그래야 모델이 직접 언어를 감지한다.
    if language != "auto":
        generate_kwargs["language"] = language
    if mode == "sequential":
        # Whisper 고유의 long-form 설정(모델 카드 참고). 디코드가 무너진 것으로 보이면
        # 온도를 올려 그 창만 다시 시도한다.
        generate_kwargs.update(
            # 0.0(그리디)부터 차례로 올린다. 아래 두 임계값을 통과하면 그 단계에서 멈춘다.
            temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
            # 압축률이 이보다 높으면 같은 말이 반복된 것으로 보고 재시도한다.
            compression_ratio_threshold=1.35,
            # 평균 로그확률이 이보다 낮으면 신뢰할 수 없는 디코드로 보고 재시도한다.
            logprob_threshold=-1.0,
            # 모델 카드는 True 를 권하지만 껐다. True 로 두면 반복 붕괴가 일어났다.
            # 14분 통화에서 세그먼트 810개가 나왔는데 고유 문장은 35개뿐이었다. README 참고.
            condition_on_prev_tokens=False,
        )
        # 기본적으로 비활성이다. 0.6 으로 두니 36분 회의의 마지막 68초가 통째로 사라졌다.
        # 실제 발화를 조용히 버리는 억제라, 필요한 사람만 명시적으로 켜도록 했다. README 참고.
        if no_speech is not None:
            generate_kwargs["no_speech_threshold"] = no_speech

    # 진행률 분모용 어림값이다. 창 하나가 실제로 몇 초씩 나아가는지는 모드와 stride
    # 설정에 따라 달라지므로 정확하지 않을 수 있다. 표시에만 쓰이고 전사 결과에는
    # 영향을 주지 않는다.
    window = 25 if mode == "chunked" else 30
    # -(-a // b) 는 올림 나눗셈. 끝에 남는 자투리 창도 한 개로 센다.
    total_windows = max(1, -(-int(duration) // window))
    original_forward = attach_progress(asr.model, total_windows)
    started = time.time()
    try:
        result = asr(audio, return_timestamps=True, generate_kwargs=generate_kwargs)
    finally:
        # 전사가 실패해도 반드시 원래 forward 로 되돌린다. 안 그러면 같은 파이프라인을
        # 재사용하는 다음 파일이 이전 카운터를 물고 간다.
        asr.model.model.encoder.forward = original_forward
        print(flush=True)

    took = time.time() - started
    log(f"  transcribed in {fmt_hms(took)} ({duration / took:.1f}x realtime)")

    raw = []
    for chunk in result.get("chunks", []):
        # 타임스탬프를 못 붙인 세그먼트는 timestamp 가 None 이거나 끝값이 비어 있다.
        start, end = (chunk.get("timestamp") or (None, None))[:2]
        # 공백뿐인 세그먼트는 버린다. 반복 판정과 출력 양쪽에서 잡음이 된다.
        if chunk["text"].strip():
            raw.append({"text": chunk["text"], "start": start, "end": end})

    if run_len > 0:
        segs, collapsed = collapse_repeats(raw, run_len)
        if collapsed:
            log(f"  collapsed {collapsed} repeated filler segment(s)")
    else:
        # 접기를 꺼도 공백 정리와 runs 키는 맞춰 둔다. 이후 처리가 두 형태를 구분하지 않도록.
        segs = [{**c, "text": c["text"].strip(), "runs": 1} for c in raw]

    text = " ".join(s["text"] for s in segs)
    plain = outdir / f"{path.stem}.txt"
    plain.write_text(text + "\n", encoding="utf-8")

    segments = outdir / f"{path.stem}.segments.txt"
    with segments.open("w", encoding="utf-8") as fh:
        # duration 헤더는 검증용이다. 마지막 타임스탬프가 이 값에 한참 못 미치면
        # 뒷부분이 통째로 누락된 것이다. README 의 "결과 검증" 참고.
        fh.write(f"# {path.name}\n# duration: {fmt_hms(duration)}\n\n")
        for seg in segs:
            fh.write(f"[{fmt_hms(seg['start'])} -> {fmt_hms(seg['end'])}] {seg['text']}\n")

    log(f"  wrote {plain.name} ({len(text)} chars) and {segments.name}")


def main():
    ap = argparse.ArgumentParser()
    # 기본 경로는 컨테이너 안의 마운트 지점이다. docker-compose.yml 에서
    # 호스트의 input/ 이 /audio 로, output/ 이 /output 으로 연결된다.
    ap.add_argument("--input", default="/audio")
    ap.add_argument("--output", default="/output")
    ap.add_argument("--model", default="openai/whisper-large-v3")
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

    # output/ 전체를 재귀로 훑는다. 정리되어 하위 폴더로 옮겨진 결과도 "이미 완료"로 인식한다.
    # Path.stem 을 쓰면 "이름.segments" 가 남으므로 접미사 길이만큼 직접 잘라낸다.
    done = {p.name[: -len(".segments.txt")] for p in outdir.rglob("*.segments.txt")}
    pending = [f for f in files if args.force or f.stem not in done]
    log(f"found {len(files)} audio file(s), {len(pending)} to transcribe")
    for f in files:
        if f not in pending:
            log(f"  skip (already done): {f.name}")
    if not pending:
        return 0

    # GPU 가 보이면 fp16, 없으면 CPU + fp32 로 떨어진다. CPU 는 실사용이 어려울 만큼
    # 느리므로 경고를 남긴다. 실행 로그 첫 줄의 GPU 표시로 확인할 수 있다.
    if torch.cuda.is_available():
        device, dtype = "cuda:0", torch.float16
        log(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        device, dtype = "cpu", torch.float32
        log("WARNING: no CUDA device visible, falling back to CPU (very slow)")

    # 모델 로딩은 한 번만 하고 아래 루프에서 같은 파이프라인을 재사용한다.
    asr = build_pipeline(args.model, args.batch_size, dtype, device, args.mode)

    failures = []
    for i, path in enumerate(pending, 1):
        log(f"=== [{i}/{len(pending)}] {path.name}")
        try:
            transcribe(
                asr, path, inbox, args.language, args.beams,
                args.mode, args.no_speech_threshold, args.collapse_repeats,
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
