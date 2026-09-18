"""Batch STT for the files in input/ using openai/whisper-large-v3.

Writes two text files per input into output/_inbox/:
  <name>.txt            plain transcript
  <name>.segments.txt   same transcript with [hh:mm:ss] timestamps
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

AUDIO_EXTS = {".m4a", ".mp3", ".wav", ".flac", ".ogg", ".opus", ".aac", ".wma", ".mp4", ".webm"}
SAMPLE_RATE = 16000


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def fmt_hms(seconds):
    if seconds is None:
        return "??:??:??"
    seconds = int(seconds)
    return f"{seconds // 3600:02d}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"


def decode_audio(path):
    """Decode any ffmpeg-readable file to mono float32 PCM at 16 kHz."""
    cmd = [
        "ffmpeg", "-nostdin", "-threads", "0", "-i", str(path),
        "-f", "f32le", "-ac", "1", "-ar", str(SAMPLE_RATE), "-v", "error", "pipe:1",
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed to decode {path.name}: {proc.stderr.decode(errors='replace')}")
    return np.frombuffer(proc.stdout, dtype=np.float32)


def find_audio(indir, only=None):
    found = sorted(p for p in Path(indir).iterdir() if p.is_file() and p.suffix.lower() in AUDIO_EXTS)
    if only:
        found = [p for p in found if only.lower() in p.name.lower()]
    return found


def build_pipeline(model_id, batch_size, dtype, device, mode):
    log(f"loading {model_id} (device={device}, dtype={dtype})")
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=dtype, low_cpu_mem_usage=True, attn_implementation="sdpa"
    ).to(device)
    processor = AutoProcessor.from_pretrained(model_id)
    kwargs = {}
    if mode == "chunked":
        # Fast: independent 30s windows decoded in parallel, stitched by overlap.
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
    """The encoder runs exactly once per 30s window, in both modes.

    Counting those calls gives real progress without having to split the audio.
    """
    encoder = model.model.encoder
    original = encoder.forward
    state = {"done": 0, "start": time.time()}

    def counting_forward(*args, **kwargs):
        out = original(*args, **kwargs)
        feats = kwargs.get("input_features")
        if feats is None and args:
            feats = args[0]
        step = feats.shape[0] if hasattr(feats, "shape") else 1
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
    """Fold runs of `run_len` or more identical consecutive segments into one.

    Whisper fills near-silence by repeating a filler token on a fixed cadence
    ("아.." every 2s, a dozen times over). Real speech never repeats that
    regularly, so folding the run keeps the utterance and drops the artifact.
    Shorter runs are left alone -- people do repeat themselves twice.
    """
    out = []
    collapsed = 0
    for _, group in itertools.groupby(chunks, key=lambda c: c["text"].strip()):
        run = list(group)
        if len(run) >= run_len:
            out.append(
                {
                    "text": run[0]["text"].strip(),
                    "start": run[0]["start"],
                    "end": run[-1]["end"],
                    "runs": len(run),
                }
            )
            collapsed += len(run) - 1
        else:
            out.extend({**c, "text": c["text"].strip(), "runs": 1} for c in run)
    return out, collapsed


def transcribe(asr, path, outdir, language, beams, mode, no_speech, run_len):
    log(f"decoding audio: {path.name}")
    audio = decode_audio(path)
    duration = len(audio) / SAMPLE_RATE
    log(f"  duration {fmt_hms(duration)}")

    generate_kwargs = {"task": "transcribe", "num_beams": beams}
    if language != "auto":
        generate_kwargs["language"] = language
    if mode == "sequential":
        # Whisper's own long-form settings (see the model card): retry a window at a
        # higher temperature when the decode looks degenerate, and carry context over.
        generate_kwargs.update(
            temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
            compression_ratio_threshold=1.35,
            logprob_threshold=-1.0,
            condition_on_prev_tokens=False,
        )
        if no_speech is not None:
            generate_kwargs["no_speech_threshold"] = no_speech

    window = 25 if mode == "chunked" else 30
    total_windows = max(1, -(-int(duration) // window))
    original_forward = attach_progress(asr.model, total_windows)
    started = time.time()
    try:
        result = asr(audio, return_timestamps=True, generate_kwargs=generate_kwargs)
    finally:
        asr.model.model.encoder.forward = original_forward
        print(flush=True)

    took = time.time() - started
    log(f"  transcribed in {fmt_hms(took)} ({duration / took:.1f}x realtime)")

    raw = []
    for chunk in result.get("chunks", []):
        start, end = (chunk.get("timestamp") or (None, None))[:2]
        if chunk["text"].strip():
            raw.append({"text": chunk["text"], "start": start, "end": end})

    if run_len > 0:
        segs, collapsed = collapse_repeats(raw, run_len)
        if collapsed:
            log(f"  collapsed {collapsed} repeated filler segment(s)")
    else:
        segs = [{**c, "text": c["text"].strip(), "runs": 1} for c in raw]

    text = " ".join(s["text"] for s in segs)
    plain = outdir / f"{path.stem}.txt"
    plain.write_text(text + "\n", encoding="utf-8")

    segments = outdir / f"{path.stem}.segments.txt"
    with segments.open("w", encoding="utf-8") as fh:
        fh.write(f"# {path.name}\n# duration: {fmt_hms(duration)}\n\n")
        for seg in segs:
            fh.write(f"[{fmt_hms(seg['start'])} -> {fmt_hms(seg['end'])}] {seg['text']}\n")

    log(f"  wrote {plain.name} ({len(text)} chars) and {segments.name}")


def main():
    ap = argparse.ArgumentParser()
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
    done = {p.name[: -len(".segments.txt")] for p in outdir.rglob("*.segments.txt")}
    pending = [f for f in files if args.force or f.stem not in done]
    log(f"found {len(files)} audio file(s), {len(pending)} to transcribe")
    for f in files:
        if f not in pending:
            log(f"  skip (already done): {f.name}")
    if not pending:
        return 0

    if torch.cuda.is_available():
        device, dtype = "cuda:0", torch.float16
        log(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        device, dtype = "cpu", torch.float32
        log("WARNING: no CUDA device visible, falling back to CPU (very slow)")

    asr = build_pipeline(args.model, args.batch_size, dtype, device, args.mode)

    failures = []
    for i, path in enumerate(pending, 1):
        log(f"=== [{i}/{len(pending)}] {path.name}")
        try:
            transcribe(
                asr, path, inbox, args.language, args.beams,
                args.mode, args.no_speech_threshold, args.collapse_repeats,
            )
        except Exception as exc:  # keep going so one bad file cannot stop the batch
            failures.append((path.name, exc))
            log(f"  FAILED: {type(exc).__name__}: {exc}")

    if failures:
        log(f"done with {len(failures)} failure(s):")
        for name, exc in failures:
            log(f"  {name}: {exc}")
        return 1
    log("all files transcribed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
