"""전사 엔진. 화면에 출력하지 않고, 파일도 만들지 않는다.

CLI(cli.py)와 앞으로 붙일 HTTP 서버가 이 모듈을 함께 쓴다. 그래서 두 가지를 지킨다.

  - 아무것도 print 하지 않는다. 진행 상황은 on_progress 콜백으로 넘긴다.
  - 파일을 쓰지 않는다. 결과는 TranscriptResult 로 돌려준다.

결과를 어디에 둘지, 진행률을 어디에 보여줄지는 부르는 쪽이 정한다. 전사를 어떻게
할지만 여기서 정한다. 튜닝값이 이 파일 한 곳에만 있어야 CLI 와 서버의 결과가
갈라지지 않는다.

모델 카드 권장값에서 의도적으로 벗어난 두 곳이 _generate_kwargs() 안에 있다.
근거는 README 의 "모델 카드 권장값에서 의도적으로 벗어난 두 가지" 참고.
"""

import itertools
import subprocess
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

# ffmpeg 가 읽을 수 있는 형식 중 실제로 들어오는 것들. 확장자로만 1차 선별한다.
# 서버도 업로드를 거를 때 같은 목록을 써야 하므로 여기에 둔다.
AUDIO_EXTS = {".m4a", ".mp3", ".wav", ".flac", ".ogg", ".opus", ".aac", ".wma", ".mp4", ".webm"}
# Whisper 가 요구하는 입력 샘플레이트. 원본이 무엇이든 여기에 맞춰 리샘플링한다.
SAMPLE_RATE = 16000
DEFAULT_MODEL = "openai/whisper-large-v3"


# ---------------------------------------------------------------- 결과 자료구조


@dataclass
class Segment:
    """전사된 문장 하나."""

    text: str
    # 타임스탬프를 붙이지 못한 세그먼트가 있어 None 을 허용한다. 초 단위 실수다.
    start: float | None
    end: float | None
    # collapse_repeats 가 이 세그먼트로 접은 원본 개수. 1이면 접히지 않았다.
    runs: int = 1


@dataclass
class TranscriptResult:
    """전사 결과 전체. 파일로 쓰든 JSON 으로 내보내든 여기서 출발한다."""

    segments: list[Segment] = field(default_factory=list)
    source: str = ""        # 원본 파일명
    duration: float = 0.0   # 원본 길이(초)
    elapsed: float = 0.0    # 전사에 걸린 시간(초)
    collapsed: int = 0      # 접힌 세그먼트 총수
    language: str = ""
    model: str = ""
    mode: str = ""

    @property
    def text(self) -> str:
        """타임스탬프 없이 이어 붙인 전문. 끝에 개행은 붙이지 않는다."""
        return " ".join(s.text for s in self.segments)

    @property
    def covered(self) -> float:
        """마지막 세그먼트의 끝 시각(초). duration 과 크게 벌어지면 뒷부분이 누락된 것이다."""
        ends = [s.end for s in self.segments if s.end is not None]
        return max(ends) if ends else 0.0


@dataclass
class Progress:
    """진행 상황 한 틱. on_progress 콜백으로 전달된다."""

    done: int       # 지금까지 인코딩한 창 수
    total: int      # 전체 창 수(어림값)
    elapsed: float  # 전사 시작부터 지난 시간(초)
    eta: float      # 남은 시간 추정(초)

    @property
    def fraction(self) -> float:
        return self.done / self.total if self.total else 0.0


# ---------------------------------------------------------------- 유틸


def fmt_hms(seconds):
    """초를 hh:mm:ss 로 바꾼다. 타임스탬프가 없는 세그먼트를 위해 None 도 받는다."""
    if seconds is None:
        return "??:??:??"
    seconds = int(seconds)
    return f"{seconds // 3600:02d}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"


def pick_device():
    """CUDA 가 보이면 (cuda:0, fp16), 없으면 (cpu, fp32) 를 돌려준다.

    CPU 로도 돌아가지만 실사용이 어려울 만큼 느리다. 경고를 띄울지는 부르는 쪽이 정한다.
    """
    if torch.cuda.is_available():
        return "cuda:0", torch.float16
    return "cpu", torch.float32


def gpu_name():
    """인식된 GPU 이름. 없으면 None."""
    return torch.cuda.get_device_name(0) if torch.cuda.is_available() else None


def decode_audio(path):
    """ffmpeg 로 읽을 수 있는 파일을 16kHz 모노 float32 PCM 으로 디코딩한다.

    디코딩을 ffmpeg 에 맡기므로 파이썬 쪽에 형식별 코덱이 필요 없고,
    지원 형식은 컨테이너에 설치된 ffmpeg 가 읽을 수 있는 범위와 같아진다.
    결과는 임시 파일로 떨구지 않고 파이프(pipe:1)로 바로 받는다.

    입력은 경로로만 받는다. 업로드를 표준입력으로 흘리면 moov atom 이 파일 끝에 있는
    m4a/mp4 에서 ffmpeg 가 seek 하지 못해 실패한다. 서버는 업로드를 임시 파일로
    떨군 뒤 그 경로를 넘겨야 한다.
    """
    path = Path(path)
    cmd = [
        # -nostdin: 배치 루프 도중 ffmpeg 가 표준입력을 가로채지 않도록 막는다.
        # -threads 0: 디코딩 스레드 수를 ffmpeg 가 알아서 정하게 한다.
        "ffmpeg", "-nostdin", "-threads", "0", "-i", str(path),
        # f32le / ac 1 / ar 16000: Whisper 입력 규격(32비트 부동소수 리틀엔디언, 모노, 16kHz).
        "-f", "f32le", "-ac", "1", "-ar", str(SAMPLE_RATE), "-v", "error", "pipe:1",
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        # 부르는 쪽에서 잡아 이 파일만 건너뛰고 나머지를 계속 처리한다.
        raise RuntimeError(f"ffmpeg failed to decode {path.name}: {proc.stderr.decode(errors='replace')}")
    return np.frombuffer(proc.stdout, dtype=np.float32)


def segments_from_chunks(chunks):
    """transformers 파이프라인이 준 chunk 목록을 Segment 목록으로 옮긴다.

    공백뿐인 세그먼트는 버린다. 반복 판정과 출력 양쪽에서 잡음이 된다.
    """
    segments = []
    for chunk in chunks:
        # 타임스탬프를 못 붙인 세그먼트는 timestamp 가 None 이거나 끝값이 비어 있다.
        start, end = (chunk.get("timestamp") or (None, None))[:2]
        text = chunk["text"].strip()
        if text:
            segments.append(Segment(text=text, start=start, end=end))
    return segments


def collapse_repeats(segments, run_len):
    """같은 문장이 run_len 회 이상 연달아 나오면 하나로 접는다. (접힌 목록, 접힌 개수)를 돌려준다.

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
    for _, group in itertools.groupby(segments, key=lambda s: s.text):
        run = list(group)
        if len(run) >= run_len:
            # 몇 개를 접었는지 runs 에 남긴다. 나중에 환각 구간을 되짚을 때 쓸 수 있다.
            out.append(Segment(text=run[0].text, start=run[0].start, end=run[-1].end, runs=len(run)))
            collapsed += len(run) - 1
        else:
            out.extend(run)
    return out, collapsed


# ---------------------------------------------------------------- 출력 렌더러


def render_plain(result):
    """타임스탬프 없는 전문. 그대로 파일에 쓰면 되도록 끝에 개행을 붙인다."""
    return result.text + "\n"


def render_segments(result):
    """문장별 타임스탬프가 붙은 전문. 그대로 파일에 쓰면 된다.

    duration 헤더는 검증용이다. 마지막 타임스탬프가 이 값에 한참 못 미치면
    뒷부분이 통째로 누락된 것이다. README 의 "결과 검증" 참고.
    """
    lines = [f"# {result.source}", f"# duration: {fmt_hms(result.duration)}", ""]
    for seg in result.segments:
        lines.append(f"[{fmt_hms(seg.start)} -> {fmt_hms(seg.end)}] {seg.text}")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- 엔진


@contextmanager
def _progress_hook(model, total_windows, on_progress):
    """인코더 호출을 세어 진행 상황을 콜백으로 넘긴다. 빠져나갈 때 반드시 원복한다.

    두 모드 모두 30초 창 하나당 인코더가 정확히 한 번 돈다. 그래서 인코더 호출을
    세면 오디오를 미리 쪼개 보지 않고도 실제 진행률을 알 수 있다.

    주의: 모델 객체의 메서드를 갈아끼우는 방식이라, 같은 엔진으로 두 전사를 동시에
    돌릴 수 없다. 카운터가 서로 섞이고 원복 순서도 꼬인다. 서버에서 워커를 하나만
    두고 큐로 직렬화해야 하는 이유가 VRAM 만이 아니라 여기에도 있다.
    """
    if on_progress is None:
        yield
        return

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
        elapsed = time.time() - state["start"]
        frac = state["done"] / total_windows if total_windows else 0
        eta = elapsed / frac - elapsed if frac > 0 else 0
        on_progress(Progress(done=state["done"], total=total_windows, elapsed=elapsed, eta=eta))
        return out

    encoder.forward = counting_forward
    try:
        yield
    finally:
        # 전사가 실패해도 반드시 되돌린다. 안 그러면 같은 엔진을 재사용하는
        # 다음 전사가 이전 카운터를 물고 간다.
        encoder.forward = original


class SttEngine:
    """모델을 올려 두고 전사 요청을 받는다. 생성 비용이 크므로 만들어서 재사용한다.

    chunked 설정은 파이프라인 구성 자체를 바꾸므로 모드마다 파이프라인이 따로 필요하다.
    그래서 모델과 프로세서는 한 번만 올리고, 파이프라인만 모드별로 만들어 캐시한다.
    가중치는 파이프라인끼리 공유되므로 두 모드를 다 써도 VRAM 추가 비용이 거의 없다.
    덕분에 상주 서버가 요청마다 모드를 골라 받을 수 있다.
    """

    def __init__(self, model_id=DEFAULT_MODEL, batch_size=8, device=None, dtype=None):
        auto_device, auto_dtype = pick_device()
        self.model_id = model_id
        self.batch_size = batch_size
        self.device = device if device is not None else auto_device
        self.dtype = dtype if dtype is not None else auto_dtype

        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            # low_cpu_mem_usage: 가중치를 CPU 메모리에 통째로 펼치지 않고 올려 최초 로딩 부담을 줄인다.
            # sdpa: PyTorch 내장 스케일드 닷프로덕트 어텐션. 별도 패키지 설치 없이 쓸 수 있다.
            model_id, torch_dtype=self.dtype, low_cpu_mem_usage=True, attn_implementation="sdpa"
        ).to(self.device)
        self.processor = AutoProcessor.from_pretrained(model_id)
        self._pipelines = {}

    def pipeline_for(self, mode):
        """모드에 맞는 파이프라인을 돌려준다. 처음 쓰는 모드면 만들어 캐시한다."""
        if mode not in self._pipelines:
            kwargs = {}
            if mode == "chunked":
                # 빠른 쪽: 30초 창을 서로 독립적으로 병렬 디코딩하고 겹치는 부분으로 이어 붙인다.
                # 앞뒤 문맥을 보지 않으므로 타임스탬프가 거칠고 이음매에서 드물게 글자가 깨진다.
                kwargs = {"chunk_length_s": 30, "stride_length_s": 5, "batch_size": self.batch_size}
            self._pipelines[mode] = pipeline(
                "automatic-speech-recognition",
                model=self.model,
                tokenizer=self.processor.tokenizer,
                feature_extractor=self.processor.feature_extractor,
                torch_dtype=self.dtype,
                device=self.device,
                **kwargs,
            )
        return self._pipelines[mode]

    def _generate_kwargs(self, language, beams, no_speech, mode):
        """전사 품질을 좌우하는 설정. 이 저장소에서 가장 값비싼 부분이다."""
        kwargs = {"task": "transcribe", "num_beams": beams}
        # auto 면 language 를 아예 넘기지 않는다. 그래야 모델이 직접 언어를 감지한다.
        if language != "auto":
            kwargs["language"] = language
        if mode == "sequential":
            # Whisper 고유의 long-form 설정(모델 카드 참고). 디코드가 무너진 것으로 보이면
            # 온도를 올려 그 창만 다시 시도한다.
            kwargs.update(
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
                kwargs["no_speech_threshold"] = no_speech
        return kwargs

    def transcribe(self, path, language="ko", beams=1, mode="sequential", no_speech=None,
                   run_len=3, source=None, on_decoded=None, on_progress=None):
        """파일 하나를 전사해 TranscriptResult 를 돌려준다. 파일은 만들지 않는다.

        source 는 결과에 기록할 원본 이름이다. 기본은 path 의 파일명이지만, 서버처럼
        임시 파일로 받아 처리하는 쪽은 사용자가 올린 진짜 이름을 넘겨야 한다.
        이 값이 .segments.txt 헤더에 그대로 찍힌다.

        on_decoded(duration) 는 길이가 확정된 직후, 긴 전사가 시작되기 전에 불린다.
        CLI 는 길이를 미리 찍는 데 쓰고, 서버는 대기열 ETA 를 잡는 데 쓸 수 있다.
        on_progress(Progress) 는 인코더 창 하나를 지날 때마다 불린다.

        _progress_hook 이 모델 메서드를 갈아끼우므로 한 엔진으로 동시에 두 번
        부를 수 없다. 서버는 워커를 하나만 두고 큐로 직렬화해야 한다.
        """
        path = Path(path)
        asr = self.pipeline_for(mode)
        audio = decode_audio(path)
        # ffmpeg 가 16kHz 모노로 맞춰 줬으므로 샘플 수를 나누면 그대로 길이(초)가 된다.
        duration = len(audio) / SAMPLE_RATE
        if on_decoded is not None:
            on_decoded(duration)

        # 진행률 분모용 어림값이다. 창 하나가 실제로 몇 초씩 나아가는지는 모드와 stride
        # 설정에 따라 달라지므로 정확하지 않을 수 있다. 표시에만 쓰이고 전사 결과에는
        # 영향을 주지 않는다.
        window = 25 if mode == "chunked" else 30
        # -(-a // b) 는 올림 나눗셈. 끝에 남는 자투리 창도 한 개로 센다.
        total_windows = max(1, -(-int(duration) // window))

        generate_kwargs = self._generate_kwargs(language, beams, no_speech, mode)
        started = time.time()
        with _progress_hook(asr.model, total_windows, on_progress):
            raw_result = asr(audio, return_timestamps=True, generate_kwargs=generate_kwargs)
        elapsed = time.time() - started

        segments = segments_from_chunks(raw_result.get("chunks", []))
        collapsed = 0
        if run_len > 0:
            segments, collapsed = collapse_repeats(segments, run_len)

        return TranscriptResult(
            segments=segments,
            source=source if source is not None else path.name,
            duration=duration,
            elapsed=elapsed,
            collapsed=collapsed,
            language=language,
            model=self.model_id,
            mode=mode,
        )
