"""
Core transcription engine using faster-whisper (CTranslate2 backend).

Handles model loading, audio extraction from video, and transcription
with timestamp segments. CPU-only with int8 quantization.

Extracted from qdayanon-content-engine's audio_transcription.py.
"""

import logging
import os
import subprocess
import tempfile

from faster_whisper import WhisperModel

from cantonese import apply_cantonese_corrections, get_domain_prompt

logger = logging.getLogger(__name__)

# Model cache to avoid reloading
_model_cache: dict[str, WhisperModel] = {}


def load_model(
    model_size: str,
    compute_type: str | None = None,
    device: str = "cpu",
) -> WhisperModel:
    """
    Load a faster-whisper model.

    Auto-selects compute type based on device:
      - cuda: float16 (fastest, most accurate on GPU)
      - cpu:  int8 if AVX2 available, float32 otherwise

    Models are cached globally so subsequent calls with the same
    (model, device, compute_type) triple are instant.

    Args:
        model_size: A standard size ('tiny'/'base'/'small'/'medium'/'large-v3'),
                    an HF repo id (e.g. 'alvanlii/whisper-small-cantonese'),
                    or a local directory path containing CT2 model files.
        compute_type: Override quantization type. None = auto-detect by device.
        device: 'cpu' or 'cuda'.

    Returns:
        WhisperModel instance.
    """
    if not model_size or not isinstance(model_size, str) or not model_size.strip():
        logger.warning(f"Invalid model_size '{model_size}', defaulting to 'base'")
        model_size = "base"

    if compute_type is None:
        try:
            import ctranslate2
            supported = ctranslate2.get_supported_compute_types(device)
            if device == "cuda":
                compute_type = "float16" if "float16" in supported else "float32"
            else:
                compute_type = "int8" if "int8" in supported else "float32"
        except Exception:
            compute_type = "float32"

    cache_key = f"{model_size}_{device}_{compute_type}"
    if cache_key not in _model_cache:
        logger.info(f"Loading Whisper model: {model_size} ({device}, {compute_type})")
        print(f"Loading model: {model_size} ({device}, {compute_type} quantization)...")
        _model_cache[cache_key] = WhisperModel(model_size, device=device, compute_type=compute_type)
        print("Model loaded.")

    return _model_cache[cache_key]


def check_ffmpeg() -> bool:
    """Check if ffmpeg is available on the system."""
    try:
        subprocess.run(
            ["ffmpeg", "-version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def video_has_audio_stream(video_path: str) -> bool:
    """Check if a video file contains an audio stream using ffprobe."""
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-select_streams", "a",
                "-show_entries", "stream=codec_type",
                "-of", "csv=p=0",
                video_path,
            ],
            capture_output=True,
            text=True,
        )
        return "audio" in result.stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
        # If ffprobe is missing, assume audio exists and let ffmpeg handle it
        return True


def extract_audio_from_video(video_path: str) -> str:
    """
    Extract audio from a video file to a temporary WAV file using ffmpeg.

    Args:
        video_path: Path to the video file.

    Returns:
        Path to the extracted .wav audio file (caller should clean up).

    Raises:
        RuntimeError: If extraction fails or video has no audio.
    """
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    if not video_has_audio_stream(video_path):
        raise RuntimeError(f"Video file has no audio track: {video_path}")

    # Create temp file for extracted audio
    temp_fd, temp_path = tempfile.mkstemp(suffix=".wav")
    os.close(temp_fd)

    try:
        print("Extracting audio from video...")
        result = subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", video_path,
                "-vn",                    # No video
                "-acodec", "pcm_s16le",   # 16-bit PCM WAV
                "-ar", "16000",           # 16kHz (Whisper's native rate)
                "-ac", "1",               # Mono
                temp_path,
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg audio extraction failed: {result.stderr[-500:]}")

        if not os.path.exists(temp_path) or os.path.getsize(temp_path) == 0:
            raise RuntimeError("ffmpeg produced empty audio file")

        return temp_path

    except Exception:
        # Clean up on failure
        if os.path.exists(temp_path):
            os.unlink(temp_path)
        raise


def _segment_to_dict(seg) -> dict:
    """Convert a faster-whisper Segment (possibly with words) to a plain dict."""
    entry = {
        "start": seg.start,
        "end": seg.end,
        "text": seg.text.strip() if seg.text else "",
    }
    # Preserve word-level timestamps when available
    words = getattr(seg, "words", None)
    if words:
        entry["words"] = [
            {
                "start": w.start,
                "end": w.end,
                "word": w.word,
                "probability": getattr(w, "probability", None),
            }
            for w in words
        ]
    return entry


def transcribe_with_timestamps(
    audio_path: str,
    model_size: str = "base",
    language: str | None = None,
    domain_category: str | None = None,
    initial_prompt: str | None = None,
    max_duration: int | None = None,
    device: str = "cpu",
    compute_type: str | None = None,
    vad_filter: bool = False,
    word_timestamps: bool = False,
    condition_on_previous_text: bool = True,
) -> dict:
    """
    Transcribe an audio file and return segments with timestamps.

    Args:
        audio_path: Path to audio file (.wav, .mp3, etc.).
        model_size: Whisper model size, HF repo id, or local CT2 dir.
        language: Force language code (e.g., 'en', 'zh'). None = auto-detect.
        domain_category: Domain for Cantonese vocabulary hints (e.g., 'iching').
        initial_prompt: Free-form text passed to Whisper as initial_prompt to bias
            vocabulary recognition. Works in any language. If both this and
            domain_category are provided, this explicit prompt wins.
        max_duration: Only transcribe first N seconds (for testing).
        device: 'cpu' or 'cuda'.
        compute_type: CTranslate2 compute type override. None = auto by device.
        vad_filter: Enable Silero VAD silence pre-filtering.
        word_timestamps: Include per-word timestamps inside each segment.
        condition_on_previous_text: When False, reduces hallucination loops at the
            cost of slightly less-consistent phrasing across segments. Default True
            matches Whisper's reference behavior.

    Returns:
        Dict with 'text' (str), 'segments' (list of dicts), 'language' (str),
        'duration' (float).
    """
    model = load_model(model_size, compute_type=compute_type, device=device)

    # Resolve initial prompt: explicit caller-supplied value wins; otherwise
    # fall back to the Cantonese domain dictionary.
    if initial_prompt is None and domain_category:
        initial_prompt = get_domain_prompt(domain_category)
        if initial_prompt:
            logger.info(f"Using domain prompt for '{domain_category}'")
    elif initial_prompt:
        logger.info("Using caller-supplied initial_prompt (overrides --domain).")

    print(f"Transcribing (language: {language or 'auto-detect'})...")

    transcribe_kwargs = {
        "audio": audio_path,
        "language": language,
        "initial_prompt": initial_prompt,
        "log_progress": True,
        "vad_filter": vad_filter,
        "word_timestamps": word_timestamps,
        "condition_on_previous_text": condition_on_previous_text,
    }
    if max_duration:
        transcribe_kwargs["clip_timestamps"] = f"0,{max_duration}"

    segments_generator, info = model.transcribe(**transcribe_kwargs)

    if info is None:
        raise RuntimeError("Transcription returned no info — model may have failed to process audio")

    # Collect segments
    segment_list = []
    for seg in segments_generator:
        if seg is None:
            continue
        segment_list.append(_segment_to_dict(seg))

    transcript = " ".join(seg["text"] for seg in segment_list)
    detected_language = info.language

    logger.info(f"Detected language: {detected_language} (probability: {info.language_probability:.3f})")

    # Garbled transcription detection — retry with English
    if not language and detected_language != "en":
        garbled_indicators = [
            "Fuck replacement",
            "Mae aa",
            "yn cfangygyng",
            "fel ddyn nad",
        ]
        if any(indicator in transcript for indicator in garbled_indicators):
            print("Transcription looks garbled, retrying with English...")
            retry_kwargs = {
                "audio": audio_path,
                "language": "en",
                "initial_prompt": initial_prompt,
                "vad_filter": vad_filter,
                "word_timestamps": word_timestamps,
                "condition_on_previous_text": condition_on_previous_text,
            }
            # Honour --max-duration on retry too (mirror the original call).
            if max_duration:
                retry_kwargs["clip_timestamps"] = f"0,{max_duration}"
            retry_segments, retry_info = model.transcribe(**retry_kwargs)
            segment_list = []
            for seg in retry_segments:
                if seg is None:
                    continue
                segment_list.append(_segment_to_dict(seg))
            transcript = " ".join(seg["text"] for seg in segment_list)
            detected_language = "en"

    # Apply Cantonese corrections when Chinese-family language detected
    if detected_language in ("yue", "zh", "zh-TW", "zh-HK", "cmn"):
        original = transcript
        transcript = apply_cantonese_corrections(transcript, domain=domain_category)
        for seg in segment_list:
            seg["text"] = apply_cantonese_corrections(seg["text"], domain=domain_category)
        if transcript != original:
            print(f"Applied Cantonese corrections (domain: {domain_category or 'general'})")

    # Compute duration from last segment
    duration = segment_list[-1]["end"] if segment_list else 0

    print(f"Transcription complete: {len(segment_list)} segments, {duration:.1f}s, language={detected_language}")

    return {
        "text": transcript,
        "segments": segment_list,
        "language": detected_language,
        "duration": duration,
    }
