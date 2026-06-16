"""
Chunk-based multilingual transcription.

Handles recordings that alternate between two declared languages — e.g. an
English interview in which a Japanese expert answers in Japanese (EN+JA), or
the Korean equivalent for a Seoul mission (EN+KO). Whisper picks a single
language per `transcribe()` call and cannot switch mid-stream, so this module:

  1. extracts the speech timeline with Silero VAD,
  2. keeps the VAD segments as chunks (no coarse 60s grouping — short turns
     such as a ~10s Japanese answer are preserved as their own chunk),
  3. detects the language of each chunk independently (encoder-only pass),
  4. transcribes each chunk forcing the detected language, with a per-language
     initial_prompt, falling back to a default language when detection is
     out-of-set or low-confidence,
  5. concatenates the segments on the original timeline and writes output with
     a per-language marker, plus an optional audit CSV of every decision.

Language pairs are mutually exclusive presets (`--mode jp` => {en, ja};
`--mode kr` => {en, ko}) so the EN-vs-X decision stays acoustically easy and
JA/KO are never pitted against each other (they are the closest pair).
"""

from __future__ import annotations

import csv
import logging
import sys

from faster_whisper import decode_audio
from faster_whisper.vad import VadOptions, get_speech_timestamps

from srt_formatter import format_timestamp, generate_output_path
from transcription import _segment_to_dict, load_model

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000

# Mutually-exclusive language presets. The first entry is the conventional
# default/fallback language for the pair.
MODE_LANGUAGE_SETS: dict[str, tuple[str, ...]] = {
    "jp": ("en", "ja"),
    "kr": ("en", "ko"),
}

# Above this confidence, trust the detected language even on a chunk shorter
# than --min-chunk-duration. EN vs JA/KO is acoustically very distinct, so a
# high-probability detection on a short clip is still reliable.
_HIGH_CONFIDENCE = 0.80


def _marker_time(seconds: float) -> str:
    """Wall-clock marker HH:MM:SS (no milliseconds) for transcript headers."""
    seconds = max(0.0, seconds)
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _vad_segments(
    audio,
    threshold: float,
    min_silence_ms: int,
    speech_pad_ms: int,
) -> list[tuple[float, float]]:
    """Run Silero VAD and return speech spans as (start_s, end_s) tuples."""
    options = VadOptions(
        threshold=threshold,
        min_silence_duration_ms=min_silence_ms,
        speech_pad_ms=speech_pad_ms,
    )
    raw = get_speech_timestamps(audio, options, sampling_rate=SAMPLE_RATE)
    # get_speech_timestamps returns sample indices, not seconds.
    return [(t["start"] / SAMPLE_RATE, t["end"] / SAMPLE_RATE) for t in raw]


def _decide_language(
    detected: str,
    prob: float,
    duration: float,
    active_set: tuple[str, ...],
    default_lang: str,
    threshold: float,
    min_chunk_duration: float,
) -> tuple[str, str]:
    """
    Resolve which language to force for a chunk.

    Returns (used_language, fallback_reason). fallback_reason is "" when the
    detected language was accepted.
    """
    if detected not in active_set:
        return default_lang, "out_of_set"
    if prob < threshold:
        return default_lang, "low_confidence"
    if duration < min_chunk_duration and prob <= _HIGH_CONFIDENCE:
        return default_lang, "short_low_conf"
    return detected, ""


def run_multilingual(audio_path: str, input_name: str, args, device: str) -> int:
    """
    Execute the chunk-based multilingual pipeline and write output files.

    Args:
        audio_path: Path to a decodable audio file (wav extracted from video,
            or the original audio file). decode_audio handles m4a/mp3/wav/etc.
        input_name: Original input name used to derive output file paths.
        args: Parsed argparse namespace from transcribe.py.
        device: Resolved inference device ('cpu' or 'cuda').

    Returns:
        Process exit code (0 on success).
    """
    active_set = MODE_LANGUAGE_SETS[args.mode]
    default_lang = (args.default_language or "en").lower()
    if default_lang not in active_set:
        fallback = "en" if "en" in active_set else active_set[0]
        print(
            f"WARNING: --default-language '{default_lang}' is not in the "
            f"--mode {args.mode} set {active_set}; using '{fallback}'.",
            file=sys.stderr,
        )
        default_lang = fallback

    # Per-language initial prompts. Only languages in the active set are used;
    # warn on prompts supplied for the inactive language.
    prompts = {
        "en": getattr(args, "initial_prompt_en", None),
        "ja": getattr(args, "initial_prompt_ja", None),
        "ko": getattr(args, "initial_prompt_ko", None),
    }
    for lang, prompt in prompts.items():
        if prompt and lang not in active_set:
            print(
                f"WARNING: --initial-prompt-{lang} supplied but '{lang}' is not "
                f"active in --mode {args.mode}; it will be ignored.",
                file=sys.stderr,
            )
    if getattr(args, "initial_prompt", None):
        print(
            "WARNING: --initial-prompt is ignored in multilingual mode; use "
            "--initial-prompt-en / --initial-prompt-ja / --initial-prompt-ko.",
            file=sys.stderr,
        )
    if getattr(args, "language", None):
        print(
            "WARNING: -l/--language is ignored in multilingual mode; languages "
            f"are fixed by --mode {args.mode} {active_set}.",
            file=sys.stderr,
        )

    model = load_model(args.model, compute_type=args.compute_type, device=device)

    print("Decoding audio...")
    audio = decode_audio(audio_path, sampling_rate=SAMPLE_RATE)

    if args.max_duration:
        limit = int(args.max_duration * SAMPLE_RATE)
        audio = audio[:limit]
    total_dur = len(audio) / SAMPLE_RATE

    print("Running VAD to find speech segments...")
    chunks = _vad_segments(
        audio,
        threshold=args.vad_threshold,
        min_silence_ms=args.vad_min_silence_ms,
        speech_pad_ms=args.vad_speech_pad_ms,
    )
    if not chunks:
        print(
            "WARNING: No speech detected. Try lowering --vad-threshold or "
            "raising --vad-min-silence-ms.",
            file=sys.stderr,
        )
        return 1

    print(
        f"Found {len(chunks)} speech chunks across {total_dur/60:.1f} min. "
        f"Detecting language and transcribing per chunk "
        f"(mode={args.mode} {active_set}, default={default_lang})..."
    )

    condition_prev = not getattr(args, "no_condition_prev", False)
    all_segments: list[dict] = []
    audit: list[dict] = []
    lang_counts: dict[str, int] = {}

    for idx, (start_s, end_s) in enumerate(chunks):
        start_i = int(start_s * SAMPLE_RATE)
        end_i = int(end_s * SAMPLE_RATE)
        clip = audio[start_i:end_i]
        duration = end_s - start_s
        if clip.shape[0] == 0:
            # Sub-sample chunk (unreachable with default VAD padding, but keep
            # the audit complete: one row per VAD chunk).
            audit.append(
                {
                    "chunk_idx": idx,
                    "start_s": round(start_s, 3),
                    "end_s": round(end_s, 3),
                    "duration_s": round(duration, 3),
                    "detected_lang": "",
                    "detected_prob": 0.0,
                    "used_lang": "",
                    "fallback_reason": "zero_length",
                }
            )
            continue

        # --- Language detection (encoder-only) ---
        try:
            detected, prob, _ = model.detect_language(audio=clip)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("detect_language failed on chunk %d: %s", idx, exc)
            detected, prob = default_lang, 0.0

        used, reason = _decide_language(
            detected,
            prob,
            duration,
            active_set,
            default_lang,
            args.lang_detect_threshold,
            args.min_chunk_duration,
        )

        audit.append(
            {
                "chunk_idx": idx,
                "start_s": round(start_s, 3),
                "end_s": round(end_s, 3),
                "duration_s": round(duration, 3),
                "detected_lang": detected,
                "detected_prob": round(float(prob), 4),
                "used_lang": used,
                "fallback_reason": reason,
            }
        )

        # --- Transcribe the chunk forcing the resolved language ---
        seg_gen, info = model.transcribe(
            audio=clip,
            language=used,
            initial_prompt=prompts.get(used),
            vad_filter=False,  # already VAD-segmented upstream
            word_timestamps=args.word_timestamps,
            condition_on_previous_text=condition_prev,
        )

        n_seg = 0
        for seg in seg_gen:
            if seg is None:
                continue
            entry = _segment_to_dict(seg)
            if not entry["text"]:
                continue
            # Rebase chunk-local timestamps onto the global timeline.
            entry["start"] += start_s
            entry["end"] += start_s
            if "words" in entry:
                for w in entry["words"]:
                    if w.get("start") is not None:
                        w["start"] += start_s
                    if w.get("end") is not None:
                        w["end"] += start_s
            entry["lang"] = used.upper()
            all_segments.append(entry)
            n_seg += 1

        lang_counts[used.upper()] = lang_counts.get(used.upper(), 0) + n_seg
        flag = f" -> {reason}" if reason else ""
        print(
            f"  [{idx + 1}/{len(chunks)}] {_marker_time(start_s)} "
            f"{used.upper()}  (det={detected}:{prob:.2f}{flag}, {duration:.1f}s)"
        )

    if not all_segments:
        print(
            "WARNING: speech chunks were found but none produced any text. "
            "No output written. Try a larger model or check the audio.",
            file=sys.stderr,
        )
        return 1

    # --- Write outputs ---
    output_files: list[str] = []
    if args.format in ("srt", "both"):
        srt_path = generate_output_path(input_name, args.output, ".srt")
        write_marked_srt(all_segments, srt_path)
        output_files.append(srt_path)
    if args.format in ("txt", "both"):
        txt_path = generate_output_path(input_name, args.output, ".txt")
        write_marked_txt(all_segments, txt_path)
        output_files.append(txt_path)
    if getattr(args, "audit_csv", False):
        csv_path = generate_output_path(input_name, args.output, ".lang.csv")
        write_audit_csv(audit, csv_path)
        output_files.append(csv_path)

    # --- Summary ---
    print()
    print("=" * 60)
    print(f"  Mode:       {args.mode} {active_set}")
    print(f"  Duration:   {total_dur:.1f}s ({total_dur/60:.1f} min)")
    print(f"  Chunks:     {len(chunks)}")
    print(f"  Segments:   {len(all_segments)}")
    by_lang = ", ".join(f"{k}:{v}" for k, v in sorted(lang_counts.items()))
    print(f"  By language: {by_lang or '(none)'}")
    fallbacks = sum(1 for a in audit if a["fallback_reason"])
    print(f"  Fallbacks:  {fallbacks}/{len(audit)} chunks -> {default_lang}")
    print(f"  Model:      {args.model}")
    for path in output_files:
        print(f"  Output:     {path}")
    print("=" * 60)

    return 0


def write_marked_txt(segments: list[dict], output_path: str) -> str:
    """
    Plain-text transcript with one paragraph per language run.

    Consecutive same-language segments are merged into a single paragraph
    headed by a [HH:MM:SS LANG] marker, so language switches are obvious:

        [00:00:00 EN] Good morning, thanks for joining us...
        [00:00:23 JA] 中国の輸出戦略については...
        [00:00:41 EN] He's saying that China's export strategy...
    """
    import os

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    blocks: list[str] = []
    cur_lang: str | None = None
    cur_start = 0.0
    cur_text: list[str] = []

    def flush() -> None:
        if cur_text:
            blocks.append(
                f"[{_marker_time(cur_start)} {cur_lang}] " + " ".join(cur_text)
            )

    for seg in segments:
        text = seg.get("text", "").strip()
        if not text:
            continue
        lang = seg.get("lang", "")
        if lang != cur_lang:
            flush()
            cur_lang = lang
            cur_start = seg["start"]
            cur_text = [text]
        else:
            cur_text.append(text)
    flush()

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(blocks) + ("\n" if blocks else ""))

    return output_path


def write_marked_srt(segments: list[dict], output_path: str) -> str:
    """SRT with a [LANG] tag prefixed to each subtitle's text."""
    import os

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    with open(output_path, "w", encoding="utf-8-sig") as f:
        index = 1
        for seg in segments:
            text = seg.get("text", "").strip()
            if not text:
                continue
            lang = seg.get("lang", "")
            start = format_timestamp(seg["start"])
            end = format_timestamp(seg["end"])
            tag = f"[{lang}] " if lang else ""
            f.write(f"{index}\n")
            f.write(f"{start} --> {end}\n")
            f.write(f"{tag}{text}\n\n")
            index += 1

    return output_path


def write_audit_csv(audit: list[dict], output_path: str) -> str:
    """Per-chunk language-decision log for post-hoc inspection and tuning."""
    import os

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    fields = [
        "chunk_idx",
        "start_s",
        "end_s",
        "duration_s",
        "detected_lang",
        "detected_prob",
        "used_lang",
        "fallback_reason",
    ]
    # utf-8-sig so Excel opens it cleanly on Windows.
    with open(output_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in audit:
            writer.writerow(row)

    return output_path
