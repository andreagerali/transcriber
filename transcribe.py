#!/usr/bin/env python3
"""
Video/audio transcription tool — generates .srt and .txt subtitle files.

Supports local MP4 files, YouTube URLs, and generic video/audio URLs.
Optimized for Cantonese + English transcription on CPU.

Usage:
    python transcribe.py video.mp4
    python transcribe.py video.mp4 -l zh --domain iching
    python transcribe.py "https://youtube.com/watch?v=..." -o subtitles/
    python transcribe.py audio.mp3 --format srt
    python transcribe.py video.mp4 --compress
    python transcribe.py video.mp4 --compress --compress-quality balanced
"""

import argparse
import logging
import os
import sys
import time

from cantonese import get_available_domains


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Transcribe video/audio to .srt and .txt subtitle files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python transcribe.py video.mp4\n"
            "  python transcribe.py video.mp4 -l zh --domain iching\n"
            '  python transcribe.py "https://youtube.com/watch?v=..." -o subs/\n'
            "  python transcribe.py lecture.mp4 -m small --format srt\n"
            "  python transcribe.py video.mp4 --compress\n"
            "  python transcribe.py video.mp4 --compress --compress-quality balanced\n"
            '  python transcribe.py interview.m4a -m turbo --langs en,ja --audit-csv\n'
            '  python transcribe.py seoul.m4a -m turbo --langs en,ko --prompt en="..."\n'
            '  python transcribe.py beijing.m4a -m turbo --langs en,zh\n'
            "\n"
            "Multilingual mode (two declared languages):\n"
            "  --langs en,ja  (or --langs en,zh / en,vi / en,th / en,id ...)\n"
            "  --mode jp == --langs en,ja ; --mode kr == --langs en,ko (shortcuts)\n"
            "  Splits on VAD, detects language per chunk, transcribes each in its language.\n"
            "  Add --prompt CODE=TEXT per language, and --audit-csv to log decisions.\n"
            "  Near-identical pairs (id+ms, hi+ur) are refused unless --force-langs.\n"
            "\n"
            "Language tips:\n"
            "  For Cantonese content, use: -l zh  (NOT -l yue)\n"
            "  For English content, use: -l en\n"
            "  Omit -l to auto-detect language\n"
            "\n"
            f"Available domains for --domain: {', '.join(get_available_domains())}\n"
        ),
    )

    parser.add_argument(
        "input",
        help="MP4/audio file path, YouTube URL, or any video/audio URL",
    )
    parser.add_argument(
        "-o", "--output",
        metavar="DIR",
        help="Output directory (default: same dir as input file, or ./output for URLs)",
    )
    parser.add_argument(
        "-m", "--model",
        default="base",
        help=(
            "Whisper model to load. Accepts: (a) a standard size — "
            "tiny/base/small/medium/large-v3 (default: base); "
            "(b) an HF repo id like 'alvanlii/whisper-small-cantonese'; "
            "(c) a local directory containing CTranslate2 model files."
        ),
    )
    parser.add_argument(
        "-l", "--language",
        default=None,
        help="Force language: en, zh (for Cantonese), etc. Default: auto-detect.",
    )
    parser.add_argument(
        "--domain",
        default=None,
        help="Cantonese domain vocabulary hints (e.g., iching, geopolitics, finance).",
    )
    parser.add_argument(
        "--initial-prompt",
        default=None,
        metavar="TEXT",
        help=(
            "Free-form text passed to Whisper as initial_prompt to bias terminology "
            "recognition. Works in any language. Example for an English economics "
            "interview: --initial-prompt \"Discussion of US-China tariffs, "
            "transshipment, rules of origin, FDI screening, semiconductors.\" "
            "If both --initial-prompt and --domain are given, --initial-prompt wins. "
            "Ignored in --mode multilingual (use --initial-prompt-en/-ja/-ko)."
        ),
    )

    # --- Multilingual (chunk-based, two-language) options ---
    multi = parser.add_argument_group(
        "multilingual mode",
        "Transcribe recordings that alternate between two declared languages "
        "(e.g. an English interview with a Japanese expert answering in Japanese). "
        "Splits the audio on VAD boundaries, detects the language of each chunk, "
        "and transcribes each chunk in its own language.",
    )
    multi.add_argument(
        "--langs",
        default=None,
        metavar="CODES",
        help=(
            "Enable the multilingual chunk pipeline for the comma-separated "
            "language codes actually present in the recording, e.g. 'en,ja', "
            "'en,zh', 'en,vi'. Keep it to two (English + one). When set, "
            "-l/--language is ignored. Acoustically near-identical pairs (e.g. "
            "id+ms, hi+ur) are refused unless --force-langs."
        ),
    )
    multi.add_argument(
        "--mode",
        choices=["jp", "kr"],
        default=None,
        help="Shortcut presets: 'jp' = --langs en,ja ; 'kr' = --langs en,ko. Prefer --langs.",
    )
    multi.add_argument(
        "--force-langs",
        action="store_true",
        help="Override the refusal of an acoustically confusable language pair (e.g. id+ms).",
    )
    multi.add_argument(
        "--default-language",
        default=None,
        metavar="CODE",
        help=(
            "Fallback language for chunks with out-of-set or low-confidence "
            "detection (default: the first language in --langs)."
        ),
    )
    multi.add_argument(
        "--min-chunk-duration",
        type=float,
        default=10.0,
        metavar="SECS",
        help=(
            "Chunks shorter than this require high detection confidence (>0.80) to "
            "trust the detected language; otherwise they fall back to the default "
            "language (default: 10)."
        ),
    )
    multi.add_argument(
        "--lang-detect-threshold",
        type=float,
        default=0.45,
        metavar="P",
        help="Minimum detection probability to accept a detected language (default: 0.45).",
    )
    multi.add_argument(
        "--prompt",
        action="append",
        default=None,
        metavar="CODE=TEXT",
        help=(
            "Per-language initial_prompt, repeatable: "
            "--prompt en=\"...\" --prompt ja=\"...\". Biases terminology for "
            "chunks transcribed as CODE. Add one per active language."
        ),
    )
    # Back-compat aliases for --prompt (prefer --prompt CODE=TEXT).
    multi.add_argument("--initial-prompt-en", default=None, metavar="TEXT", help=argparse.SUPPRESS)
    multi.add_argument("--initial-prompt-ja", default=None, metavar="TEXT", help=argparse.SUPPRESS)
    multi.add_argument("--initial-prompt-ko", default=None, metavar="TEXT", help=argparse.SUPPRESS)
    multi.add_argument(
        "--vad-threshold",
        type=float,
        default=0.5,
        metavar="P",
        help="Silero VAD speech probability threshold (default: 0.5).",
    )
    multi.add_argument(
        "--vad-min-silence-ms",
        type=int,
        default=700,
        metavar="MS",
        help=(
            "Minimum silence (ms) that splits two speech chunks. Lower = finer "
            "turn boundaries (better language separation), higher = fewer, longer "
            "chunks (default: 700; faster-whisper's own default is 2000)."
        ),
    )
    multi.add_argument(
        "--vad-speech-pad-ms",
        type=int,
        default=200,
        metavar="MS",
        help="Padding (ms) added around each VAD speech chunk (default: 200).",
    )
    multi.add_argument(
        "--audit-csv",
        action="store_true",
        help="Also write a <name>.lang.csv logging the language decision for every chunk.",
    )
    parser.add_argument(
        "--format",
        default="both",
        choices=["srt", "txt", "both"],
        help="Output format (default: both).",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["cpu", "cuda", "auto"],
        help="Inference device (default: auto — use cuda if available, else cpu).",
    )
    parser.add_argument(
        "--compute-type",
        default=None,
        help=(
            "CTranslate2 compute type. Default: float16 on cuda, int8 on cpu. "
            "Valid: float32, float16, int8, int8_float16, int8_float32, int16."
        ),
    )
    parser.add_argument(
        "--vad-filter",
        action="store_true",
        help=(
            "Enable Silero VAD pre-filter — skips silence regions, producing "
            "longer coherent speech segments. Recommended for podcasts/meetings."
        ),
    )
    parser.add_argument(
        "--word-timestamps",
        action="store_true",
        help="Emit word-level timestamps inside each segment (useful for downstream batching).",
    )
    parser.add_argument(
        "--no-condition-prev",
        action="store_true",
        help=(
            "Disable condition_on_previous_text. Reduces hallucination/repetition loops "
            "at the cost of slightly less-consistent phrasing across segments."
        ),
    )
    parser.add_argument(
        "--max-duration",
        type=int,
        default=None,
        metavar="SECS",
        help="Only transcribe first N seconds (for testing).",
    )
    parser.add_argument(
        "--keep-audio",
        action="store_true",
        help="Keep downloaded audio file (default: clean up).",
    )
    parser.add_argument(
        "--compress",
        action="store_true",
        help="Compress video for YouTube upload (H.264, no transcription).",
    )
    parser.add_argument(
        "--compress-quality",
        default="max_compression",
        choices=["high_quality", "balanced", "max_compression"],
        help="Compression quality preset (default: max_compression = CRF 28).",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )

    return parser.parse_args()


def main() -> int:
    # Make console output UTF-8 safe so non-ASCII paths/filenames (e.g. a
    # Japanese-named recording) never crash the run on a cp1252 Windows console.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass

    args = parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.WARNING
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Import here so --help is fast (no model loading)
    from transcription import check_ffmpeg, extract_audio_from_video, transcribe_with_timestamps
    from downloader import is_url, download_url
    from srt_formatter import write_srt, write_txt, generate_output_path

    # Pre-flight checks
    if not check_ffmpeg():
        print("ERROR: ffmpeg not found on your system.", file=sys.stderr)
        print("Install it:", file=sys.stderr)
        print("  Linux:   sudo apt install ffmpeg", file=sys.stderr)
        print("  macOS:   brew install ffmpeg", file=sys.stderr)
        print("  Windows: winget install ffmpeg", file=sys.stderr)
        return 1

    # --- Compress-only mode ---
    if args.compress:
        from compress import compress_for_youtube

        if not os.path.isfile(args.input):
            print(f"ERROR: File not found: {args.input}", file=sys.stderr)
            return 1

        try:
            start_time = time.time()
            output_path = None
            if args.output:
                # Use output dir with original filename + _compressed
                base = os.path.splitext(os.path.basename(args.input))[0]
                output_path = os.path.join(args.output, f"{base}_compressed.mp4")

            result = compress_for_youtube(
                args.input,
                output_path=output_path,
                quality=args.compress_quality,
            )
            elapsed = time.time() - start_time

            print()
            print("=" * 60)
            print(f"  Compression: {result['ratio']:.1f}x smaller")
            print(f"  Time taken:  {elapsed:.1f}s")
            print(f"  Output:      {result['output_path']}")
            print("=" * 60)
            return 0

        except Exception as e:
            print(f"ERROR: {e}", file=sys.stderr)
            if args.verbose:
                import traceback
                traceback.print_exc()
            return 1

    # Speed warnings for large models on CPU (only when explicitly running on CPU)
    effective_device = args.device
    if effective_device == "auto":
        try:
            import ctranslate2
            effective_device = "cuda" if "float16" in ctranslate2.get_supported_compute_types("cuda") else "cpu"
        except Exception:
            effective_device = "cpu"

    if effective_device == "cpu":
        if args.model == "large-v3":
            print("WARNING: large-v3 on CPU is very slow (expect 10-20x real-time).")
            print("         Consider 'base' or 'small' for faster results.")
            print()
        elif args.model == "medium":
            print("Note: 'medium' model on CPU is slow (expect 5-10x real-time).")
            print("      Use 'base' for faster results with good accuracy.")
            print()

    # Track temp files for cleanup
    temp_files: list[str] = []

    try:
        # --- Step 1: Get audio file ---
        if is_url(args.input):
            audio_path, title = download_url(args.input)
            if not args.keep_audio:
                temp_files.append(audio_path)
            input_name = title
        else:
            # Local file
            if not os.path.isfile(args.input):
                print(f"ERROR: File not found: {args.input}", file=sys.stderr)
                return 1

            input_name = args.input

            # Check if it's a video file that needs audio extraction
            ext = os.path.splitext(args.input)[1].lower()
            video_extensions = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".m4v", ".flv", ".3gp"}
            audio_extensions = {".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg", ".opus", ".wma"}

            if ext in video_extensions:
                audio_path = extract_audio_from_video(args.input)
                temp_files.append(audio_path)
            elif ext in audio_extensions:
                audio_path = args.input
            else:
                # Try treating it as video (let ffmpeg figure it out)
                print(f"Unknown extension '{ext}', attempting audio extraction...")
                try:
                    audio_path = extract_audio_from_video(args.input)
                    temp_files.append(audio_path)
                except RuntimeError:
                    # Maybe it's already audio — try directly
                    audio_path = args.input

        # --- Multilingual chunk-based path (two declared languages) ---
        if args.langs or args.mode:
            from multilingual import run_multilingual
            try:
                return run_multilingual(audio_path, input_name, args, effective_device)
            except ValueError as exc:
                print(f"ERROR: {exc}", file=sys.stderr)
                return 2

        # --- Step 2: Transcribe ---
        start_time = time.time()

        result = transcribe_with_timestamps(
            audio_path=audio_path,
            model_size=args.model,
            language=args.language,
            domain_category=args.domain,
            initial_prompt=args.initial_prompt,
            max_duration=args.max_duration,
            device=effective_device,
            compute_type=args.compute_type,
            vad_filter=args.vad_filter,
            word_timestamps=args.word_timestamps,
            condition_on_previous_text=not args.no_condition_prev,
        )

        elapsed = time.time() - start_time
        segments = result["segments"]
        detected_lang = result["language"]
        duration = result["duration"]

        if not segments:
            print("WARNING: No speech detected in the audio.", file=sys.stderr)
            return 1

        # --- Step 3: Write output files ---
        output_files = []

        if args.format in ("srt", "both"):
            srt_path = generate_output_path(input_name, args.output, ".srt")
            write_srt(segments, srt_path)
            output_files.append(srt_path)

        if args.format in ("txt", "both"):
            txt_path = generate_output_path(input_name, args.output, ".txt")
            write_txt(segments, txt_path)
            output_files.append(txt_path)

        # --- Step 4: Summary ---
        print()
        print("=" * 60)
        print(f"  Language:   {detected_lang}")
        print(f"  Duration:   {duration:.1f}s ({duration/60:.1f} min)")
        print(f"  Segments:   {len(segments)}")
        print(f"  Time taken: {elapsed:.1f}s ({elapsed/duration:.1f}x real-time)" if duration > 0 else f"  Time taken: {elapsed:.1f}s")
        print(f"  Model:      {args.model}")
        for f in output_files:
            print(f"  Output:     {f}")
        print("=" * 60)

        return 0

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        return 130

    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1

    finally:
        # Clean up temp files
        for f in temp_files:
            try:
                if os.path.exists(f):
                    os.unlink(f)
            except OSError:
                pass


if __name__ == "__main__":
    sys.exit(main())
