#!/usr/bin/env python3
# save as split_audio_streams.py

"""
Split all audio streams from a video into individual WAV files.

Requirements:
  - ffmpeg and ffprobe must be installed and on PATH.

Importable API:
  split_audio_streams_to_wav(input_path, output_dir, sample_rate=48000, bit_depth=16, overwrite=False) -> list[str]

CLI usage:
  python split_audio_streams.py /path/to/video.mp4 /path/to/outdir --sr 48000 --bit-depth 16 --overwrite
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import List, Optional


class FFmpegNotFoundError(RuntimeError):
    pass


def _check_binaries():
    """Ensure ffmpeg and ffprobe are available."""
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        raise FFmpegNotFoundError(
            "ffmpeg and/or ffprobe not found. Please install FFmpeg and make sure it's on your PATH."
        )


def _safe_slug(value: Optional[str]) -> str:
    """Make a filesystem-safe slug from tags like language/title."""
    if not value:
        return ""
    # collapse non-word characters to hyphens
    value = value.strip().lower()
    slug = re.sub(r"[^\w\-]+", "-", value)
    slug = re.sub(r"-{2,}", "-", slug).strip("-")
    return slug


def _probe_audio_streams(input_path: Path) -> list[dict]:
    """Return a list of audio stream dicts with index and tags via ffprobe."""
    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "a",
        "-show_entries", "stream=index,codec_name,channels,channel_layout:stream_tags=language,title",
        "-of", "json",
        str(input_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {result.stderr.strip()}")
    data = json.loads(result.stdout or "{}")
    return data.get("streams", [])


def _build_wav_name(base: str, stream_idx: int, lang: Optional[str], title: Optional[str]) -> str:
    parts = [f"{base}", f"a{stream_idx}"]
    if lang_s := _safe_slug(lang):
        parts.append(lang_s)
    if title_s := _safe_slug(title):
        parts.append(title_s)
    return "_".join(parts) + ".wav"


def split_audio_streams_to_wav(
    input_path: str | os.PathLike,
    output_dir: str | os.PathLike,
    sample_rate: int = 48000,
    bit_depth: int = 16,
    overwrite: bool = False,
) -> List[str]:
    """
    Extract each audio stream in `input_path` to a separate WAV in `output_dir`.

    Parameters
    ----------
    input_path : str | Path
        Video file path (any container; must be readable by ffmpeg).
    output_dir : str | Path
        Directory to write WAVs (created if missing).
    sample_rate : int
        Target WAV sample rate (e.g., 16000, 44100, 48000).
    bit_depth : int
        WAV PCM bit depth: 16 or 24 or 32. (Uses signed PCM LE.)
    overwrite : bool
        If True, overwrite existing files; otherwise fail if a target exists.

    Returns
    -------
    List[str]
        Paths to the created WAV files.

    Raises
    ------
    FFmpegNotFoundError
        If ffmpeg/ffprobe are not available on PATH.
    FileNotFoundError
        If input_path does not exist.
    ValueError
        If no audio streams are found.
    RuntimeError
        If ffmpeg/ffprobe invocations fail.
    """
    _check_binaries()

    in_path = Path(input_path)
    if not in_path.exists():
        raise FileNotFoundError(f"Input file not found: {in_path}")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    streams = _probe_audio_streams(in_path)
    if not streams:
        raise ValueError("No audio streams found in input.")

    # Choose PCM format based on bit depth
    pcm_fmt_map = {16: "pcm_s16le", 24: "pcm_s24le", 32: "pcm_s32le"}
    if bit_depth not in pcm_fmt_map:
        raise ValueError("bit_depth must be one of {16, 24, 32}.")
    pcm_codec = pcm_fmt_map[bit_depth]

    created_files: List[str] = []
    base = in_path.stem

    for s in streams:
        idx = s.get("index")
        tags = s.get("tags", {}) or {}
        lang = tags.get("language")
        title = tags.get("title")

        out_name = _build_wav_name(base, idx, lang, title)
        out_path = out_dir / out_name

        ffmpeg_cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel", "error",
            "-y" if overwrite else "-n",
            "-i", str(in_path),
            "-map", f"0:a:{streams.index(s)}",  # map the Nth audio stream
            "-acodec", pcm_codec,
            "-ar", str(sample_rate),
            str(out_path),
        ]

        result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            # If not overwriting and file exists, ffmpeg returns non-zero; provide clearer message.
            if not overwrite and out_path.exists():
                raise FileExistsError(f"Target exists (use overwrite=True): {out_path}")
            raise RuntimeError(f"ffmpeg failed for stream {idx}: {result.stderr.strip()}")

        created_files.append(str(out_path))

    return created_files


# --- Optional CLI ---
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Split all audio streams from a video to individual WAV files.")
    parser.add_argument("input", help="Path to input video file")
    parser.add_argument("output_dir", help="Directory to write WAV files")
    parser.add_argument("--sr", type=int, default=48000, help="Output sample rate (default: 48000)")
    parser.add_argument("--bit-depth", type=int, default=16, choices=[16, 24, 32], help="PCM bit depth (default: 16)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files")
    args = parser.parse_args()

    paths = split_audio_streams_to_wav(
        args.input,
        args.output_dir,
        sample_rate=args.sr,
        bit_depth=args.bit_depth,
        overwrite=args.overwrite,
    )
    print("\n".join(paths))