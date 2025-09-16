from pathlib import Path
from typing import Optional, Dict, Literal

from ..audio.split_wav_by_speaker import make_speaker_wavs_from_csv

def split_wav_by_speaker(
    self,
    source_wav: str | Path,
    transcript_csv: str | Path,
    out_dir: str | Path,
    *,
    time_unit: Literal["ms", "s"] = "ms",
    silence_ms: int = 1000,
    pre_silence_ms: Optional[int] = None,
    post_silence_ms: Optional[int] = None,
    sr: Optional[int] = 16000,
    mono: bool = True,
    min_dur_ms: int = 50,
    start_col: str = "start_time",
    end_col: str = "end_time",
    speaker_col: str = "speaker",
) -> Dict[str, Path]:
    """
    Create one WAV per speaker using a timestamped transcript CSV.

    - Inserts silence before and after each segment for that speaker.
    - Defaults: timestamps in ms; 1s (1000 ms) of silence before/after each clip.

    Returns:
        Dict[str, Path]: {speaker_name -> path_to_wav}
    """
    # Make output dir if it doesn't already exist
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    return make_speaker_wavs_from_csv(
        source_wav=Path(source_wav),
        csv_path=Path(transcript_csv),
        out_dir=Path(out_dir),
        start_col=start_col,
        end_col=end_col,
        speaker_col=speaker_col,
        time_unit=time_unit,
        silence_ms=silence_ms,
        pre_silence_ms=pre_silence_ms,
        post_silence_ms=post_silence_ms,
        sr=sr,
        mono=mono,
        min_dur_ms=min_dur_ms,
    )