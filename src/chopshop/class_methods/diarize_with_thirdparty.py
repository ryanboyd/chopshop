from pathlib import Path
from typing import Optional

from ..audio.diarizer.whisper_diar_wrapper import DiarizationOutputFiles, run_whisper_diarization_repo

def diarize_with_thirdparty(
    self,
    input_audio: str | Path,
    out_dir: str | Path,
    *,
    repo_dir: str | Path | None = None,
    whisper_model: str = "medium.en",
    language: Optional[str] = None,
    device: Optional[str] = None,     # "cuda" or "cpu"; None = leave to wrapper/env
    batch_size: int = 0,              # 0 = non-batched (repo default)
    no_stem: bool = False,            # True skips Demucs
    suppress_numerals: bool = False,
    parallel: bool = False,           # use diarize_parallel.py if you want
    timeout: Optional[int] = None,    # seconds; None = no timeout
    use_custom: bool = True,          # prefer diarize_custom.py when present
    keep_temp: bool = False,          # keep/remove temp_outputs* (Demucs, etc.)
    num_speakers: Optional[int] = None, # in case we know how many speakers there should be
) -> DiarizationOutputFiles:
    """
    Run the third-party whisper-diarization pipeline and return normalized outputs.

    Produces (in a per-file work_dir):
    - <stem>.srt
    - <stem>.txt
    - <stem>.csv (utterances; if csv_out not provided, defaults under work_dir)
    Also removes the copied local WAV and, by default, temp_outputs* folders.
    """
    # Make output dir if it doesn't already exist
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    return run_whisper_diarization_repo(
        audio_path=input_audio,
        out_dir=out_dir,
        repo_dir=repo_dir,
        whisper_model=whisper_model,
        language=language,
        device=device,
        batch_size=batch_size,
        no_stem=no_stem,
        suppress_numerals=suppress_numerals,
        parallel=parallel,
        timeout=timeout,
        use_custom=use_custom,
        keep_temp=keep_temp,
        num_speakers=num_speakers,
    )