# ChopShop.py
from typing import Optional
from pathlib import Path
from . import extract_wav_from_video
from .diarizer import whisper_diar_wrapper

"""
ChopShop: A toolkit for breaking down input files (e.g., video)
into their constituent parts (video, audio, text) and generating
massive feature sets. Currently just a placeholder class.
"""

class ChopShop:
    def __init__(self):
        """Initialize the ChopShop instance."""
        pass

    def split_audio_streams(self, input_path, output_dir, **kwargs):
        """
        Split all audio streams from a video file into WAVs.

        Parameters
        ----------
        input_path : str | PathLike
            Path to the input video file.
        output_dir : str | PathLike
            Directory to write the WAV files.
        **kwargs
            Extra keyword args passed to split_audio_streams_to_wav,
            e.g. sample_rate, bit_depth, overwrite.

        Returns
        -------
        List[str]
            Paths to the created WAV files.
        """
        return extract_wav_from_video.split_audio_streams_to_wav(input_path, output_dir, **kwargs)
    
    def diarize_with_thirdparty(
        self,
        input_audio: str | Path,
        out_dir: str | Path,
        *,
        repo_dir: str | Path = "diarizer/whisper-diarization",
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
    ) -> whisper_diar_wrapper.DiarizationOutputFiles:
        """
        Run the third-party whisper-diarization pipeline and return normalized outputs.

        Produces (in a per-file work_dir):
        - <stem>.srt
        - <stem>.txt
        - <stem>.csv (utterances; if csv_out not provided, defaults under work_dir)
        Also removes the copied local WAV and, by default, temp_outputs* folders.
        """
        return whisper_diar_wrapper.run_whisper_diarization_repo(
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
    )

    
