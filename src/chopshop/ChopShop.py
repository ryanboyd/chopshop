# ChopShop.py
import os
import importlib
import sys
import shlex
import subprocess
from pathlib import Path
from typing import Optional, Dict, Literal

# other imports
from . import extract_wav_from_video
from .diarizer.whisper_diar_wrapper import DiarizationOutputFiles, run_whisper_diarization_repo
from .split_wav_by_speaker import make_speaker_wavs_from_csv
from .extract_whisper_embeddings import export_segment_embeddings_csv, EmbedConfig

"""
ChopShop: A toolkit for breaking down input files (e.g., video)
into their constituent parts (video, audio, text) and generating
feature sets.
"""

def _abs(p: str | Path) -> str:
    return str(Path(p).resolve())


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


    def export_embeddings(
        self,
        transcript_csv: str | Path,
        source_wav: str | Path,
        *,
        output_dir: str | Path | None = None,
        model_name: str = "base",
        device: Literal["auto", "cuda", "cpu"] = "cuda",
        compute_type: str = "float16",
        time_unit: Literal["auto", "ms", "s", "samples"] = "ms",
        start_col: str = "start_time",
        end_col: str = "end_time",
        speaker_col: str = "speaker",
        sr: int = 16000,
        run_in_subprocess: bool = True,
        extra_env: Optional[dict] = None,
        verbose: bool = True,
    ) -> Path:
        """
        Export Whisper encoder embeddings (one row per transcript segment) to CSV.

        If run_in_subprocess=True (default), spawns a clean Python process that never imports torch,
        so GPU via CTranslate2 works without cuDNN conflicts.

        Returns:
            Path to the written CSV at <output_dir>/<source_stem>_embeddings.csv
            (output_dir defaults to source_wav.parent).
        """
        transcript_csv = Path(transcript_csv)
        source_wav = Path(source_wav)

        # Decide output directory and final CSV path
        out_dir_final = Path(output_dir) if output_dir is not None else source_wav.parent
        out_dir_final.mkdir(parents=True, exist_ok=True)
        output_csv = out_dir_final / f"{source_wav.stem}_embeddings.csv"

        if not run_in_subprocess:
            # In-process path (works if torch hasn't polluted the process)
            from .extract_whisper_embeddings import export_segment_embeddings_csv, EmbedConfig
            cfg = EmbedConfig(
                model_name=model_name,
                device=device,
                compute_type=compute_type,
                time_unit=time_unit,
            )
            csv_path = export_segment_embeddings_csv(
                transcript_csv=transcript_csv,
                source_wav=source_wav,
                output_dir=out_dir_final,
                config=cfg,
                start_col=start_col,
                end_col=end_col,
                speaker_col=speaker_col,
                sr=sr,
            )
            if verbose:
                print(f"Embeddings CSV written to: {csv_path}")
            return Path(csv_path)

        # --- Subprocess path (recommended) ---
        env = os.environ.copy()

        # Honor caller-provided env overrides first
        if extra_env:
            env.update({k: str(v) for k, v in extra_env.items()})

        # CPU isolation vs CUDA libs
        if (device or "").lower() == "cpu":
            env.update({"CUDA_VISIBLE_DEVICES": "", "USE_CUDA": "0", "FORCE_CPU": "1"})
        else:
            # Try to prepend the cuDNN wheel lib dir to LD_LIBRARY_PATH if available
            try:
                import nvidia.cudnn, pathlib  # type: ignore
                cudnn_lib = str(pathlib.Path(nvidia.cudnn.__file__).with_name("lib"))
                env["LD_LIBRARY_PATH"] = cudnn_lib + ":" + env.get("LD_LIBRARY_PATH", "")
            except Exception:
                pass

        # Ensure transformers never pulls torch/TF/Flax in the child
        env.setdefault("TRANSFORMERS_NO_TORCH", "1")
        env.setdefault("TRANSFORMERS_NO_TF", "1")
        env.setdefault("TRANSFORMERS_NO_FLAX", "1")
        # Uncomment for verbose extractor logs:
        # env["CHOPSHOP_DEBUG"] = "1"

        # Call: python -m chopshop.extract_whisper_embeddings ...
        cmd = [
            sys.executable, "-m", "chopshop.extract_whisper_embeddings",
            "--transcript_csv", str(transcript_csv),
            "--source_wav", str(source_wav),
            "--output_dir", str(out_dir_final),  # extractor derives <stem>_embeddings.csv
            "--model_name", model_name,
            "--device", device,
            "--compute_type", compute_type,
            "--time_unit", time_unit,
        ]

        if verbose:
            print("Launching embedding subprocess:")
            print(" ", shlex.join(cmd))

        try:
            subprocess.run(cmd, check=True, env=env)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"Embedding subprocess failed with code {e.returncode}\n"
                f"STDERR:\n{getattr(e, 'stderr', b'').decode(errors='ignore') if hasattr(e, 'stderr') else ''}"
            ) from e

        if verbose:
            print(f"Embeddings CSV written to: {output_csv}")

        return output_csv

