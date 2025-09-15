from __future__ import annotations
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional
from importlib.resources import files, as_file
from contextlib import ExitStack

def _resolve_vendored_repo_dir() -> Path:
    # This returns a Traversable pointing at the directory inside your wheel.
    return files("chopshop.audio.diarizer").joinpath("whisper-diarization")

@dataclass
class DiarizationOutputFiles:
    work_dir: Path
    raw_files: Dict[str, Path]    # {"srt": ..., "txt": ..., "csv": ...}
    speaker_wavs: Dict[str, Path] # kept for compatibility; always {} here

def _run_repo_script(
    repo_dir: Path,
    audio_path: Path,
    work_dir: Path,
    whisper_model: str,
    language: Optional[str],
    device: Optional[str],
    batch_size: int,
    no_stem: bool,
    suppress_numerals: bool,
    parallel: bool,
    timeout: Optional[int],
    use_custom: bool,
    csv_out: Optional[Path],
    num_speakers: Optional[int],
) -> None:

    script = (
        "diarize_custom.py" if (use_custom and (repo_dir / "diarize_custom.py").exists())
        else ("diarize_parallel.py" if parallel else "diarize.py")
    )
    cmd = [sys.executable, str((repo_dir / script).resolve()), "-a", str(audio_path)]

    if whisper_model:
        cmd += ["--whisper-model", whisper_model]
    if language:
        cmd += ["--language", language]
    if device:
        cmd += ["--device", device]
    
    cmd += ["--batch-size", str(batch_size)]   # 0 == non-batched
    
    if no_stem:
        cmd += ["--no-stem"]
    if suppress_numerals:
        cmd += ["--suppress_numerals"]

    cmd += ["--csv-out", str(csv_out)]

    if num_speakers is not None:
        cmd += ["--num-speakers", str(num_speakers)]
    
    

    work_dir.mkdir(parents=True, exist_ok=True)

    # Prepare environment (ensure CPU really hides GPUs; on CUDA add pip cuDNN path)
    env = os.environ.copy()
    if (device or "").lower() == "cpu":
        env.update({"CUDA_VISIBLE_DEVICES": "", "USE_CUDA": "0", "FORCE_CPU": "1"})
    else:
        try:
            import nvidia.cudnn, pathlib
            cudnn_lib = str(pathlib.Path(nvidia.cudnn.__file__).with_name("lib"))
            env["LD_LIBRARY_PATH"] = cudnn_lib + ":" + env.get("LD_LIBRARY_PATH", "")
        except Exception:
            pass

    subprocess.run(cmd, cwd=work_dir, check=True, timeout=timeout, env=env)

def _guess_outputs_from_stem(work_dir: Path, stem: str) -> Dict[str, Path]:
    exts = ["srt", "txt", "csv"]
    out: Dict[str, Path] = {}
    for ext in exts:
        p = work_dir / f"{stem}.{ext}"
        if p.exists():
            out[ext] = p
    return out

def _cleanup_temps(work_dir: Path, keep_temp: bool) -> None:
    if keep_temp:
        return
    # diarize.py & demucs write under CWD (we run with cwd=work_dir)
    for d in work_dir.glob("temp_outputs*"):
        try:
            shutil.rmtree(d, ignore_errors=True)
        except Exception:
            pass

def run_whisper_diarization_repo(
    audio_path: str | Path,
    out_dir: str | Path,
    *,
    repo_dir: str | Path | None = None,      # ← now Optional
    whisper_model: str = "medium.en",
    language: Optional[str] = None,
    device: Optional[str] = None,            # "cuda" / "cpu"
    batch_size: int = 0,
    no_stem: bool = False,
    suppress_numerals: bool = False,
    parallel: bool = False,
    timeout: Optional[int] = None,
    use_custom: bool = True,
    keep_temp: bool = False,
    num_speakers: Optional[int] = None,
) -> DiarizationOutputFiles:
    """
    Wraps MahmoudAshraf97/whisper-diarization and normalizes outputs.
    Produces: .txt, .srt, .csv in a per-file work_dir, then deletes the local WAV copy
    and (by default) purges temp_outputs* folders.
    """
    audio_path = Path(audio_path).resolve()
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Isolated working folder
    work_dir = out_dir / f"{audio_path.stem}"
    work_dir.mkdir(parents=True, exist_ok=True)

    # Copy input audio next to outputs so the CLI can use simple relative paths
    local_audio = work_dir / audio_path.name
    if not local_audio.exists():
        shutil.copy2(audio_path, local_audio)

    # CSV default: <work_dir>/<stem>.csv
    csv_path = work_dir / f"{local_audio.stem}.csv"

    # Resolve path to vendored repo (or use user-supplied path)
    with ExitStack() as stack:
        if repo_dir is None:
            repo_trav = _resolve_vendored_repo_dir()
            repo_dir_path = stack.enter_context(as_file(repo_trav))  # real FS path
        else:
            repo_dir_path = Path(repo_dir).resolve()

        # Validate the script exists inside the repo
        script_name = ("diarize_custom.py" if (use_custom and (repo_dir_path / "diarize_custom.py").exists())
                       else ("diarize_parallel.py" if parallel else "diarize.py"))
        script_path = (repo_dir_path / script_name)
        if not script_path.exists():
            raise FileNotFoundError(f"Expected script not found: {script_path}")

        # Run the repo script (cwd = work_dir so temp_outputs land there)
        _run_repo_script(
            repo_dir=repo_dir_path,
            audio_path=local_audio,
            work_dir=work_dir,
            whisper_model=whisper_model,
            language=language,
            device=device,
            batch_size=batch_size,
            no_stem=no_stem,
            suppress_numerals=suppress_numerals,
            parallel=parallel,
            timeout=timeout,
            use_custom=use_custom,
            csv_out=csv_path,
            num_speakers=num_speakers,
        )

    # Tidy temp dirs
    _cleanup_temps(work_dir, keep_temp)

    # Collect outputs (.txt/.srt/.csv)
    raw = _guess_outputs_from_stem(work_dir, local_audio.stem)

    # Remove the copied WAV now that we're done
    try:
        if local_audio.exists():
            local_audio.unlink()
    except Exception:
        pass

    return DiarizationOutputFiles(work_dir=work_dir, raw_files=raw, speaker_wavs={})

