import os
import sys
import shlex
import subprocess
from pathlib import Path
from typing import Optional, Literal

def export_whisper_embeddings(
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

    If run_in_subprocess=True, spawns a clean Python process (avoids cuDNN conflicts).
    Returns: <output_dir>/<source_stem>_embeddings.csv (default output_dir = source_wav.parent).
    """
    # Resolve absolute paths so the child process is independent of CWD
    transcript_csv = Path(transcript_csv).resolve()
    source_wav = Path(source_wav).resolve()

    out_dir_final = Path(output_dir).resolve() if output_dir is not None else source_wav.parent
    out_dir_final.mkdir(parents=True, exist_ok=True)
    output_csv = out_dir_final / f"{source_wav.stem}_embeddings.csv"

    if not run_in_subprocess:
        # In-process execution (only works if torch/cuDNN aren't already imported... maybe?)
        from ..audio.extract_whisper_embeddings import export_segment_embeddings_csv, EmbedConfig
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
    if extra_env:
        env.update({k: str(v) for k, v in extra_env.items()})

    # CPU isolation vs CUDA libs
    if (device or "").lower() == "cpu":
        env.update({"CUDA_VISIBLE_DEVICES": "", "USE_CUDA": "0", "FORCE_CPU": "1"})
    else:
        # Prepend cuDNN wheel's lib dir if available (helps with "libcudnn_ops" not found)
        try:
            import nvidia.cudnn, pathlib  # type: ignore
            cudnn_lib = str(pathlib.Path(nvidia.cudnn.__file__).with_name("lib"))
            env["LD_LIBRARY_PATH"] = cudnn_lib + ":" + env.get("LD_LIBRARY_PATH", "")
        except Exception:
            pass

    # Keep transformers from pulling heavy backends in the child, because we don't need them
    env.setdefault("TRANSFORMERS_NO_TORCH", "1")
    env.setdefault("TRANSFORMERS_NO_TF", "1")
    env.setdefault("TRANSFORMERS_NO_FLAX", "1")

    cmd = [
        sys.executable, "-m", "chopshop.audio.extract_whisper_embeddings",
        "--transcript_csv", str(transcript_csv),
        "--source_wav", str(source_wav),
        "--output_dir", str(out_dir_final),    # child derives <stem>_embeddings.csv
        "--model_name", model_name,
        "--device", device,
        "--compute_type", compute_type,
        "--time_unit", time_unit,
    ]

    if verbose:
        print("Launching embedding subprocess:")
        print(" ", shlex.join(cmd))

    try:
        # Capture output so failures show useful diagnostics
        res = subprocess.run(cmd, check=True, env=env, capture_output=True, text=True)
        if verbose and res.stdout:
            print(res.stdout.strip())
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"Embedding subprocess failed with code {e.returncode}\n"
            f"STDOUT:\n{(e.stdout or '').strip()}\n\nSTDERR:\n{(e.stderr or '').strip()}"
        ) from e

    if not output_csv.exists():
        # Defensive: child should have created it
        raise FileNotFoundError(f"Expected embeddings CSV not found: {output_csv}")

    if verbose:
        print(f"Embeddings CSV written to: {output_csv}")

    return output_csv
