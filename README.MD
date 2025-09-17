# ChopShop

![ChopShop header](img/chopshop.png)

A toolkit for turning messy A/V and text into clean, analysis-ready artifacts and features. Think of it as a small pit crew for your data: split → diarize → transcribe → gather text → extract features (dictionaries, archetypes, whisper embeddings) — with predictable filenames and folders.
And, also, everything in-between.

**Status:** early WIP. It works, but expect rough edges and occasional breaking changes.&#x20;

---

## What it does (high level)

* **Audio from video** — pull each audio stream from a container into WAV.
* **Diarize + transcribe** — wrapper around Mahmoud Ashraf's `whisper-diarization` (CSV/SRT/TXT outputs).
* **Per-speaker WAVs** — cut a source WAV into one file per speaker using the transcript.
* **Whisper encoder embeddings** — segment-level embeddings (and general audio modes) via Faster-Whisper (CTranslate2).
* **Text gatherer** — stream/scale a CSV or folder of `.txt` into a single “analysis-ready” CSV (optionally grouped).
* **Feature extraction**

  * **Dictionary / ContentCoder** across any number of dictionaries → one wide CSV with stable column order.
  * **Archetypes** using `archetypes` (sentence-transformer) → one CSV mirroring your analysis-ready file name.
* **Predictable outputs** — if you don't provide an output path, ChopShop writes to `./features/<kind>/<filename>.csv`, where `<filename>` comes from your analysis-ready CSV (so grouping/concat choices are visible in the name).

---

## The API you'll use

ChopShop exposes namespaced sub-APIs for clarity:

```python
from chopshop import ChopShop
cs = ChopShop()

# Audio
wav_paths = cs.audio.extract_wavs_from_video(input_path="input.mp4", output_dir="audio_out/")
tp = cs.diarizer.with_thirdparty(audio_path=wav_paths[0], out_dir="transcripts/", whisper_model="small", device="cuda")
cs.audio.split_wav_by_speaker(source_wav=wav_paths[0], transcript_csv=tp["csv"], out_dir="per_speaker/")

# Embeddings (transcript-driven OR general-audio)
cs.audio.export_whisper_embeddings(source_wav=wav_paths[0], transcript_csv=tp["csv"])          # segment CSV
cs.audio.export_whisper_embeddings(source_wav=wav_paths[0], strategy="nonsilent", aggregate="mean")  # general audio

# Text gather → Dictionaries
feat_csv = cs.text.analyze_with_dictionaries(
    csv_path="transcripts/session.csv",
    dict_paths=["dictionaries/LIWC-22.dicx", "dictionaries/empath-default.dicx"],
    text_cols=["text"], id_cols=["speaker"], group_by=["speaker"], delimiter=",",
)

# Text gather → Archetypes
arch_csv = cs.text.analyze_with_archetypes(
    csv_path="transcripts/session.csv",
    archetype_csvs=["dictionaries/archetypes/Suicidality.csv", "dictionaries/archetypes/Resilience.csv"],
    text_cols=["text"], id_cols=["speaker"], group_by=["speaker"], delimiter=",",
)
```

**Default output locations**
If you omit `out_features_csv`, ChopShop writes to:

* Dictionaries → `./features/dictionary/<analysis_ready_filename>.csv`
* Archetypes → `./features/archetypes/<analysis_ready_filename>.csv`
* Whisper embeddings → `./features/whisper_embed/<analysis_ready_filename>.csv`

The `<analysis_ready_filename>` comes from the text-gather step (e.g., `dataset_grouped_speaker.csv`), or from your provided `analysis_csv`.

---

## CLI (quick hits)

Anything you can do in Python... well, you can also run from the terminal.

```bash
# Gather text from a CSV (auto-named output if --out omitted)
python -m chopshop.helpers.text_gather \
  --csv transcripts/session.csv \
  --text-col text --group-by speaker --delimiter , --encoding utf-8-sig

# Diarization (wrapper; writes CSV/SRT/TXT under out_dir/<basename>/)
python -m chopshop.audio.diarize_with_thirdparty \
  --audio_path audio/session_a1.wav --out_dir transcripts/ --whisper_model small --device cuda --num_speakers 2

# Whisper embeddings (general audio; nonsilent with mean pool)
python -m chopshop.audio.extract_whisper_embeddings \
  --source_wav audio/session_a1.wav \
  --strategy nonsilent --aggregate mean --output_dir features/whisper_embed/
```

---

## Installation

A fresh virtual environment is strongly recommended.

```bash
python -m venv venv-chopshop
source venv-chopshop/bin/activate
```

### Quick path (when available)

```bash
pip install "chopshop[diarization,cuda]"
```

Then install the three git extras used by the diarization wrapper:

```bash
pip install git+https://github.com/MahmoudAshraf97/demucs.git
pip install git+https://github.com/oliverguhr/deepmultilingualpunctuation.git
pip install git+https://github.com/MahmoudAshraf97/ctc-forced-aligner.git
```

Install PyTorch built for **CUDA 12.4** (the stack ChopShop targets):

```bash
pip install --force-reinstall --no-cache-dir \
  torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
  --index-url https://download.pytorch.org/whl/cu124
```

And ensure **FFmpeg** is on your `PATH` (Ubuntu: `sudo apt-get install ffmpeg`, macOS: `brew install ffmpeg`).

### Manual stack (same versions, explicit)

```bash
# Core pieces
pip install "faster-whisper>=1.1.0"
pip install "nemo-toolkit[asr]>=2.dev"
pip install git+https://github.com/MahmoudAshraf97/demucs.git
pip install git+https://github.com/oliverguhr/deepmultilingualpunctuation.git
pip install git+https://github.com/MahmoudAshraf97/ctc-forced-aligner.git

# cuDNN user-space libs (CUDA 12)
pip install -U nvidia-cudnn-cu12

# PyTorch for CUDA 12.4
pip install --force-reinstall --no-cache-dir \
  torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
  --index-url https://download.pytorch.org/whl/cu124

# Text features
pip install contentcoder archetyper
```

> If you hit CUDA/cuDNN loader errors, it usually means the runtime and wheel builds don't match. Keep CUDA **12.4**, `cu124` wheels, and cuDNN 9 aligned.

---



## Troubleshooting quickies

* **Delimiter or encoding issues** when gathering text
  Pass `--delimiter` and `--encoding` parameters explicitly for CSV inputs, just to be safe. If you run into any errors, try using `--delimiter ,` and `--encoding utf-8-sig` as a starting point.

* **Diarizer ignores `--num_speakers`**
  Use the custom entrypoint (enabled by default) which wires `num_speakers` through properly... for now. If needed, pin `min_num_speakers == max_num_speakers == N`.

* **cuDNN / CUDA symbol errors**
  Mismatched CUDA/cuDNN vs wheel builds. Reinstall the `cu124` PyTorch wheels and `nvidia-cudnn-cu12`.

* **Embeddings subprocess fails**
  Use `device=cpu` to rule out GPU issues; or set `CHOPSHOP_DEBUG=1` to surface more logs.

---

## Credits

* Diarization stack adapted from **Mahmoud Ashraf's** excellent [`whisper-diarization`](https://github.com/MahmoudAshraf97/whisper-diarization).
* Dictionaries via **ContentCoder-Py**; archetypes via **archetypes** (sentence-transformers). Well, okay, I wrote those. But I didn't know at the time that they'd be so handy. So... good job, former me.

---

## License & status

MIT (see LICENSE). Active WIP; APIs and default paths may (read: will) shift as the project settles — release notes will most likely call out breaking changes.

*Happy chopping.*
