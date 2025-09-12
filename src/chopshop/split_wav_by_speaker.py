# -------------------------
# Functions used to get per-speaker WAVs
# -------------------------

def _merge_spans(spans: List[List[float]], tol: float = 0.05) -> List[List[float]]:
    if not spans:
        return []
    spans = sorted(spans, key=lambda x: (x[0], x[1]))
    merged = [spans[0][:]]
    for s in spans[1:]:
        last = merged[-1]
        if s[0] <= last[1] + tol:  # join touching/overlapping segments
            last[1] = max(last[1], s[1])
        else:
            merged.append(s[:])
    return merged

def _extract_concat(src_wav: Path, spans: List[List[float]], out_wav: Path, sr: int = 16000):
    out_wav = Path(out_wav)
    tmp_dir = out_wav.parent / f".tmp_{out_wav.stem}"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    parts = []
    for i, (st, en) in enumerate(spans):
        dur = max(0.0, en - st)
        if dur <= 0:
            continue
        part = tmp_dir / f"part_{i:05d}.wav"
        cmd = [
            "ffmpeg", "-y",
            "-ss", f"{st:.3f}",
            "-i", str(src_wav),
            "-t", f"{dur:.3f}",
            "-ac", "1", "-ar", str(sr),
            "-vn", "-acodec", "pcm_s16le",
            str(part),
        ]
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        parts.append(part)
    if not parts:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return
    # concat list
    concat_list = tmp_dir / "list.txt"
    with concat_list.open("w", encoding="utf-8") as f:
        for p in parts:
            f.write(f"file '{p.as_posix()}'\n")
    # concat to target
    cmd = [
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0",
        "-i", str(concat_list),
        "-ac", "1", "-ar", str(sr),
        "-acodec", "pcm_s16le",
        str(out_wav),
    ]
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    shutil.rmtree(tmp_dir, ignore_errors=True)

def export_speaker_wavs(vocal_source: str, ssm_obj, speaker_dir: Path, sr: int = 16000) -> Dict[str, Path]:
    speaker_dir = Path(speaker_dir)
    speaker_dir.mkdir(parents=True, exist_ok=True)
    spans_by_spk: Dict[str, List[List[float]]] = {}
    for u in _iter_utterances(ssm_obj):
        spk = u["speaker"] or "SPEAKER_0"
        spans_by_spk.setdefault(spk, []).append([float(u["start_time"]), float(u["end_time"])])
    results: Dict[str, Path] = {}
    for spk, spans in spans_by_spk.items():
        merged = _merge_spans(spans, tol=0.05)
        safe = re.sub(r"[^A-Za-z0-9_]+", "_", spk)
        out_wav = speaker_dir / f"{safe}.wav"
        _extract_concat(Path(vocal_source), merged, out_wav, sr=sr)
        results[spk] = out_wav
    return results

if args.speaker_dir:
    spk_dir = Path(args.speaker_dir)
    wavs = export_speaker_wavs(vocal_target, ssm, spk_dir, sr=args.sr)
    print("[custom] Speaker WAVs:", json.dumps({k: str(v) for k, v in wavs.items()}, indent=2))