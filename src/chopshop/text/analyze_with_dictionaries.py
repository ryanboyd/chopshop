from pathlib import Path
from typing import Optional, Literal, Union, Sequence, Iterable, Tuple
import csv

from .dictionary_analyzers import multi_dict_analyzer as mda
from ..helpers.text_gather import (
    csv_to_analysis_ready_csv,
    txt_folder_to_analysis_ready_csv,
)

def analyze_with_dictionaries(
    *,
    # ----- Input source (choose exactly one, or pass analysis_csv directly) -----
    csv_path: Optional[Union[str, Path]] = None,
    txt_dir: Optional[Union[str, Path]] = None,
    analysis_csv: Optional[Union[str, Path]] = None,  # if provided, gathering is skipped

    # ----- Output -----
    out_features_csv: Optional[Union[str, Path]] = None,

    # ----- Dictionaries -----
    dict_paths: Sequence[Union[str, Path]],

    # ====== SHARED I/O OPTIONS ======
    encoding: str = "utf-8-sig",

    # ====== CSV GATHER OPTIONS ======
    # Only used when csv_path is provided
    text_cols: Sequence[str] = ("text",),
    id_cols: Optional[Sequence[str]] = None,
    mode: Literal["concat", "separate"] = "concat",
    group_by: Optional[Sequence[str]] = None,
    delimiter: str = ",",
    joiner: str = " ",
    num_buckets: int = 512,
    max_open_bucket_files: int = 64,
    tmp_root: Optional[Union[str, Path]] = None,

    # ====== TXT FOLDER GATHER OPTIONS ======
    # Only used when txt_dir is provided
    recursive: bool = True,
    pattern: str = "*.txt",
    id_from: Literal["stem", "name", "path"] = "stem",
    include_source_path: bool = True,

    # ====== ANALYZER OPTIONS (passed through to ContentCoder) ======
    relative_freq: bool = True,
    drop_punct: bool = True,
    rounding: int = 4,
    retain_captures: bool = False,
    wildcard_mem: bool = True,
) -> Path:
    """Gather text into an analysis-ready CSV and pipe it to multi_dict_analyzer,
    which writes one wide CSV with globals once (from first dict) then per-dict blocks.
    Returns the path to `out_features_csv`.
    """

    # 1) Produce or accept the analysis-ready CSV (must have columns: text_id,text)
    if analysis_csv is not None:
        analysis_ready = Path(analysis_csv)
        if not analysis_ready.exists():
            raise FileNotFoundError(f"analysis_csv not found: {analysis_ready}")
    else:
        if (csv_path is None) == (txt_dir is None):
            raise ValueError("Provide exactly one of csv_path or txt_dir (or pass analysis_csv).")

        if csv_path is not None:
            analysis_ready = Path(
                csv_to_analysis_ready_csv(
                    csv_path=csv_path,
                    text_cols=list(text_cols),
                    id_cols=list(id_cols) if id_cols else None,
                    mode=mode,
                    group_by=list(group_by) if group_by else None,
                    delimiter=delimiter,
                    encoding=encoding,
                    joiner=joiner,
                    num_buckets=num_buckets,
                    max_open_bucket_files=max_open_bucket_files,
                    tmp_root=tmp_root,
                )
            )
        else:
            analysis_ready = Path(
                txt_folder_to_analysis_ready_csv(
                    root_dir=txt_dir,
                    recursive=recursive,
                    pattern=pattern,
                    encoding=encoding,
                    id_from=id_from,
                    include_source_path=include_source_path,
                )
            )

    # 1b) Decide default features path if not provided:
    #     <cwd>/features/dictionary/<analysis_ready_filename>
    if out_features_csv is None:
        out_features_csv = Path.cwd() / "features" / "dictionary" / analysis_ready.name
    out_features_csv = Path(out_features_csv)
    out_features_csv.parent.mkdir(parents=True, exist_ok=True)


    # 2) Validate dictionaries
    dict_paths = [Path(p) for p in dict_paths]
    if not dict_paths:
        raise ValueError("dict_paths must contain at least one dictionary file.")
    for p in dict_paths:
        if not p.exists():
            raise FileNotFoundError(f"Dictionary not found: {p}")

    # 3) Stream the analysis-ready CSV into the analyzer → features CSV
    def _iter_items_from_csv(
        path: Path, *, id_col: str = "text_id", text_col: str = "text"
    ) -> Iterable[Tuple[str, str]]:
        with path.open("r", newline="", encoding=encoding) as f:
            reader = csv.DictReader(f, delimiter=delimiter)
            if id_col not in reader.fieldnames or text_col not in reader.fieldnames:
                raise ValueError(
                    f"Expected columns '{id_col}' and '{text_col}' in {path}; found {reader.fieldnames}"
                )
            for row in reader:
                yield str(row[id_col]), (row.get(text_col) or "")

    # Use multi_dict_analyzer as the middle layer (new API)
    mda.analyze_texts_to_csv(
        items=_iter_items_from_csv(analysis_ready),
        dict_files=dict_paths,
        out_csv=out_features_csv,
        relative_freq=relative_freq,
        drop_punct=drop_punct,
        rounding=rounding,
        retain_captures=retain_captures,
        wildcard_mem=wildcard_mem,
        id_col_name="text_id",
        encoding=encoding,
    )

    return out_features_csv
