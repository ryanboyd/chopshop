from pathlib import Path
from typing import Optional, Literal, Union, Sequence, Iterable, Tuple
import csv

from .dictionary_analyzers import multi_archetype_analyzer as maa
from ..helpers.text_gather import (
    csv_to_analysis_ready_csv,
    txt_folder_to_analysis_ready_csv,
)

def analyze_with_archetypes(
    *,
    # ----- Input source (choose exactly one, OR pass analysis_csv to skip gathering) -----
    csv_path: Optional[Union[str, Path]] = None,
    txt_dir: Optional[Union[str, Path]] = None,
    analysis_csv: Optional[Union[str, Path]] = None,   # <- NEW: skip gathering if provided

    # ----- Output -----
    out_features_csv: Optional[Union[str, Path]] = None,

    # ----- Archetype CSVs (one or more) -----
    archetype_csvs: Sequence[Union[str, Path]],

    # ====== SHARED I/O OPTIONS ======
    encoding: str = "utf-8-sig",
    delimiter: str = ",",

    # ====== CSV GATHER OPTIONS (when csv_path is provided) ======
    text_cols: Sequence[str] = ("text",),
    id_cols: Optional[Sequence[str]] = None,
    mode: Literal["concat", "separate"] = "concat",
    group_by: Optional[Sequence[str]] = None,
    joiner: str = " ",
    num_buckets: int = 512,
    max_open_bucket_files: int = 64,
    tmp_root: Optional[Union[str, Path]] = None,

    # ====== TXT FOLDER GATHER OPTIONS (when txt_dir is provided) ======
    recursive: bool = True,
    pattern: str = "*.txt",
    id_from: Literal["stem", "name", "path"] = "stem",
    include_source_path: bool = True,

    # ====== Archetyper scoring options ======
    model_name: str = "sentence-transformers/all-roberta-large-v1",
    mean_center_vectors: bool = True,
    fisher_z_transform: bool = False,
    rounding: int = 4,
) -> Path:
    """
    Build/accept an analysis-ready CSV (columns: text_id,text) and compute archetype scores
    via multi_archetype_analyzer, writing one wide CSV:
      text_id, WC, <fileprefix>__<ArchetypeName>, ...
    """

    # 1) Use analysis-ready CSV if given; otherwise gather from csv_path or txt_dir
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

    # 2) Validate archetype CSVs
    archetype_csvs = [Path(p) for p in archetype_csvs]
    if not archetype_csvs:
        raise ValueError("archetype_csvs must contain at least one CSV file.")
    for p in archetype_csvs:
        if not p.exists():
            raise FileNotFoundError(f"Archetype CSV not found: {p}")

    # 3) Stream (text_id, text) → middle layer → features CSV
    def _iter_items_from_csv(path: Path, *, id_col: str = "text_id", text_col: str = "text") -> Iterable[Tuple[str, str]]:
        with path.open("r", newline="", encoding=encoding) as f:
            reader = csv.DictReader(f, delimiter=delimiter)
            if id_col not in reader.fieldnames or text_col not in reader.fieldnames:
                raise ValueError(
                    f"Expected columns '{id_col}' and '{text_col}' in {path}; found {reader.fieldnames}"
                )
            for row in reader:
                yield str(row[id_col]), (row.get(text_col) or "")

    
    # 1b) Decide default features path if not provided:
    #     <analysis_ready_dir>/features/archetypes/<analysis_ready_filename>
    if out_features_csv is None:
        out_features_csv = Path.cwd() / "features" / "archetypes" / analysis_ready.name
    out_features_csv = Path(out_features_csv)
    out_features_csv.parent.mkdir(parents=True, exist_ok=True)

    maa.analyze_texts_to_csv(
        items=_iter_items_from_csv(analysis_ready),
        archetype_csvs=archetype_csvs,
        out_csv=out_features_csv,
        model_name=model_name,
        mean_center_vectors=mean_center_vectors,
        fisher_z_transform=fisher_z_transform,
        rounding=rounding,
        encoding=encoding,
        delimiter=delimiter,
        id_col_name="text_id",
    )

    return out_features_csv
