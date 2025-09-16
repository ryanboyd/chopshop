from typing import Optional, Literal, Union, Sequence
from pathlib import Path

from ..text.text_gather import csv_to_analysis_ready_csv, txt_folder_to_analysis_ready_csv

def gather_text(
    self,
    *,
    # choose exactly one of these:
    csv_path: Optional[Union[str, Path]] = None,
    txt_dir: Optional[Union[str, Path]] = None,

    # common output
    out_csv: Union[str, Path],

    # CSV mode options
    text_cols: Optional[Sequence[str]] = None,
    id_cols: Optional[Sequence[str]] = None,
    mode: Literal["concat", "separate"] = "concat",
    group_by: Optional[Sequence[str]] = None,
    delimiter: Optional[str] = None,
    encoding: str = "utf-8-sig",
    joiner: str = " ",

    # external grouping tuning (CSV mode w/ group_by)
    num_buckets: int = 1024,
    max_open_bucket_files: int = 64,
    tmp_root: Optional[Union[str, Path]] = None,

    # TXT folder mode options
    recursive: bool = False,
    pattern: str = "*.txt",
    id_from: Literal["stem", "name", "path"] = "stem",
    include_source_path: bool = True,
) -> Path:
    """
    Build an analysis-ready CSV (schema: text_id,text[,source_col|source_path]) from:
    - a large/unsorted CSV (streaming, optional external grouping), OR
    - a folder of .txt files (streaming).

    Exactly one of `csv_path` or `txt_dir` must be provided.

    Returns:
        Path to the written analysis-ready CSV.

    Examples:
        # CSV → analysis CSV (no grouping)
        cs.gather_text(
            csv_path="data/raw.csv",
            out_csv="work/analysis_ready.csv",
            text_cols=["utterance", "notes"],
            id_cols=["session_id"],
            mode="concat",
        )

        # CSV → analysis CSV with grouping by speaker (unsorted input OK)
        cs.gather_text(
            csv_path="data/transcripts.csv",
            out_csv="work/by_speaker.csv",
            text_cols=["utterance"],
            group_by=["speaker_id"],
            mode="concat",
            num_buckets=512,
        )

        # Folder of .txt files → analysis CSV
        cs.gather_text(
            txt_dir="notes/",
            out_csv="work/notes.csv",
            recursive=True,
            id_from="path",
        )
    """
    out_csv = Path(out_csv)

    # Validate mutually exclusive inputs
    if (csv_path is None) == (txt_dir is None):
        raise ValueError("Provide exactly one of csv_path or txt_dir.")

    # Route: CSV mode
    if csv_path is not None:
        if not text_cols:
            raise ValueError("When using csv_path, you must provide non-empty `text_cols`.")
        return Path(
            csv_to_analysis_ready_csv(
                csv_path=csv_path,
                out_csv=out_csv,
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

    # Route: TXT folder mode
    return Path(
        txt_folder_to_analysis_ready_csv(
            root_dir=txt_dir,
            out_csv=out_csv,
            recursive=recursive,
            pattern=pattern,
            encoding=encoding,
            id_from=id_from,
            include_source_path=include_source_path,
        )
    )
