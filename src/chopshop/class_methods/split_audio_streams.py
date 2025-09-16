from pathlib import Path

from ..audio import extract_wav_from_video

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
    # Make output dir if it doesn't already exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    return extract_wav_from_video.split_audio_streams_to_wav(input_path, output_dir, **kwargs)