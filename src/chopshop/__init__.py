from .ChopShop import ChopShop

# Let's import all of our methods

# audio stuff
from .class_methods.diarize_with_thirdparty import diarize_with_thirdparty
ChopShop.diarize_with_thirdparty = diarize_with_thirdparty

from .class_methods.export_whisper_embeddings import export_whisper_embeddings
ChopShop.export_whisper_embeddings = export_whisper_embeddings

from .class_methods.split_audio_streams import split_audio_streams
ChopShop.split_audio_streams = split_audio_streams

from .class_methods.split_wav_by_speaker import split_wav_by_speaker
ChopShop.split_wav_by_speaker = split_wav_by_speaker


# text stuff
from .class_methods.analyze_with_dictionaries import analyze_with_dictionaries
ChopShop.analyze_with_dictionaries = analyze_with_dictionaries

from .class_methods.gather_text import gather_text
ChopShop.gather_text = gather_text

from .class_methods.analyze_with_archetypes import analyze_with_archetypes
ChopShop.analyze_with_archetypes = analyze_with_archetypes