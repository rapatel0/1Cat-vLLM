# Trimmed gemma4_unified package for the V100 fork: processing/preprocessing only.
# The HF modeling file is intentionally omitted — vLLM uses its own ported model
# (vllm/model_executor/models/gemma4_unified.py). All base classes these processors
# need (TorchvisionBackend, SequenceFeatureExtractor, BaseImageProcessorFast,
# MultiModalData, VideosKwargs, VideoInput) already exist in transformers 5.5, so a
# plain (non-lazy) __init__ avoids define_import_structure pulling in modeling.
from .feature_extraction_gemma4_unified import Gemma4UnifiedAudioFeatureExtractor
from .image_processing_gemma4_unified import Gemma4UnifiedImageProcessor
from .video_processing_gemma4_unified import Gemma4UnifiedVideoProcessor
from .processing_gemma4_unified import Gemma4UnifiedProcessor

__all__ = [
    "Gemma4UnifiedAudioFeatureExtractor",
    "Gemma4UnifiedImageProcessor",
    "Gemma4UnifiedVideoProcessor",
    "Gemma4UnifiedProcessor",
]
