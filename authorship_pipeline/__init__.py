from .attribution import EvaluationSummary, StylometryModel, evaluate_model, score_chunk, train_model
from .corpus import ProcessedChunk, SourceDocument, ensure_nltk_resources, load_local_corpus, prepare_chunks, split_chunks_by_author

__all__ = [
    "EvaluationSummary",
    "ProcessedChunk",
    "SourceDocument",
    "StylometryModel",
    "ensure_nltk_resources",
    "evaluate_model",
    "load_local_corpus",
    "prepare_chunks",
    "score_chunk",
    "split_chunks_by_author",
    "train_model",
]