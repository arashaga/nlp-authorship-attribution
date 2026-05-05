from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from statistics import fmean, stdev

from .corpus import ProcessedChunk
from .features import (
    PosTrigram,
    build_function_word_vocabulary,
    build_pos_trigram_vocabulary,
    cosine_similarity,
    euclidean_distance,
    function_word_vector,
    lexical_profile,
    pos_trigram_vector,
)


"""Authorship attribution tools for comparing writing style.

This module trains a small stylometry model from already processed text chunks.
The main idea is to describe each author's style with function words, lexical
features, and part-of-speech trigrams, then compare new chunks against those
author profiles.
"""

"""
This is how it looks like when this object is populated it will be accessed
as model.author_profiles

AuthorProfile(
    author="Jane Austen",
    delta_centroid={
        "the": 0.35,
        "and": 0.80,
        "of": -0.20,
    },
    lexical_profile={
        "avg_word_length": 4.6,
        "type_token_ratio": 0.42,
    },
    pos_trigram_profile={
        ("DET", "NOUN", "VERB"): 0.03,
    },
    training_chunks=12,
)
"""

@dataclass(frozen=True)
class AuthorProfile:
    """Stores the average writing-style measurements for one author."""

    author: str
    delta_centroid: dict[str, float] # average Burrows's Delta-style function-word scores
    lexical_profile: dict[str, float]
    pos_trigram_profile: dict[PosTrigram, float]
    training_chunks: int


@dataclass(frozen=True)
class AttributionResult:
    """When you give the model a new unknown chunk, it compares 
    that chunk against each AuthorProfile. For each author, it creates an
    AttributionResult showing how close the unknown chunk was to that author."""

    author: str
    delta_score: float
    pos_similarity: float
    lexical_distance: float


@dataclass(frozen=True)
class ChunkScore:
    """Stores the predicted author and full ranking for one text chunk.
    for the ranking here:
    First, prefer the lowest delta_score.
    If there is a tie, prefer the highest pos_similarity.
    If there is still a tie, prefer the lowest lexical_distance."""

    chunk: ProcessedChunk
    predicted_author: str
    ranking: list[AttributionResult]


@dataclass(frozen=True)
class EvaluationSummary:
    """Summarizes how well the model performed on the test chunks."""

    total_chunks: int
    correct_predictions: int #is the number of test chunks that the model guessed correctly.
    accuracy: float
    chunk_scores: list[ChunkScore]


@dataclass(frozen=True)
class StylometryModel:
    """Stores all learned information needed to score new chunks."""

    function_words: tuple[str, ...]
    corpus_means: dict[str, float]
    corpus_stdevs: dict[str, float]
    pos_trigrams: tuple[PosTrigram, ...]
    author_profiles: dict[str, AuthorProfile]


def _mean_mapping(
    mappings: Sequence[Mapping[object, float]],
) -> dict[object, float]:
    """Average a list of dictionary-like feature profiles by matching keys."""

    if not mappings:
        raise ValueError("Cannot average an empty sequence of mappings.")
    keys = tuple(mappings[0].keys())
    return {key: fmean(mapping[key] for mapping in mappings) for key in keys}


def _z_score_vector(
    raw_vector: Mapping[str, float],
    means: Mapping[str, float],
    stdevs: Mapping[str, float],
) -> dict[str, float]:
    """Convert raw feature values into z-scores for Burrows's Delta.

    A z-score shows whether a feature is above or below the corpus average.
    This makes common and uncommon function words easier to compare fairly.
    """

    return {
        feature: (raw_vector[feature] - means[feature]) / stdevs[feature]
        for feature in means
    }


def train_model(
    training_chunks: Sequence[ProcessedChunk],
    function_word_features: int = 40,
    pos_trigram_features: int = 30,
) -> StylometryModel:
    """Train the authorship model from known chunks.

    The training step finds useful function words, calculates corpus-wide
    averages, and then builds one profile for each author. Each profile becomes
    the baseline that unknown chunks are compared against later.
    """

    if not training_chunks:
        raise ValueError("Training data is empty.")

    # Start with the most useful function words, then remove any that do not vary.
    initial_function_words = build_function_word_vocabulary(training_chunks, max_features=function_word_features)
    raw_function_vectors = {
        id(chunk): function_word_vector(chunk, initial_function_words)
        for chunk in training_chunks
    }

    usable_function_words: list[str] = []
    corpus_means: dict[str, float] = {}
    corpus_stdevs: dict[str, float] = {}
    for feature in initial_function_words:
        values = [raw_function_vectors[id(chunk)][feature] for chunk in training_chunks]
        feature_stdev = stdev(values) if len(values) > 1 else 0.0
        if feature_stdev == 0.0:
            continue
        usable_function_words.append(feature)
        corpus_means[feature] = fmean(values)
        corpus_stdevs[feature] = feature_stdev

    if not usable_function_words:
        raise ValueError("No usable function-word features were found. Increase the corpus size.")

    # Rebuild vectors using only features that have a usable standard deviation.
    raw_function_vectors = {
        id(chunk): function_word_vector(chunk, usable_function_words)
        for chunk in training_chunks
    }
    z_score_vectors = {
        id(chunk): _z_score_vector(raw_function_vectors[id(chunk)], corpus_means, corpus_stdevs)
        for chunk in training_chunks
    }

    pos_trigrams = build_pos_trigram_vocabulary(training_chunks, max_features=pos_trigram_features)

    # Group chunks by author so each author gets one combined style profile.
    by_author: dict[str, list[ProcessedChunk]] = defaultdict(list)
    for chunk in training_chunks:
        by_author[chunk.author].append(chunk)

    author_profiles: dict[str, AuthorProfile] = {}
    for author, author_chunks in sorted(by_author.items()):
        author_z_vectors = [z_score_vectors[id(chunk)] for chunk in author_chunks]
        author_lexical_profiles = [lexical_profile(chunk) for chunk in author_chunks]
        author_pos_profiles = [pos_trigram_vector(chunk, pos_trigrams) for chunk in author_chunks]

        author_profiles[author] = AuthorProfile(
            author=author,
            delta_centroid={
                feature: fmean(vector[feature] for vector in author_z_vectors)
                for feature in usable_function_words
            },
            lexical_profile={
                feature: fmean(profile[feature] for profile in author_lexical_profiles)
                for feature in author_lexical_profiles[0]
            },
            pos_trigram_profile={
                trigram: fmean(profile[trigram] for profile in author_pos_profiles)
                for trigram in pos_trigrams
            },
            training_chunks=len(author_chunks),
        )

    return StylometryModel(
        function_words=tuple(usable_function_words),
        corpus_means=corpus_means,
        corpus_stdevs=corpus_stdevs,
        pos_trigrams=tuple(pos_trigrams),
        author_profiles=author_profiles,
    )


def score_chunk(model: StylometryModel, chunk: ProcessedChunk) -> ChunkScore:
    """Compare one chunk against every author profile and choose the closest one."""

    raw_function_vector = function_word_vector(chunk, model.function_words)
    z_score_vector = _z_score_vector(raw_function_vector, model.corpus_means, model.corpus_stdevs)
    chunk_lexical_profile = lexical_profile(chunk)
    chunk_pos_profile = pos_trigram_vector(chunk, model.pos_trigrams)

    ranking: list[AttributionResult] = []
    for author, profile in model.author_profiles.items():
        # Lower Delta distance means the function-word pattern is more similar.
        delta_score = fmean(
            abs(z_score_vector[feature] - profile.delta_centroid[feature])
            for feature in model.function_words
        )
        ranking.append(
            AttributionResult(
                author=author,
                delta_score=delta_score,
                pos_similarity=cosine_similarity(chunk_pos_profile, profile.pos_trigram_profile),
                lexical_distance=euclidean_distance(chunk_lexical_profile, profile.lexical_profile),
            )
        )

    ranking.sort(key=lambda result: (result.delta_score, -result.pos_similarity, result.lexical_distance))
    return ChunkScore(chunk=chunk, predicted_author=ranking[0].author, ranking=ranking)


def evaluate_model(
    model: StylometryModel,
    test_chunks: Sequence[ProcessedChunk],
) -> EvaluationSummary:
    """Score all test chunks and calculate the model's accuracy."""

    if not test_chunks:
        raise ValueError("Test data is empty.")

    scored_chunks = [score_chunk(model, chunk) for chunk in test_chunks]
    correct_predictions = sum(1 for scored_chunk in scored_chunks if scored_chunk.predicted_author == scored_chunk.chunk.author)
    return EvaluationSummary(
        total_chunks=len(scored_chunks),
        correct_predictions=correct_predictions,
        accuracy=correct_predictions / len(scored_chunks),
        chunk_scores=scored_chunks,
    )