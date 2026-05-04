from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from itertools import zip_longest

from .corpus import ProcessedChunk


FUNCTION_WORD_CANDIDATES = (
    # These are common words that can show writing style because authors use
    # them in different patterns even when the topic changes.
    "a",
    "about",
    "above",
    "after",
    "again",
    "against",
    "all",
    "almost",
    "also",
    "am",
    "among",
    "an",
    "and",
    "any",
    "are",
    "as",
    "at",
    "be",
    "because",
    "been",
    "before",
    "being",
    "below",
    "between",
    "both",
    "but",
    "by",
    "can",
    "could",
    "did",
    "do",
    "does",
    "doing",
    "down",
    "during",
    "each",
    "few",
    "for",
    "from",
    "further",
    "had",
    "has",
    "have",
    "having",
    "he",
    "her",
    "here",
    "hers",
    "herself",
    "him",
    "himself",
    "his",
    "how",
    "i",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "itself",
    "just",
    "me",
    "more",
    "most",
    "my",
    "myself",
    "no",
    "nor",
    "not",
    "now",
    "of",
    "off",
    "on",
    "once",
    "only",
    "or",
    "other",
    "our",
    "ours",
    "ourselves",
    "out",
    "over",
    "own",
    "same",
    "she",
    "should",
    "so",
    "some",
    "such",
    "than",
    "that",
    "the",
    "their",
    "theirs",
    "them",
    "themselves",
    "then",
    "there",
    "these",
    "they",
    "this",
    "those",
    "through",
    "to",
    "too",
    "under",
    "until",
    "up",
    "very",
    "was",
    "we",
    "were",
    "what",
    "when",
    "where",
    "which",
    "while",
    "who",
    "whom",
    "why",
    "with",
    "would",
    "you",
    "your",
    "yours",
    "yourself",
    "yourselves",
)


PosTrigram = tuple[str, str, str]
"""A group of three part-of-speech tags in a row."""


def lexical_profile(chunk: ProcessedChunk) -> dict[str, float]:
    """Calculate basic word and sentence features for one chunk.

    These features describe the writing style in a general way, like how many
    different words are used, how many words only appear once, and how long the
    words and sentences tend to be.
    """

    tokens = chunk.normalized_tokens
    token_count = len(tokens)
    if token_count == 0:
        raise ValueError("Cannot calculate lexical features for an empty chunk.")

    counts = Counter(tokens)
    hapax_count = sum(1 for count in counts.values() if count == 1)
    average_sentence_length = token_count / max(len(chunk.sentence_lengths), 1)

    return {
        "type_token_ratio": len(counts) / token_count,
        "hapax_ratio": hapax_count / token_count,
        "average_word_length": sum(len(token) for token in tokens) / token_count,
        "average_sentence_length": average_sentence_length,
    }


def build_function_word_vocabulary(
    chunks: Sequence[ProcessedChunk],
    max_features: int = 40,
) -> list[str]:
    """Choose the most common function words from the training chunks.

    I use this to make sure every chunk gets measured with the same set of
    function words. The max_features value controls how many of those words are
    kept.
    """

    counts: Counter[str] = Counter()
    candidates = set(FUNCTION_WORD_CANDIDATES)
    for chunk in chunks:
        counts.update(token for token in chunk.normalized_tokens if token in candidates)
    return [token for token, _ in counts.most_common(max_features)]


def function_word_vector(
    chunk: ProcessedChunk,
    vocabulary: Sequence[str],
) -> dict[str, float]:
    """Turn one chunk into function word frequency features.

    The result tells me what fraction of the chunk is made up by each function
    word in the vocabulary. Using fractions is better than raw counts because
    chunks may not all have exactly the same length.
    """

    counts = Counter(chunk.normalized_tokens)
    total_tokens = len(chunk.normalized_tokens)
    return {
        token: counts[token] / total_tokens if total_tokens else 0.0
        for token in vocabulary
    }


def _pos_trigrams(chunk: ProcessedChunk) -> list[PosTrigram]:
    """Make groups of three POS tags from a chunk.

    This is a helper for grammar-style features. For example, a pattern like
    determiner, adjective, noun can show up as one trigram.
    """

    tags = [tag for _, tag in chunk.pos_tags]
    return list(zip(tags, tags[1:], tags[2:]))


def build_pos_trigram_vocabulary(
    chunks: Sequence[ProcessedChunk],
    max_features: int = 30,
) -> list[PosTrigram]:
    """Choose the most common POS tag trigrams from the chunks.

    This creates a shared grammar-pattern vocabulary, similar to the function
    word vocabulary. The model can then compare chunks using the same set of
    common tag patterns.
    """

    counts: Counter[PosTrigram] = Counter()
    for chunk in chunks:
        counts.update(_pos_trigrams(chunk))
    return [trigram for trigram, _ in counts.most_common(max_features)]


def pos_trigram_vector(
    chunk: ProcessedChunk,
    vocabulary: Sequence[PosTrigram],
) -> dict[PosTrigram, float]:
    """Turn one chunk into POS trigram frequency features.

    Each value is the fraction of all POS trigrams in the chunk that match a
    specific trigram from the vocabulary. This helps capture grammar habits, not
    just word choice.
    """

    counts = Counter(_pos_trigrams(chunk))
    total_trigrams = sum(counts.values())
    return {
        trigram: counts[trigram] / total_trigrams if total_trigrams else 0.0
        for trigram in vocabulary
    }

#it was easier to write my own cosine sim than using Scikit-learn version
def cosine_similarity(
    left: Mapping[object, float],
    right: Mapping[object, float],
) -> float:
    """Measure how similar two feature vectors are by their direction.

    A value closer to 1 means the vectors have a similar pattern. A value closer
    to 0 means they are less similar. This is useful when I care more about the
    shape of the feature pattern than the exact size of the numbers.
    """

    numerator = sum(left[key] * right[key] for key in left)
    left_norm = sum(value * value for value in left.values()) ** 0.5
    right_norm = sum(value * value for value in right.values()) ** 0.5
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return numerator / (left_norm * right_norm)


def euclidean_distance(
    left: Mapping[str, float],
    right: Mapping[str, float],
) -> float:
    """Measure the straight-line distance between two feature vectors.

    Smaller numbers mean the chunks are more alike for these features. Bigger
    numbers mean they are farther apart.
    """

    return sum((left[key] - right[key]) ** 2 for key in left) ** 0.5