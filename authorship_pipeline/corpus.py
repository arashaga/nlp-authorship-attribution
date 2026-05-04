from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
import re

import nltk
from nltk import pos_tag, sent_tokenize, word_tokenize

@dataclass(frozen=True)
class SourceDocument:
    """A text from the corpus before it has been split up or processed.

    I keep the author, title, and full text together so the rest of the
    pipeline can pass one object around instead of three separate values.
    """

    author: str
    title: str
    text: str


@dataclass(frozen=True)
class ProcessedChunk:
    """One processed section of a source document.

    This is the version of the text that is ready for feature extraction or
    model training. I freeze it because once a chunk is made, I do not want its
    author, tokens, or tags to accidentally change later in the project.
    """

    author: str
    title: str
    chunk_index: int
    raw_text: str
    normalized_tokens: tuple[str, ...]
    sentence_lengths: tuple[int, ...]
    pos_tags: tuple[tuple[str, str], ...]


def _resource_exists(resource_path: str) -> bool:
    """Check if an NLTK resource is already installed.

    This helper lets the setup function avoid downloading the same tokenizer ( imnportant) or
    tagger every time the program runs.
    """

    try:
        nltk.data.find(resource_path)
        return True
    except LookupError:
        return False

#punkt helps the program break the text into sentences and words in a smarter
# way before counting tokens, sentence lengths, and POS tags
def ensure_nltk_resources(include_gutenberg: bool = False) -> None:
    """Download the NLTK resources this pipeline needs if they are missing.

    The tokenizer resources are needed for splitting text into sentences and
    words. The tagger resources are needed for part-of-speech tags. Gutenberg is
    optional because I only need it when I am using NLTK's sample texts.
    """

    required_resources = [
        ("tokenizers/punkt", "punkt"),
        ("tokenizers/punkt_tab", "punkt_tab"),# I added thi sisnce newer NLTK might throw errors
    ]
    if include_gutenberg:
        required_resources.append(("corpora/gutenberg", "gutenberg"))
    #part of speech taggers
    tagger_resources = [
        ("taggers/averaged_perceptron_tagger_eng", "averaged_perceptron_tagger_eng"),
        ("taggers/averaged_perceptron_tagger", "averaged_perceptron_tagger"),
    ]

    for resource_path, package_name in required_resources:
        if not _resource_exists(resource_path):
            nltk.download(package_name, quiet=True)

    if not any(_resource_exists(resource_path) for resource_path, _ in tagger_resources):
        for _, package_name in tagger_resources:
            nltk.download(package_name, quiet=True)


def load_local_corpus(corpus_dir: str | Path) -> list[SourceDocument]:
    """Load text files from a local folder into SourceDocument objects.

    The expected folder setup is one folder per author, with .txt files inside
    later on TODO to add PDF parser
    each author folder. The folder name becomes the author name, and the file
    name becomes the title.
    """

    corpus_path = Path(corpus_dir)
    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus directory was not found: {corpus_path}")

    documents: list[SourceDocument] = []
    for author_dir in sorted(path for path in corpus_path.iterdir() if path.is_dir()):
        for text_path in sorted(author_dir.glob("*.txt")):
            documents.append(
                SourceDocument(
                    author=author_dir.name,
                    title=text_path.stem.replace("_", " "),
                    text=text_path.read_text(encoding="utf-8"),
                )
            )

    if not documents:
        raise ValueError(
            "No text files were found. Use a structure like data/corpus/<author>/<title>.txt."
        )

    return documents

# on important note this will remove contractions like "don't" -> "dont" and numbers
# I want to normalize it to remove noise but if you think contractions etc are important in 
#Stylometry then keep them
def normalize_tokens(tokens: Sequence[str]) -> list[str]:
    """Clean a list of tokens so only lowercase word tokens are kept.

    This removes punctuation, numbers, and empty strings. For example, "Hello!"
    becomes "hello", while "123" gets dropped.
    """

    normalized: list[str] = []
    for token in tokens:
        cleaned = re.sub(r"[^A-Za-z]", "", token).lower()
        if cleaned:
            normalized.append(cleaned)
    return normalized


def _build_chunk(
    document: SourceDocument,
    chunk_index: int,
    sentence_texts: Sequence[str],
    sentence_token_lists: Sequence[Sequence[str]],
    min_chunk_size: int,
    allow_small_chunk: bool,
) -> ProcessedChunk | None:
    """Turn a group of sentences into one ProcessedChunk.

    This is a private helper because chunk_document does the actual deciding
    about where chunks starts and end. Here I collect the normalized tokens,
    sentence lengths, and part-of-speech tags for that chunk. If the chunk is
    empty or too small, it returns None instead.
    """

    normalized_tokens: list[str] = []
    sentence_lengths: list[int] = []
    alpha_surface_tokens: list[str] = []

    for sentence_tokens in sentence_token_lists:
        normalized_sentence = normalize_tokens(sentence_tokens)
        if not normalized_sentence:
            continue
        normalized_tokens.extend(normalized_sentence) #extend it since it sould get a list
        sentence_lengths.append(len(normalized_sentence))
        alpha_surface_tokens.extend(token for token in sentence_tokens if re.search(r"[A-Za-z]", token))

    if not normalized_tokens:
        return None
    if len(normalized_tokens) < min_chunk_size and not allow_small_chunk:
        return None

    tagged_tokens = tuple(pos_tag(alpha_surface_tokens))
    return ProcessedChunk(
        author=document.author,
        title=document.title,
        chunk_index=chunk_index,
        raw_text="\n".join(sentence_texts),
        normalized_tokens=tuple(normalized_tokens),
        sentence_lengths=tuple(sentence_lengths),
        pos_tags=tagged_tokens,
    )


def chunk_document(
    document: SourceDocument,
    chunk_size: int = 1500,
    min_chunk_size: int = 900,
) -> list[ProcessedChunk]:
    """Split one SourceDocument into processed chunks.

    The function walks through the document sentence by sentence and keeps
    adding sentences until the chunk is close to the target size. It keeps whole
    sentences together instead of cutting a sentence in the middle.
    """

    sentence_texts = sent_tokenize(document.text)
    chunks: list[ProcessedChunk] = []

    current_sentence_texts: list[str] = []
    current_sentence_tokens: list[list[str]] = []
    current_token_total = 0

    for sentence_text in sentence_texts:
        sentence_tokens = word_tokenize(sentence_text)
        normalized_sentence = normalize_tokens(sentence_tokens)
        if not normalized_sentence:
            continue

        if current_sentence_texts and current_token_total + len(normalized_sentence) > chunk_size:
            chunk = _build_chunk(
                document=document,
                chunk_index=len(chunks),
                sentence_texts=current_sentence_texts,
                sentence_token_lists=current_sentence_tokens,
                min_chunk_size=min_chunk_size,
                allow_small_chunk=not chunks,
            )
            if chunk is not None:
                chunks.append(chunk)
            current_sentence_texts = []
            current_sentence_tokens = []
            current_token_total = 0

        current_sentence_texts.append(sentence_text)
        current_sentence_tokens.append(sentence_tokens)
        current_token_total += len(normalized_sentence)

    final_chunk = _build_chunk(
        document=document,
        chunk_index=len(chunks),
        sentence_texts=current_sentence_texts,
        sentence_token_lists=current_sentence_tokens,
        min_chunk_size=min_chunk_size,
        allow_small_chunk=not chunks,
    )
    if final_chunk is not None:
        chunks.append(final_chunk)

    return chunks


def prepare_chunks(
    documents: Sequence[SourceDocument],
    chunk_size: int = 1500, # I use this in GenAI RAG usually a commmon chunk size
    min_chunk_size: int = 900, # do not keep a chunk unless this big
) -> list[ProcessedChunk]:
    """Process every document and combine all of the chunks into one list.

    This is basically the batch version of chunk_document. It is useful when I
    already loaded the whole corpus and want one list of chunks for training or
    testing.
    """

    chunks: list[ProcessedChunk] = []
    for document in documents:
        chunks.extend(chunk_document(document, chunk_size=chunk_size, min_chunk_size=min_chunk_size))
    return chunks


def split_chunks_by_author(
    chunks: Sequence[ProcessedChunk],
    holdout_per_author: int = 1,
) -> tuple[list[ProcessedChunk], list[ProcessedChunk]]:
    """Split chunks into training and test groups for each author.

    For every author, this keeps the last few chunks as test examples and uses
    the rest for training. That way each author is represented in both groups.
    """

    by_author: dict[str, list[ProcessedChunk]] = defaultdict(list)
    for chunk in chunks:
        by_author[chunk.author].append(chunk)

    training_chunks: list[ProcessedChunk] = []
    test_chunks: list[ProcessedChunk] = []

    for author, author_chunks in sorted(by_author.items()):
        ordered_chunks = sorted(author_chunks, key=lambda chunk: (chunk.title, chunk.chunk_index))
        if len(ordered_chunks) <= holdout_per_author:
            raise ValueError(
                f"Author '{author}' only has {len(ordered_chunks)} chunk(s). Increase the corpus size or lower holdout_per_author."
            )
        training_chunks.extend(ordered_chunks[:-holdout_per_author])
        test_chunks.extend(ordered_chunks[-holdout_per_author:])

    return training_chunks, test_chunks