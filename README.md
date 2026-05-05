# Whose Words Are These?

## A Rule-Based Stylometric Authorship Fingerprinting Pipeline

**Arash Mosharraf**  
Symbolic Computational Linguistics 


## Project Overview

Authorship attribution asks whether measurable patterns in language can help identify who likely wrote a text. Although readers often recognize style intuitively, computational stylometry tries to make authorship style more explicit by measuring repeated habits in word choice, grammatical patterning, and lexical variety.

This project builds a small, interpretable Python pipeline that uses NLTK to compare public-domain literary texts and rank likely authors for holdout or "unknown" writing samples. The goal is not to prove authorship with certainty. Instead, the goal is to demonstrate how symbolic linguistic rules and statistical stylometric measures can work together in a classroom-scale authorship fingerprinting system.

The pipeline is rule-based in how it chooses and organizes linguistic evidence, but statistical in how it scores similarity and evaluates predictions. It uses explicit features such as function words, sentence-respecting chunks, part-of-speech trigrams, type-token ratio, hapax ratio, average word length, and average sentence length. Then it compares those features with z-scores, Burrows's Delta, cosine similarity, Euclidean distance, and holdout accuracy.

## Research Background

The project builds on a long tradition of authorship attribution. Mosteller and Wallace's 1963 study, "Inference in an Authorship Problem," is an important historical precedent because it showed that authorship questions could be studied through measurable word-use patterns rather than only through literary intuition. Their work helped establish the idea that small, frequent, often overlooked words can carry meaningful stylistic information.

This project applies that same general principle at a smaller classroom scale by focusing especially on function words. Function words are useful because they are common, relatively topic-independent, and often difficult for writers to control consciously.

The main attribution method is based on John Burrows's Delta. Burrows's 2002 article, "'Delta': A Measure of Stylistic Difference and a Guide to Likely Authorship," introduced a widely used stylometric distance measure that compares standardized word-frequency profiles. In this pipeline, each text chunk is represented by relative frequencies for selected function words. Those frequencies are standardized with z-scores based on the training corpus, and each author is represented by an average profile. An unknown chunk is then compared with each candidate author by averaging the absolute differences between the chunk's z-scores and the author's profile.

The project also takes seriously David L. Hoover's 2004 critique, "Testing Burrows's Delta." Hoover shows that Delta is powerful but not automatic. Results can depend heavily on feature choices, corpus design, text length, and evaluation method. Because of that, this project does not treat Delta as a magic answer. Delta is the main ranking score, while POS-trigram similarity and lexical-richness distance are reported as supporting diagnostics.

## Methodology

The symbolic side of the project defines the linguistic objects to be measured:

- function-word lists
- preprocessing rules
- sentence-based chunking rules
- part-of-speech trigram extraction
- lexical-richness features

The statistical side compares those symbolic features across authors using:

- frequency distributions
- z-scores
- Burrows's Delta
- cosine similarity
- Euclidean distance
- holdout accuracy evaluation

The final prediction is based on a ranked list of candidate authors. The best author is the one with the lowest Delta score. If there is a tie, the system prefers a higher POS-trigram cosine similarity and then a lower lexical Euclidean distance.

## Corpus Design

The planned corpus uses public-domain literary texts from Project Gutenberg. Text files should be organized by author with a structure like this:

```text
data/
  corpus/
    Jane_Austen/
      pride_and_prejudice.txt
      sense_and_sensibility.txt
    Charles_Dickens/
      great_expectations.txt
      oliver_twist.txt
```

Each author should ideally have more than one work, or at least enough text to create multiple comparable chunks. The pipeline splits longer documents into sentence-respecting chunks of similar size, so that the system compares samples that are roughly equivalent in length. For testing, one or more chunks per author are held out as "unknown" samples. Because the true author of each held-out sample is still known to the researcher, the system can be evaluated with simple accuracy measures.

## Project Structure

```text
authorship_pipeline/
  __init__.py
  attribution.py
  corpus.py
  features.py
authorship_stylometry_project.ipynb
requirements.txt
README.md
```

The notebook is the main demonstration space. The Python package contains the reusable pipeline code.

## Pipeline Workflow

The main workflow is:

1. Load texts as `SourceDocument` objects.
2. Preprocess and split texts into `ProcessedChunk` objects.
3. Split chunks into training and holdout test sets.
4. Train a `StylometryModel` from the training chunks.
5. Score each holdout chunk against every author profile.
6. Rank likely authors and calculate accuracy.

The package exports the main functions from `authorship_pipeline/__init__.py`, so they can be imported directly:

```python
from authorship_pipeline import (
    ensure_nltk_resources,
    load_local_corpus,
    prepare_chunks,
    split_chunks_by_author,
    train_model,
    evaluate_model,
)
```

A typical usage pattern looks like this:

```python
from authorship_pipeline import (
    ensure_nltk_resources,
    load_local_corpus,
    prepare_chunks,
    split_chunks_by_author,
    train_model,
    evaluate_model,
)

ensure_nltk_resources()

documents = load_local_corpus("data/corpus")
chunks = prepare_chunks(documents, chunk_size=1500, min_chunk_size=900)
training_chunks, test_chunks = split_chunks_by_author(chunks, holdout_per_author=1)

model = train_model(
    training_chunks,
    function_word_features=40,
    pos_trigram_features=30,
)

summary = evaluate_model(model, test_chunks)

print(summary.accuracy)
for chunk_score in summary.chunk_scores:
    print(chunk_score.chunk.author, "->", chunk_score.predicted_author)
```

To install the basic dependencies:

```bash
pip install -r requirements.txt
```

## How the Pipeline Works

### 1. Load the Corpus

Texts are loaded as `SourceDocument` objects. Each object stores:

- author
- title
- raw text

This keeps the original document information attached before the text is split into chunks.

### 2. Tokenize

For a sentence like:

```text
The little cat sat on the mat, and it slept.
```

NLTK tokenization produces a sequence like:

```python
["The", "little", "cat", "sat", "on", "the", "mat", ",", "and", "it", "slept", "."]
```

### 3. Normalize

Punctuation is removed, non-letter characters are dropped, and words are lowercased:

```python
["the", "little", "cat", "sat", "on", "the", "mat", "and", "it", "slept"]
```

This gives 10 normalized word tokens.

### 4. POS-Tag

The original alphabetic tokens are grammatically tagged. For this example, the tags might look like this:

```text
The/DT little/JJ cat/NN sat/VBD on/IN the/DT mat/NN and/CC it/PRP slept/VBD
```

Some common tags used in this example are:

```text
DT  = determiner
NN  = noun, singular
JJ  = adjective
VBD = verb, past tense
PRP = personal pronoun
```

### 5. Extract Function-Word Features

Function words are small grammatical words such as:

```text
the, and, of, to, in, was, it, that, with, he
```

Burrows's Delta focuses mostly on function words because they are common, hard to consciously control, and less topic-dependent than nouns like cat, war, or money.

For the example sentence:

```text
the little cat sat on the mat and it slept
```

The function-word frequencies are:

```text
the = 2/10 = 0.20
on  = 1/10 = 0.10
and = 1/10 = 0.10
it  = 1/10 = 0.10
```

The program uses relative frequencies instead of raw counts because chunks may not all have exactly the same length.

### 6. Extract Lexical Features

The lexical profile describes broader vocabulary behavior.

For the example sentence:

```text
total tokens = 10
unique words = 9
type-token ratio = 9/10 = 0.90
```

A high type-token ratio means the writer uses many different words in that sample.

Other lexical features include:

```text
average sentence length = 10
average word length = average length of all words
hapax ratio = words used once / total words
```

A hapax is a word that appears only once in a text sample. The full term is hapax legomenon. Hapax ratio is useful because it gives another clue about vocabulary behavior.

### 7. Extract POS Trigrams

POS trigrams capture grammar patterns by looking at three part-of-speech tags in a row.

From this tag sequence:

```text
DT JJ NN VBD IN DT NN CC PRP VBD
```

The program gets three-tag sequences like:

```text
DT-JJ-NN
JJ-NN-VBD
NN-VBD-IN
VBD-IN-DT
```

These patterns help the model compare grammar habits, not just word choice.

## Burrows's Delta Example

Raw function-word frequencies are not enough because some words are naturally more common than others. The project converts each function-word frequency into a z-score:

```text
z-score = (chunk frequency - corpus mean) / corpus standard deviation
```

The meaning is:

```text
0  = normal compared to the corpus
+1 = one standard deviation higher than average
-1 = one standard deviation lower than average
```

Example:

```text
corpus average for "the" = 0.06
corpus standard deviation for "the" = 0.02
chunk frequency for "the" = 0.10

z = (0.10 - 0.06) / 0.02
z = 2.0
```

So this chunk uses "the" much more often than average.

Then each author gets an average profile, called a centroid.

```text
Author A average z-scores:
the:  1.5
and:  0.2
of:  -0.4

Author B average z-scores:
the: -0.5
and:  1.2
of:   0.8
```

Now suppose an unknown chunk has these z-scores:

```text
Unknown chunk:
the:  1.7
and:  0.1
of:  -0.2
```

Burrows's Delta compares the unknown chunk to each author by averaging the absolute differences.

For Author A:

```text
|1.7 - 1.5|  = 0.2
|0.1 - 0.2|  = 0.1
|-0.2 - -0.4| = 0.2

Delta = (0.2 + 0.1 + 0.2) / 3 = 0.17
```

For Author B:

```text
|1.7 - -0.5| = 2.2
|0.1 - 1.2|  = 1.1
|-0.2 - 0.8| = 1.0

Delta = (2.2 + 1.1 + 1.0) / 3 = 1.43
```

The smaller Delta wins, so the program predicts Author A.

## Supporting Scores

### POS-Trigram Cosine Similarity

POS-trigram cosine similarity compares grammatical pattern vectors. Higher is better. If the unknown chunk has grammar patterns similar to an author profile, the cosine similarity is closer to 1.0.

For example, the unknown chunk and Author A might both often use:

```text
DT-JJ-NN
PRP-VBD-IN
NN-IN-DT
```

That would support the idea that the unknown chunk is stylistically closer to Author A.

### Lexical Euclidean Distance

Lexical Euclidean distance compares broader vocabulary statistics, such as type-token ratio, hapax ratio, average word length, and average sentence length. Lower is better.

Example:

```text
Unknown:
type-token ratio = 0.42
average sentence length = 18

Author A:
type-token ratio = 0.40
average sentence length = 19

Author B:
type-token ratio = 0.62
average sentence length = 9
```

The unknown sample is closer to Author A because its lexical measurements are more similar.

## Interpretable Design

The symbolic components are important because they make the pipeline understandable. Instead of using a black-box deep learning model, this project uses explicit linguistic rules:

- which tokens count as function words
- how punctuation and capitalization are normalized
- how long chunks should be
- which POS trigrams are retained
- which lexical-richness measurements are included

These design choices are not neutral, so they should be documented as part of the project. A reader should be able to inspect the pipeline and understand why a particular feature was counted and how it contributed to the final comparison.

The statistical components are important because authorship attribution requires comparison across many small measurements. Raw counts alone are not enough, since common words naturally appear more often than rare words. Z-scores make function-word frequencies comparable across features, Burrows's Delta summarizes stylistic distance, cosine similarity compares POS-trigram profiles, and Euclidean distance compares compact lexical profiles.

When this project mentions black-box models, it is mainly referring to newer deep learning models used in NLP. Not every machine-learning model is a black box, but this project intentionally stays with visible linguistic features and inspectable scoring rules.

## Limitations

Several limitations and methodological problems need to be considered:

- Genre and topic can bias results, especially if one author is represented by novels and another by essays, letters, or children's literature.
- Short texts may not contain enough repeated function words or grammatical patterns for stable attribution.
- A small corpus can make author profiles questionable, especially when a single long work is split into many chunks and treated as if it represents the author's whole style.
- Project Gutenberg texts may contain inconsistent headers, footers, editorial notes, or formatting artifacts that should be removed or controlled.
- POS-tagging errors may affect trigram features, especially in older literary language.
- Feature selection matters. Changing the number of function words, chunk size, or candidate authors may change the outcome.

For these reasons, the project presents stylometric attribution as evidence for likely authorship, not as proof.

## References

Burrows, J. (2002). 'Delta': A measure of stylistic difference and a guide to likely authorship. *Literary and Linguistic Computing, 17*(3), 267-287.

Hoover, D. L. (2004). Testing Burrows's Delta. *Literary and Linguistic Computing, 19*(4), 453-475.

Mosteller, F., & Wallace, D. L. (1963). Inference in an authorship problem. *Journal of the American Statistical Association, 58*(302), 275-309.