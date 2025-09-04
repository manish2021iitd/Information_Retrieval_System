# Smart Information Retrieval & Query Assistance System

## Abstract:
This project presents a modular NLP pipeline combining classical and semantic information retrieval techniques with advanced query refinement through spelling correction and autocomplete. It integrates TF-IDF, Latent Semantic Analysis (LSA), and Explicit Semantic Analysis (ESA) for document ranking, and provides real-time user assistance through trie-based phrase suggestions and bigram-based next-word predictions. Evaluation is performed on the Cranfield dataset using standard metrics such as Precision@k, Recall@k, F-score, nDCG, and MAP.

## System Architecture:

Preprocessing

Sentence Segmentation: Naive/Punkt

Tokenization: Regex/Penn Treebank

Inflection Reduction: Lemmatization

Stopword Removal: NLTK-based

## Retrieval Models

TF-IDF: Cosine similarity

LSA: TruncatedSVD on TF-IDF

ESA: Concept matrix projection

HYBRID: Weighted combination of all three

## Query Assistance

Spell Check: Edit distance + n-gram + contextual probabilities

Autocomplete:

Prefix: Phrase Trie

Infix: Inverted index

Ranking: TF-IDF + BM25

Next Word: Bigram model

## Evaluation Metrics

Precision@k, Recall@k, F-score@k

MAP, nDCG

Visualized with matplotlib plots

## Results (TF-IDF vs LSA vs ESA vs HYBRID):

Metric TF-IDF  LSA  ESA  HYBRID

MAP  0.45  0.51  0.48  0.57

nDCG@10  0.60  0.66  0.62  0.71

Precision@10   0.55  0.60  0.58  0.65

## Conclusion:
Combining semantic embeddings (LSA/ESA) with TF-IDF yields significantly better retrieval performance. Augmenting this with real-time correction and autocompletion enhances user interaction and query relevance
