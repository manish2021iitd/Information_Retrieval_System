# Importing necessary libraries
import json  # For loading JSON data
import numpy as np  # For numerical operations
from collections import defaultdict, Counter  # For dictionary with default values and frequency counting
from sklearn.feature_extraction.text import TfidfVectorizer  # For TF-IDF vectorization
from rank_bm25 import BM25Okapi  # For BM25 ranking algorithm
import nltk  # For natural language processing tools
from nltk.util import ngrams  # For generating n-grams (e.g., bigrams)
from nltk.tokenize import word_tokenize  # For tokenizing sentences into words
from spell_check import spell_check_function  # Importing custom spell check function

# Download tokenizer data if not already present
nltk.download('punkt')

# Node class for the phrase trie
class PhraseTrieNode:
    def __init__(self):
        self.children = defaultdict(PhraseTrieNode)  # Child nodes for each word
        self.is_end = False  # Marks end of a phrase
        self.freq = 0  # Frequency of the complete phrase

# Main Autocomplete System class
class AutocompleteSystem:
    def __init__(self, query_data):
        self.query_data = query_data  # Store the loaded query data
        self.query_texts = [q["query"].lower().strip() for q in query_data]  # Clean and store query strings
        self.root = PhraseTrieNode()  # Initialize root of the phrase trie
        self._build_phrase_trie()  # Build phrase trie for prefix matching
        self._build_inverted_index()  # Build inverted index for infix matching
        self._build_tfidf_bm25()  # Build TF-IDF and BM25 models
        self._build_ngram_model()  # Build bigram model for next word prediction

    # Build a phrase trie using the query texts
    def _build_phrase_trie(self):
        for query in self.query_texts:
            tokens = query.split()  # Tokenize the query
            node = self.root
            for token in tokens:
                node = node.children[token]  # Traverse/create nodes for each word
            node.is_end = True  # Mark end of phrase
            node.freq += 1  # Increment frequency

    # Depth-first search to collect completions from trie
    def _dfs_trie(self, node, path, results):
        if node.is_end:
            results.append((" ".join(path), node.freq))  # Save complete phrase with frequency
        for word, child in node.children.items():
            self._dfs_trie(child, path + [word], results)  # Recursively search deeper

    # Get completions based on prefix from phrase trie
    def _get_phrase_completions(self, prefix):
        tokens = prefix.lower().strip().split()  # Tokenize and clean prefix
        node = self.root
        for token in tokens:
            if token not in node.children:
                return []  # If prefix not found, return empty
            node = node.children[token]
        results = []
        self._dfs_trie(node, tokens, results)  # Collect completions using DFS
        return results

    # Build inverted index for infix (any-position) search
    def _build_inverted_index(self):
        self.inverted_index = defaultdict(set)
        for i, query in enumerate(self.query_texts):
            for word in query.split():
                self.inverted_index[word].add(i)  # Map word to its query indices

    # Get queries containing the word (infix match)
    def _get_infix_completions(self, word):
        indices = self.inverted_index.get(word, set())
        return [(self.query_texts[i], 1) for i in indices]

    # Build TF-IDF and BM25 vector representations for queries
    def _build_tfidf_bm25(self):
        self.tfidf_vectorizer = TfidfVectorizer()
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.query_texts)

        tokenized_corpus = [word_tokenize(q) for q in self.query_texts]  # Tokenize all queries
        self.bm25 = BM25Okapi(tokenized_corpus)  # Create BM25 model

    # Score suggestions using both TF-IDF and BM25
    def _score_tfidf_bm25(self, prefix, suggestions):
        tfidf_vec = self.tfidf_vectorizer.transform([prefix])  # Vectorize prefix
        bm25_scores = self.bm25.get_scores(word_tokenize(prefix))  # BM25 scores for prefix

        max_bm25 = np.max(bm25_scores) if bm25_scores.size > 0 else 1.0  # Avoid divide by zero
        normalized_bm25_scores = bm25_scores / max_bm25  # Normalize scores

        scores = []
        for s, _ in suggestions:
            try:
                idx = self.query_texts.index(s)  # Find index of suggestion
                tfidf_score = np.dot(self.tfidf_matrix[idx].toarray(), tfidf_vec.toarray().T)[0][0]
                bm25_score = normalized_bm25_scores[idx]
                combined = 0.5 * tfidf_score + 0.5 * bm25_score  # Combine both scores equally
                scores.append((s, combined))
            except ValueError:
                scores.append((s, 0.0))  # In case suggestion not found
        return sorted(scores, key=lambda x: -x[1])[:5]  # Return top 5 suggestions

    # Build bigram model from query texts
    def _build_ngram_model(self):
        all_tokens = []
        for query in self.query_texts:
            tokens = word_tokenize(query)
            all_tokens.extend(tokens)  # Combine tokens from all queries
        self.bigram_counts = Counter(ngrams(all_tokens, 2))  # Count bigram frequencies

    # Predict next possible words using bigram model
    def _predict_next_words(self, prefix):
        tokens = prefix.lower().strip().split()
        if not tokens:
            return []
        last = tokens[-1]  # Last word of prefix
        candidates = [(b[1], count) for b, count in self.bigram_counts.items() if b[0] == last]
        return sorted(candidates, key=lambda x: -x[1])[:3]  # Return top 3 predictions

    # Main function to generate autocomplete suggestions
    def autocomplete(self, prefix):
        completions = self._get_phrase_completions(prefix)  # From phrase trie
        last_word = prefix.strip().split()[-1] if prefix.strip() else ""
        infix_matches = self._get_infix_completions(last_word) if last_word else []

        all_candidates_dict = {c[0]: c for c in completions + infix_matches}  # Merge and deduplicate
        all_candidates = list(all_candidates_dict.values())

        ranked = self._score_tfidf_bm25(prefix, all_candidates)  # Score suggestions
        predicted_next = self._predict_next_words(prefix)  # Predict next words

        print("\n Next Word Predictions:")
        for word, freq in predicted_next:
            print(f" - {word} (freq: {freq})")

        return ranked

# Load JSON file containing query data
def load_queries_from_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

# Use the spell checker to correct the input sentence
def correct_sentence_function(sentence):
    modified_sentence = spell_check_function(sentence)  # Apply spell check
    correct_sentence = " ".join(word for word in modified_sentence)  # Reconstruct sentence
    return correct_sentence

# Complete pipeline: correct query → get and print suggestions
def autoComputation_function(query):
    queries = load_queries_from_json("cranfield/cran_queries.json")
    system = AutocompleteSystem(queries)
    corrected_user_input = correct_sentence_function(query)
    print(f"Correct sentence : {corrected_user_input}")
    suggestions = system.autocomplete(corrected_user_input)
    print("\n Suggestions:")
    sorted_suggestion = sorted(suggestions, key = lambda x:x[1], reverse = True)
    for suggestion, score in sorted_suggestion:
        print(f" - {suggestion} (score: {score:.4f})")
    return sorted_suggestion[0][0]  # Return top suggestion

# Command-line interface for autocomplete
if __name__ == "__main__":
    queries = load_queries_from_json("cranfield/cran_queries.json")
    system = AutocompleteSystem(queries)

    print(" Query Autocompletion (type 'exit' to quit):")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == "exit":
            break
        corrected_user_input = correct_sentence_function(user_input)
        print(f"Correct sentence : {corrected_user_input}")
        suggestions = system.autocomplete(corrected_user_input)
        print("\n Suggestions:")
        for suggestion, score in suggestions:
            print(f" - {suggestion} (score: {score:.4f})")
