from util import *  # Import utility functions (not defined in this snippet)
from collections import defaultdict  # Import defaultdict for efficient default dictionary creation
import numpy as np  # Import NumPy for numerical operations
import pickle  # Import pickle for saving/loading models
from sklearn.metrics.pairwise import cosine_similarity  # Import cosine_similarity for calculating vector similarity


def build_ngram_vocabulary(words, n_list=[1, 2]):
    """
    Extract all unique ngrams from the vocabulary for all n values (unigrams, bigrams, etc.)
    """
    ngram_set = set()  # Initialize a set to store unique ngrams
    for word in words:
        for n in n_list:
            # Generate ngrams for the word and add them to the ngram set
            ngrams = [word[i:i+n] for i in range(len(word)-n+1)]  
            ngram_set.update(ngrams)  # Update the ngram set with the new ngrams
    return sorted(list(ngram_set))  # Return the sorted list of unique ngrams


def vectorize_word(word, ngrams_list, n_list=[1, 2]):
    """
    Convert a word into a binary vector based on the ngrams in the vocabulary.
    """
    word_ngrams = set()  # Initialize a set to store the ngrams of the word
    for n in n_list:
        word_ngrams.update([word[i:i+n] for i in range(len(word)-n+1)])  # Generate and store word ngrams
    # Create a binary vector where each element indicates the presence of a corresponding ngram in the word
    return np.array([1 if ng in word_ngrams else 0 for ng in ngrams_list])  


def build_vocab_matrix(vocab, ngrams_list, n_list=[1, 2]):
    """
    Build a matrix where each row is a word vector, with combined ngrams.
    """
    # For each word in the vocabulary, generate a vector and stack them into a matrix
    return np.vstack([vectorize_word(word, ngrams_list, n_list) for word in vocab])


def find_candidates(wrong_word, vocab_matrix, vocab, ngrams_list, n_list=[1, 2], threshold=0.5):
    """
    Find candidates with cosine similarity above a specified threshold.
    """
    wrong_vec = vectorize_word(wrong_word, ngrams_list, n_list)  # Vectorize the wrong word
    sims = cosine_similarity([wrong_vec], vocab_matrix)[0]  # Compute cosine similarity between the wrong word and vocab matrix
    # Filter and sort candidates based on similarity score above the threshold
    candidates = [(word, sim) for word, sim in zip(vocab, sims) if sim >= threshold]
    return sorted(candidates, key=lambda x: -x[1])  # Sort candidates in descending order of similarity


def save_model(vocab_matrix, vocab_list, ngrams_list, filename_prefix='spellcheck_ngram'):
    """
    Save the vocabulary matrix, vocab list, and ngram list to files.
    """
    np.save(f"{filename_prefix}_matrix.npy", vocab_matrix)  # Save the vocab matrix as a NumPy file
    # Save metadata (vocab and ngrams) using pickle
    with open(f"{filename_prefix}_meta.pkl", "wb") as f:
        pickle.dump({'vocab': vocab_list, 'ngrams': ngrams_list}, f)


def load_model(filename_prefix='spellcheck_ngram'):
    """
    Load the model (vocab matrix and metadata) from saved files.
    """
    vocab_matrix = np.load(f"{filename_prefix}_matrix.npy")  # Load the vocab matrix
    # Load metadata (vocab and ngrams) using pickle
    with open(f"{filename_prefix}_meta.pkl", "rb") as f:
        meta = pickle.load(f)
    vocab_list = meta['vocab']  # Extract vocab list from metadata
    ngrams_list = meta['ngrams']  # Extract ngram list from metadata
    return vocab_matrix, vocab_list, ngrams_list  # Return the loaded model


if __name__ == "__main__":
    vocab = None  # Initialize vocab as None
    n_list = [1, 2]  # Define the ngram sizes (unigrams and bigrams)

    with open('vocab_words.txt', 'r') as f:
        vocab = [line.strip() for line in f]  # Read the vocabulary words from a file

    ngrams_list = build_ngram_vocabulary(vocab, n_list)  # Build the list of unique ngrams from the vocab
    print("All ngrams:", ngrams_list)  # Print the list of ngrams

    vocab_matrix = build_vocab_matrix(vocab, ngrams_list, n_list)  # Build the vocabulary matrix
    save_model(vocab_matrix, vocab, ngrams_list)  # Save the model to disk
    print("Model saved!")  # Print confirmation message
