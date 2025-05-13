from util import *  # Import utility functions (not defined in this snippet)
from collections import defaultdict  # Import defaultdict for efficient dictionary creation
import numpy as np  # Import NumPy for numerical operations
import pickle  # Import pickle for saving/loading models
from sklearn.metrics.pairwise import cosine_similarity  # Import cosine similarity for vector comparisons
from n_gram import load_model, find_candidates  # Import n-gram model functions for loading and finding candidates
from edit_distance_calculation import filter_candidates_by_distance  # Import function to filter candidates by edit distance
from nltk.corpus import stopwords  # Import stopwords from NLTK library


stop_words = set(stopwords.words('english'))  # Load stopwords for filtering
vocab_list = None  # Initialize vocab_list to store vocabulary words
# Read vocabulary words from a file
with open('vocab_words.txt', 'r') as f:
    vocab_list = [line.strip() for line in f]

# Optional: convert the vocabulary list to a set for faster lookup
vocab_set = set(vocab_list)
print(f"Loaded {len(vocab_set)} words from vocab_words.txt")

# In another script or session, load the pre-trained model (vocab matrix and ngrams)
vocab_matrix, vocab, ngrams_list = load_model()
print("Model loaded!")

# Load pre-calculated probabilities
P_w = None
with open("p_w.pkl", "rb") as f:
    P_w = pickle.load(f)

P_c_given_w = None
with open("p_c_given_w.pkl", "rb") as f:
    P_c_given_w = pickle.load(f)

print("Probabilities loaded")

# Load Good-Turing smoothed probabilities
P_w_gt = None
with open("p_w_gt.pkl", "rb") as f:
    P_w_gt = pickle.load(f)

P_c_given_w_gt = None
with open("p_c_given_w_gt.pkl", "rb") as f:
    P_c_given_w_gt = pickle.load(f)

P_unseen_w = None
with open("p_unseen_w.pkl", "rb") as f:
    P_unseen_w = pickle.load(f)

print("Loaded Good-Turing smoothed probabilities.")


def score_candidate(w, context_words, P_w, P_c_given_w, smoothing=1e-8):
    """
    Score a candidate word based on its probability and context words.
    """
    # P(w): probability of the candidate word
    prob_w = P_w.get(w, smoothing)
    score = prob_w  # Start with the probability of the candidate word
    
    # Calculate score based on context words
    for c in context_words:
        prob_c_given_w = P_c_given_w.get(w, {}).get(c, smoothing)  # Get conditional probability of context word given candidate
        score *= prob_c_given_w  # Multiply the score by the probability for each context word
    
    return score


def filtering_context_word(context_words):
    """
    Filter out stopwords from the list of context words.
    """
    # Remove stopwords from the context words
    filtered_context = [word for word in context_words if word.lower() not in stop_words]
    return filtered_context


def spell_check_function(sentence):
    """
    Main function to perform spell checking and suggestion generation.
    """
    tokens = sentence.split()  # Tokenize the input sentence into words
    print(f"The sentence tokens are {tokens}")
    
    modified_sentence = []  # Initialize an empty list to store the modified sentence
    window = 3  # Define a window size for considering context words around an error
    
    for i, token in enumerate(tokens):
        if token in vocab_set:  # If the word is valid (in the vocabulary), add it to the modified sentence
            modified_sentence.append(token)
            print("Valid word!")
        else:
            print(f"Invalid word: '{token}', Searching for candidate words")
            # Find rough candidate words based on n-gram similarity
            candidates = find_candidates(token, vocab_matrix, vocab, ngrams_list, threshold=0.01)
            rough_candidate_words = [candidate_word[0] for candidate_word in candidates]
    
            actual_candidate_words = []  # Initialize list for actual candidates
            if rough_candidate_words:
                # Filter candidates by edit distance
                actual_candidate_words = filter_candidates_by_distance(rough_candidate_words, token, max_distance=2)
    
            print(f"The actual candidate words for '{token}' are {actual_candidate_words}")
    
            if len(actual_candidate_words) == 0:  # No valid candidates, keep the original word
                modified_sentence.append(token)
            elif len(actual_candidate_words) == 1:  # Only one candidate, use it
                modified_sentence.append(actual_candidate_words[0])
            else:
                error_idx = i  # Store the index of the error
                context_words = []  # Initialize list for context words
                
                # Collect context words within a window around the error
                for j in range(error_idx - window, error_idx + window + 1):
                    if j != error_idx and 0 <= j < len(tokens):
                        context_words.append(tokens[j])
                
                # Print the context words for the current error token
                print(f"Context words for '{token}':", context_words)
                
                scores = {}  # Initialize dictionary to store scores for each candidate
                
                # Calculate scores for each candidate word based on context
                for candidate in actual_candidate_words:
                    # Simple smoothing scoring
                    scores[candidate] = score_candidate(candidate, context_words, P_w, P_c_given_w)
    
                # Sort candidates by score in descending order
                ranked_candidates = sorted(scores.items(), key=lambda x: x[1], reverse=True)
                
                # Print the candidate words and their scores
                for word, score in ranked_candidates:
                    print(f"Candidate: '{word}', Score: {score}")
    
                # Add the best candidate (highest score) to the modified sentence
                modified_sentence.append(ranked_candidates[0][0])
    
    # Print the final modified sentence
    print("The modified sentence is:")
    print(modified_sentence)
    return modified_sentence  # Return the modified sentence
