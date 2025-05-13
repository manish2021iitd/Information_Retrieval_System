from util import *  # Import all helper functions from util.py
import ast  # For safely parsing string data as Python objects
import pickle  # For saving Python objects to files
from collections import Counter, defaultdict  # For counting and nested dictionaries

# Load Cranfield dataset (tokenized queries)
with open('output/tokenized_queries.txt', 'r', encoding='utf-8') as f:
    data_str = f.read()  # Read the entire file as a string

datas = ast.literal_eval(data_str)  # Safely convert string to Python list

# Flatten nested tokenized query data into a list of documents (each is a list of words)
documents = []
for data in datas:
    for ele in data:
        documents.append(ele)

# Initialize counters
word_counts = Counter()  # Count how many times each word appears
cooc_counts = defaultdict(Counter)  # Count how many times each word co-occurs with others
window_size = 3  # Define co-occurrence window size

# Build word and co-occurrence counts
for doc in documents:
    for idx, w in enumerate(doc):  # For each word in a document
        word_counts[w] += 1  # Increment the word count
        # Define window range around the current word
        left = max(0, idx - window_size)
        right = min(len(doc), idx + window_size + 1)
        # Get context words within the window (excluding the word itself)
        context_words = doc[left:idx] + doc[idx+1:right]
        for c in context_words:
            cooc_counts[w][c] += 1  # Increment co-occurrence count

# Compute frequency of frequencies for word counts
freq_of_freq_w = Counter(word_counts.values())  # Count how many words occurred exactly c times

# Compute frequency of frequencies for co-occurrence counts
freq_of_freq_c = defaultdict(Counter)
for w, counter in cooc_counts.items():
    freq_of_freq_c[w] = Counter(counter.values())  # For each word, count how many co-words occurred c times

# Apply Good-Turing smoothing for unigram probabilities P(w)
total_words = sum(word_counts.values())  # Total number of word tokens
P_w_gt = {}  # Smoothed probabilities
for w, c in word_counts.items():
    Nc = freq_of_freq_w[c]  # How many words occurred c times
    Nc1 = freq_of_freq_w.get(c + 1, 0)  # How many words occurred (c+1) times
    if Nc1 > 0:
        c_star = (c + 1) * Nc1 / Nc  # Good-Turing adjusted count
    else:
        c_star = c  # Fallback to raw count if Nc1 is zero
    P_w_gt[w] = c_star / total_words  # Convert to probability

# Compute total probability mass for unseen words
N1_w = freq_of_freq_w.get(1, 0)  # Number of words seen only once
P_unseen_w = N1_w / total_words  # Assign probability mass to unseen words

# Apply Good-Turing smoothing for conditional probabilities P(c|w)
P_c_given_w_gt = {}
for w, counter in cooc_counts.items():
    Nw = sum(counter.values())  # Total number of co-occurrence events for word w
    freq_of_freq = freq_of_freq_c[w]  # Frequency of frequencies for co-words of w
    P_c_given_w_gt[w] = {}
    N1_c = freq_of_freq.get(1, 0)  # How many co-words occurred once with w
    for c, c_count in counter.items():
        Nc = freq_of_freq[c_count]  # How many co-words occurred c_count times
        Nc1 = freq_of_freq.get(c_count + 1, 0)  # Co-words occurred (c_count+1) times
        if Nc1 > 0:
            c_star = (c_count + 1) * Nc1 / Nc  # Good-Turing adjusted count
        else:
            c_star = c_count  # Fallback to raw count
        P_c_given_w_gt[w][c] = c_star / Nw  # Convert to conditional probability

# Save Good-Turing smoothed unigram probabilities
with open("p_w_gt.pkl", "wb") as f:
    pickle.dump(P_w_gt, f)

# Save Good-Turing smoothed conditional probabilities
with open("p_c_given_w_gt.pkl", "wb") as f:
    pickle.dump(P_c_given_w_gt, f)

# Save probability mass for unseen words
with open("p_unseen_w.pkl", "wb") as f:
    pickle.dump(P_unseen_w, f)

# Inform the user that the process is complete
print("Good-Turing probability calculation and saving completed.")
