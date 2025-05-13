# Import necessary utility functions from local module
from util import *

# Import modules for reading data and serialization
import ast
import pickle
from collections import Counter, defaultdict

# Load tokenized Cranfield queries from a text file
with open('output/tokenized_queries.txt', 'r', encoding='utf-8') as f:
    data_str = f.read()

# Safely parse the string representation of the list into actual Python objects
datas = ast.literal_eval(data_str)

# Flatten the nested list structure into a single list of documents
documents = []
for data in datas:
    for ele in data:
        documents.append(ele)

# Initialize word frequency counter
word_counts = Counter()

# Initialize co-occurrence counter as a nested dictionary
cooc_counts = defaultdict(Counter)  # cooc_counts[w][c] = number of times word c occurs near word w

# Set the size of the context window
window_size = 3

# Iterate over each document to count word occurrences and their co-occurring neighbors
for doc in documents:
    for idx, w in enumerate(doc):
        word_counts[w] += 1  # Count the word itself
        # Define the window of context words around the current word
        left = max(0, idx - window_size)
        right = min(len(doc), idx + window_size + 1)
        context_words = doc[left:idx] + doc[idx+1:right]
        for c in context_words:
            cooc_counts[w][c] += 1  # Count co-occurring word

# Compute total number of words in the corpus
total_words = sum(word_counts.values())

# Compute probability P(w) for each word
P_w = {w: count / total_words for w, count in word_counts.items()}

# Compute conditional probability P(c|w) for each word pair (context given word)
P_c_given_w = {}
for w, counter in cooc_counts.items():
    total_cooc = sum(counter.values())
    P_c_given_w[w] = {c: count / total_cooc for c, count in counter.items()}

# Save P(w) dictionary to a pickle file
with open("p_w.pkl", "wb") as f:
    pickle.dump(P_w, f)

# Save P(c|w) dictionary to a pickle file
with open("p_c_given_w.pkl", "wb") as f:
    pickle.dump(P_c_given_w, f)

# Print completion message
print("Probability calculation has been done")
