# Import necessary libraries
import wikipedia
import time
import ast
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from util import *

# For checking if a word is in English vocabulary
import nltk
nltk.download('words')
from nltk.corpus import words

# Create a lowercase vocabulary set of English words
vocab = set(word.lower() for word in words.words())

# -------- Function to get Wikipedia summary for a term --------
def get_summary(term):
    try:
        return wikipedia.summary(term)
    except wikipedia.DisambiguationError as e:
        # If term is ambiguous, try the first option
        try:
            return wikipedia.summary(e.options[0])
        except:
            return None
    except wikipedia.PageError:
        # If no page found, search similar titles
        search_results = wikipedia.search(term)
        if search_results:
            try:
                return wikipedia.summary(search_results[0])
            except:
                return None
        return None

# -------- Function to fetch Wikipedia summaries quickly --------
def fetch_wikipedia_articles_fast(terms, delay=0.5):
    articles = {}
    count = 1
    for term in terms:
        try:
            summary = get_summary(term)
            if summary:
                articles[term] = summary
                print(f"Fetched summary for {count}-th term: {term}")
            else:
                print(f"No article found for term: {term}")
        except Exception as e:
            print(f"Skipped term '{term}': {e}")
        time.sleep(delay)  # To avoid API rate limits
        count += 1
    return articles

# -------- Step 1: Extract unique English words from Cranfield --------
def extract_unique_terms(cranfield_docs):
    cranfield_terms = set()
    for doc in cranfield_docs:
        for word in doc:
            if word in vocab:  # Only include known English words
                cranfield_terms.add(word)
    return cranfield_terms

# -------- Step 2 (optional): Fetch full Wikipedia pages --------
def fetch_wikipedia_articles(terms, delay=1.0):
    articles = {}
    count = 1
    for term in terms:
        try:
            page = wikipedia.page(term)
            articles[term] = page.content
            print(f"Fetched article for {count}-th term: {term}")
        except Exception as e:
            print(f"Skipped term '{term}': {e}")
        time.sleep(delay)
        count += 1
    return articles

# -------- Step 3: Build the ESA term-concept matrix --------
def build_term_concept_matrix_from_articles(articles, max_features=10000):
    texts = list(articles.values())  # List of all summaries/articles
    vectorizer = TfidfVectorizer(max_features=max_features)  # TF-IDF to capture concept weights
    term_concept_matrix_sparse = vectorizer.fit_transform(texts)  # Shape: (n_articles, n_terms)
    term_concept_matrix = term_concept_matrix_sparse.T.toarray()  # Transpose to get (n_terms, n_articles)
    vocab = vectorizer.get_feature_names_out()
    print(f"Matrix shape: {term_concept_matrix.shape}")
    return term_concept_matrix, vocab

# -------- Step 4: Save ESA model components --------
def save_esa_model(term_concept_matrix, vocab, prefix='cranfield_esa_model'):
    np.save(f"{prefix}_matrix.npy", term_concept_matrix)
    with open(f"{prefix}_vocab.pkl", 'wb') as f:
        pickle.dump(vocab, f)
    print(f"Saved ESA matrix to {prefix}_matrix.npy and vocab to {prefix}_vocab.pkl")

# -------- Load ESA model (if needed later) --------
def load_esa_model(prefix='cranfield_esa_model'):
    term_concept_matrix = np.load(f"{prefix}_matrix.npy")
    with open(f"{prefix}_vocab.pkl", 'rb') as f:
        vocab = pickle.load(f)
    print(f"Loaded ESA matrix shape: {term_concept_matrix.shape}")
    print(f"Loaded vocab size: {len(vocab)}")
    return term_concept_matrix, vocab

# -------- Helper: Preprocess nested document list into flat structure --------
def preprocess(docs):
    """
    Flattens each document (list of sentences) into a single list of words.
    """
    flattened_docs = []
    for doc in docs:
        for sentence in doc:
            flattened_docs.append(sentence)
    return flattened_docs


# -------- MAIN execution block --------
if __name__ == "__main__":
    # Load preprocessed Cranfield documents (stopwords removed)
    with open('output/stopword_removed_docs.txt', 'r', encoding='utf-8') as f:
        data_docs_str = f.read()

    # Convert string to list object
    data_docs = ast.literal_eval(data_docs_str)

    # Flatten each document from list-of-lists to single word lists
    processed_cranfield_docs = preprocess(data_docs)

    # Step 1: Extract meaningful terms
    cranfield_terms = extract_unique_terms(processed_cranfield_docs)
    print(f"Extracted {len(cranfield_terms)} unique terms.")

    # Step 2: Fetch Wikipedia summaries for terms
    wiki_articles = fetch_wikipedia_articles_fast(cranfield_terms, delay=0.5)
    print(f"Fetched {len(wiki_articles)} Wikipedia articles.")

    # Step 3: Build term-concept matrix (ESA representation)
    term_concept_matrix, vocab = build_term_concept_matrix_from_articles(wiki_articles, max_features=10000)
    print("term_concept_matrix shape", term_concept_matrix.shape)

    # Step 4: Save matrix and vocab
    save_esa_model(term_concept_matrix, vocab)
