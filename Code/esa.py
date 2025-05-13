import numpy as np  # For numerical operations
from sklearn.feature_extraction.text import TfidfVectorizer  # For converting text to TF-IDF vectors
from sklearn.metrics.pairwise import cosine_similarity  # For computing cosine similarity

# Define a class for ESA-based document retrieval
class ESARetrieval:

    def __init__(self, term_concept_matrix, vocab):
        """
        term_concept_matrix: np.ndarray of shape (num_terms, num_concepts)
        vocab: list of terms corresponding to rows in term_concept_matrix
        """
        self.term_concept_matrix = term_concept_matrix  # ESA matrix mapping terms to concepts
        self.vocab = vocab  # Vocabulary list
        self.doc_ids = []  # List to store document IDs
        self.docs = []  # List to store processed documents
        self.vectorizer = TfidfVectorizer(vocabulary=vocab)  # TF-IDF vectorizer using fixed vocabulary
        self.doc_concept_matrix = None  # Matrix to store document vectors in concept space

    def preprocess(self, docs):
        # Convert list of tokenized documents into plain text strings
        return [' '.join(token for sentence in doc for token in sentence) for doc in docs]

    def buildIndex(self, docs, doc_ids):
        self.docs = self.preprocess(docs)  # Preprocess documents into strings
        self.doc_ids = doc_ids  # Save document IDs
        doc_term_matrix = self.vectorizer.fit_transform(self.docs)  # TF-IDF matrix of documents (num_docs × num_terms)
        print(f"doc_term_matrix shape: {doc_term_matrix.shape}")

        # Project TF-IDF vectors into concept space using ESA
        self.doc_concept_matrix = doc_term_matrix.dot(self.term_concept_matrix)  # (num_docs × num_concepts)
        print(f"doc_concept_matrix shape: {self.doc_concept_matrix.shape}")

    def rank(self, queries):
        processed_queries = self.preprocess(queries)  # Preprocess query tokens into strings
        query_term_matrix = self.vectorizer.transform(processed_queries)  # Convert queries to TF-IDF vectors
        query_concept_matrix = query_term_matrix.dot(self.term_concept_matrix)  # Map queries to concept space

        doc_IDs_ordered = []  # List to store rankings for each query
        for query_vec in query_concept_matrix:
            query_vec_2d = query_vec.reshape(1, -1)  # Reshape query vector to 2D
            query_scores = cosine_similarity(query_vec_2d, self.doc_concept_matrix).flatten()  # Compute similarities
            ranked_indices = np.argsort(query_scores)[::-1]  # Sort indices in descending order of score
            ranked_doc_ids = [self.doc_ids[i] for i in ranked_indices]  # Get document IDs by rank
            doc_IDs_ordered.append(ranked_doc_ids)  # Add to results
        return doc_IDs_ordered  # Return list of ranked doc IDs for each query
    
    def similarity(self, query):
        """
        Returns list of (doc_id, similarity_score) for a single query
        """
        processed_query = self.preprocess([query])  # Preprocess the query
        query_term_matrix = self.vectorizer.transform(processed_query)  # Convert to TF-IDF vector
        query_concept_vector = query_term_matrix.dot(self.term_concept_matrix)  # Map to concept space

        query_vector = query_concept_vector.reshape(1, -1)  # Reshape to 2D vector
        similarity_scores = cosine_similarity(query_vector, self.doc_concept_matrix).flatten()  # Compute scores

        return list(zip(self.doc_ids, similarity_scores))  # Return (doc_id, score) pairs
