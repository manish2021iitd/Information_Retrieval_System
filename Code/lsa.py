# lsa.py

from sklearn.decomposition import TruncatedSVD  # Import TruncatedSVD for Latent Semantic Analysis (LSA)
from sklearn.feature_extraction.text import TfidfVectorizer  # Import TfidfVectorizer to transform text into numerical data
import numpy as np  # Import NumPy for numerical operations
import matplotlib.pyplot as plt  # Import matplotlib for plotting

class LSA:
    def __init__(self, n_components=2):
        """
        Initialize the LSA model with a specific number of components (default is 2).
        """
        self.vectorizer = TfidfVectorizer()  # Initialize TfidfVectorizer for text vectorization
        self.svd = TruncatedSVD(n_components=n_components)  # Initialize SVD for LSA, set number of components
        self.doc_vectors = None  # Placeholder for document vectors after transformation
        self.docIDs = None  # Placeholder for document IDs

    def preprocess(self, docs):
        """
        Flattens the input (list of documents -> list of sentences -> list of words)
        into strings where each document becomes a single string.
        """
        flattened_docs = []  # Initialize an empty list to hold the processed documents
        for doc in docs:
            # Join words in each sentence to create a sentence string
            sentences = [' '.join(sentence) for sentence in doc]  
            # Join all sentences in the document to create a single document string
            doc_text = ' '.join(sentences)  
            flattened_docs.append(doc_text)  # Append the document to the list
        return flattened_docs  # Return the processed list of documents

    def fit(self, docs, docIDs):
        """
        Fit the LSA model to the given documents and associate document IDs.
        """
        self.docIDs = docIDs  # Store the document IDs
        flattened = self.preprocess(docs)  # Preprocess the documents
        tfidf = self.vectorizer.fit_transform(flattened)  # Convert documents into TF-IDF matrix
        self.doc_vectors = self.svd.fit_transform(tfidf)  # Apply SVD to the TF-IDF matrix to get document vectors

        # Plot the singular values (explained variance) from the SVD
        plt.figure(figsize=(10, 6))  # Set the figure size for the plot
        plt.plot(range(1, len(self.svd.explained_variance_) + 1), self.svd.explained_variance_, marker='o')  # Plot singular values
        plt.title('Singular Values from Truncated SVD (LSA)')  # Set title of the plot
        plt.xlabel('Component Index')  # Set x-axis label
        plt.ylabel('Explained Variance (Singular Value²)')  # Set y-axis label
        plt.grid(True)  # Add gridlines to the plot
    
        # Save the plot to a file
        plt.savefig('singular_values.png', dpi=300, bbox_inches='tight')  # Save the plot as a PNG image

    def rank(self, queries):
        """
        Rank the documents based on their similarity to the given queries.
        """
        flattened_queries = self.preprocess(queries)  # Preprocess the queries
        query_vecs = self.vectorizer.transform(flattened_queries)  # Transform the queries into TF-IDF vectors
        query_lsa = self.svd.transform(query_vecs)  # Apply SVD to the query vectors

        doc_IDs_ordered = []  # Initialize an empty list to hold ordered document IDs
        for qvec in query_lsa:
            # Compute cosine similarity between query vector and document vectors
            sims = self.doc_vectors @ qvec.T  
            ranked = np.argsort(sims)[::-1]  # Sort the similarity scores in descending order
            ranked_ids = [self.docIDs[i] for i in ranked]  # Get the document IDs in ranked order
            doc_IDs_ordered.append(ranked_ids)  # Append the ordered document IDs for this query
        return doc_IDs_ordered  # Return the list of ordered document IDs for each query
