from util import *

# Add your import statements here
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from lsa import LSA
from esa import ESARetrieval
from concept_matrix_computation import load_esa_model

term_concept_matrix, vocab = load_esa_model(prefix='cranfield_esa_model')


class InformationRetrieval():
    
    def __init__(self, method="tfidf"):
        assert method in ["TF-IDF", "LSA", "ESA", "HYBRID"], "method must be 'TF-IDF' or 'LSA' or 'ESA' or 'HYBRID'"
        self.method = method
        self.docIDs = None

        if self.method == "LSA":
            self.model = LSA(n_components=300)  # 900
        elif self.method == "ESA":
            print("calling esa")
            self.model = ESARetrieval(term_concept_matrix, vocab)
        elif self.method == "TF-IDF":
            self.vectorizer = TfidfVectorizer()
            self.doc_vectors = None
        elif self.method == "HYBRID":
            self.model = LSA(n_components=300)
            self.vectorizer = TfidfVectorizer()
            self.esa_model = ESARetrieval(term_concept_matrix, vocab)
            self.doc_vectors = None
        else:
            raise ValueError("Invalid retrieval method.")

    def preprocess(self, docs):
        """
        Flattens the input (list of documents -> list of sentences -> list of words)
        into strings where each document becomes a single string.
        """
        flattened_docs = []
        for doc in docs:
            sentences = [' '.join(sentence) for sentence in doc]  # join words in each sentence
            doc_text = ' '.join(sentences)  # join all sentences
            flattened_docs.append(doc_text)
        return flattened_docs
    

    def buildIndex(self, docs, docIDs):
        """
        Builds the document index in terms of the document
        IDs and stores it in the 'index' class variable

        Parameters
        ----------
        arg1 : list
            A list of lists of lists where each sub-list is
            a document and each sub-sub-list is a sentence of the document
        arg2 : list
            A list of integers denoting IDs of the documents
        Returns
        -------
        None
        """

        #index = None

        #Fill in code here
        self.docIDs = docIDs
        if self.method == "LSA":
            self.model.fit(docs, docIDs)
        elif self.method == "ESA":
            self.model.buildIndex(docs, docIDs)
        elif self.method == "HYBRID":
            self.docIDs = docIDs
            # TF-IDF
            flattened_docs = self.preprocess(docs)
            self.doc_vectors = self.vectorizer.fit_transform(flattened_docs)
            # LSA
            self.model.fit(docs, docIDs)
            # ESA
            self.esa_model.buildIndex(docs, docIDs)
        else:
            # Flatten documents into strings
            flattened_docs = self.preprocess(docs)
    
            # Create TF-IDF vectors
            # self.vectorizer = TfidfVectorizer()
            self.doc_vectors = self.vectorizer.fit_transform(flattened_docs)

        # Store index
        # self.index = self.doc_vectors

        #self.index = index
        # self.model.fit(docs, docIDs)


    def rank(self, queries):
        """
        Rank the documents according to relevance for each query

        Parameters
        ----------
        arg1 : list
            A list of lists of lists where each sub-list is a query and
            each sub-sub-list is a sentence of the query
        

        Returns
        -------
        list
            A list of lists of integers where the ith sub-list is a list of IDs
            of documents in their predicted order of relevance to the ith query
        """
        if self.method in ["LSA", "ESA"]:
            return self.model.rank(queries)
        elif self.method == "HYBRID":
            doc_IDs_ordered = []
            
            # Preprocess queries
            flattened_queries = self.preprocess(queries)

            # --- TF-IDF ---
            query_vectors = self.vectorizer.transform(flattened_queries)
            tfidf_sims = cosine_similarity(query_vectors, self.doc_vectors)

            # --- LSA ---
            query_lsa_vecs = self.model.svd.transform(self.model.vectorizer.transform(flattened_queries))
            doc_lsa_vecs = self.model.doc_vectors
            lsa_sims = query_lsa_vecs @ doc_lsa_vecs.T

            # --- ESA ---
            esa_sims = []
            for q in queries:
                sims = self.esa_model.similarity(q)  # list of (docID, score)
                sim_row = np.zeros(len(self.docIDs))
                id_to_index = {docID: idx for idx, docID in enumerate(self.docIDs)}
                for docID, score in sims:
                    sim_row[id_to_index[docID]] = score
                esa_sims.append(sim_row)
            esa_sims = np.array(esa_sims)

            # --- Normalize ---
            def normalize(matrix):
                return (matrix - matrix.min(axis=1, keepdims=True)) / (matrix.max(axis=1, keepdims=True) - matrix.min(axis=1, keepdims=True) + 1e-9)

            tfidf_norm = normalize(tfidf_sims)
            lsa_norm = normalize(lsa_sims)
            esa_norm = normalize(esa_sims)

            # --- Combine ---
            combined_sims = (0.2*tfidf_norm + 0.7*lsa_norm + 0.1*esa_norm)  # or use weighted average

            for similarities in combined_sims:
                ranked_doc_indices = np.argsort(similarities)[::-1]
                ranked_docIDs = [self.docIDs[idx] for idx in ranked_doc_indices]
                doc_IDs_ordered.append(ranked_docIDs)

            return doc_IDs_ordered

        elif self.method == "TF-IDF":
            doc_IDs_ordered = []
    
            #Fill in code here
            # Preprocess queries into flat text
            flattened_queries = self.preprocess(queries)
    
            # Transform queries using the trained TF-IDF vectorizer
            query_vectors = self.vectorizer.transform(flattened_queries)
    
            # Compute cosine similarity between queries and documents
            similarity_matrix = cosine_similarity(query_vectors, self.doc_vectors)
            # print("The similarity matrix is:")
            # print(similarity_matrix)
    
            # For each query, get the ranking of document IDs based on similarity scores
            for similarities in similarity_matrix:
                ranked_doc_indices = np.argsort(similarities)[::-1]  # descending order
                ranked_docIDs = [self.docIDs[idx] for idx in ranked_doc_indices]
                doc_IDs_ordered.append(ranked_docIDs)
                
            return doc_IDs_ordered


if __name__ == "__main__":
    docs = [
            [['this', 'is', 'the', 'first', 'document']],
            [['this', 'document', 'is', 'the', 'second', 'document']],
            [['and', 'this', 'is', 'the', 'third', 'one']],
            [['is', 'this', 'the', 'first', 'document']]]
    docIDs = [1, 2, 3, 4]

    queries = [
                [['this', 'is', 'the', 'first']],
                [['third', 'document']]]

    IR_system = InformationRetrieval()
    IR_system.buildIndex(docs, docIDs)
    
    print("The document vectorizer's shape is:")
    print(IR_system.doc_vectors.shape)

    ranks = IR_system.rank(queries)

    print(ranks)