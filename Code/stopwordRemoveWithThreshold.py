from util import *
from collections import Counter
import nltk

nltk.download('stopwords')

class StopwordRemoval:

    def create_corpus_stopwords(self, tokenized_documents, threshold=0.6):
        """
        Generates a stopword list based on word frequency in the dataset.

        Parameters:
        - tokenized_documents: List of tokenized sentences from all documents.
        - threshold: Percentage of documents in which a word must appear to be considered a stopword.

        Returns:
        - corpus_stopwords: A set of stopwords specific to the dataset.
        """
        # Get the total number of documents in the dataset
        doc_count = len(tokenized_documents)
        
        # Initialize a counter to track how many documents contain each word
        word_doc_frequency = Counter()
        
        # Iterate through each document in the dataset
        for doc in tokenized_documents:
            # Convert the document into a set to ensure each word is counted only once per document
            unique_words = set(doc)
            
            # Update the word frequency count for each unique word in the document
            for word in unique_words:
                word_doc_frequency[word] += 1
        
        # Define stopwords as words that appear in more than the given 'threshold' percentage of documents
        corpus_stopwords = {word for word, freq in word_doc_frequency.items() if freq / doc_count >= threshold}
        
        # Return the generated list of corpus-specific stopwords
        return corpus_stopwords


    def fromList(self, text, use_corpus_stopwords=True, corpus_threshold=0.6):
        """
        Removes stopwords from tokenized text using either NLTK's list or a corpus-specific list.

        Parameters
        ----------
        text : list
            A list of lists where each sub-list is a sequence of tokens representing a sentence.
        use_corpus_stopwords : bool
            Whether to use a corpus-specific stopword list instead of NLTK's default list.
        corpus_threshold : float
            The threshold for determining corpus-specific stopwords (default: 90% of documents).

        Returns
        -------
        list
            A list of lists where each sub-list is a sequence of tokens with stopwords removed.
        """

        stopwordRemovedText = None

        # Check whether to use corpus-specific stopwords or NLTK's stopword list
        if use_corpus_stopwords:
            # Generate a dynamic stopword list based on the given corpus using the specified threshold
            stop_words = self.create_corpus_stopwords(text, threshold=corpus_threshold)
        else:
            # Use NLTK's standard English stopword list for removal
            stop_words = set(stopwords.words('english'))
        
        # Remove stopwords from the text
        # Iterate through each sentence and keep only the tokens that are not in the stopword list
        stopwordRemovedText = [[token for token in sentence if token.lower() not in stop_words] for sentence in text]
        
        # Return the processed text with stopwords removed
        return stopwordRemovedText

