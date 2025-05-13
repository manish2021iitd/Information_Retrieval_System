from util import *

class StopwordRemoval():

    def fromList(self, text):
        """
        Sentence Segmentation using the Punkt Tokenizer

        Parameters
        ----------
        arg1 : list
            A list of lists where each sub-list is a sequence of tokens
            representing a sentence

        Returns
        -------
        list
            A list of lists where each sub-list is a sequence of tokens
            representing a sentence with stopwords removed
        """

        stopwordRemovedText = None

        #Fill in code here
        # Load the predefined list of English stopwords from NLTK
        stop_words = set(stopwords.words('english'))  
        
        # Remove stopwords from each sentence
        # Iterate through each token in the sentence and keep only those not in the stopword list
        stopwordRemovedText = [[token for token in sentence if token.lower() not in stop_words] for sentence in text]

        return stopwordRemovedText
