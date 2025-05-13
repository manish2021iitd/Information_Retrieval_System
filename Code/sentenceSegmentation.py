from util import *

class SentenceSegmentation():

    def naive(self, text):
        """
        Sentence Segmentation using a Naive Approach

        Parameters
        ----------
        arg1 : str
            A string (a bunch of sentences)

        Returns
        -------
        list
            A list of strings where each string is a single sentence
        """
        segmentedText = None

        #Fill in code here
        # Splitting by common sentence delimiters (., ?, !, ;, :)
        segmentedText = re.split(r'(?<=[.!?;:])\s+', text.strip())
        return segmentedText

    def punkt(self, text):
        """
        Sentence Segmentation using the Punkt Tokenizer

        Parameters
        ----------
        arg1 : str
            A string (a bunch of sentences)

        Returns
        -------
        list
            A list of strings where each strin is a single sentence
        """

        segmentedText = None

        #Fill in code here
        nltk.download('punkt')  # Download the Punkt tokenizer (used for sentence segmentation)
        segmentedText = sent_tokenize(text)  # Split the text into sentences using the pre-trained Punkt tokenizer
        return segmentedText