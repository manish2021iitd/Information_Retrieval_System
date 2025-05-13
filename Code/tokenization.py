from util import *

class Tokenization():

    def naive(self, text):
        """
        Tokenization using a Naive Approach
        
        Parameters
        ----------
        arg1 : list
            A list of strings where each string is a single sentence
        
        Returns
        -------
        list
            A list of lists where each sub-list is a sequence of tokens
        """

        tokenizedText = None

		#Fill in code here  
        # Initialize an empty list to store tokenized sentences
        tokenizedText = []
        
        # Loop through each sentence in the input text
        for sentence in text:
            # Use regex to extract words, contractions ("can't"), and punctuation marks
            tokens = re.findall(r"\w+(?:'\w+)?|[.,!?;:]", sentence)  
            # Append the tokenized sentence (list of words and punctuation) to the result
            tokenizedText.append(tokens)

        return tokenizedText



    def pennTreeBank(self, text):
        """
        Tokenization using the Penn Tree Bank Tokenizer
    
        Parameters
        ----------
        arg1 : list
            A list of strings where each string is a single sentence
    
        Returns
        -------
        list
            A list of lists where each sub-list is a sequence of tokens
        """
    
        tokenizedText = None
    
        #Fill in code here
        text = [sentence[:-1] for sentence in text]
        # Initialize the Penn Treebank tokenizer
        tokenizer = TreebankWordTokenizer()
        
        # Tokenize each sentence in the input text using the Penn Treebank tokenizer more accurately
        tokenizedText = [tokenizer.tokenize(sentence) for sentence in text]
     
        return tokenizedText