# from util import *

# class InflectionReduction:

#     def reduce(self, text):
#         """
#         Stemming/Lemmatization

#         Parameters
#         ----------
#         arg1 : list
#             A list of lists where each sub-list a sequence of tokens
#             representing a sentence

#         Returns
#         -------
#         list
#             A list of lists where each sub-list is a sequence of
#             stemmed/lemmatized tokens representing a sentence
#         """
#         reducedText = None
#         #Fill in code here
#         # Initialize the Porter Stemmer for stemming words
#         stemmer = PorterStemmer()  
        
#         # Apply stemming to each token in every sentence
#         # The stemmer converts words to their base form (e.g., "running" -> "run", "studies" -> "studi")
#         reducedText = [[stemmer.stem(token) for token in sentence] for sentence in text]
        
#         return reducedText

from util import *  # Import helper functions from util.py (if any)
from nltk.stem import WordNetLemmatizer  # For lemmatization
from nltk.corpus import wordnet  # To map POS tags to WordNet format
from nltk import pos_tag  # For part-of-speech tagging
import nltk

# Download required resources for POS tagging and lemmatization
nltk.download('averaged_perceptron_tagger_eng')

# Ensure the WordNet corpus is available
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

# Ensure POS tagger is available
try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger')


# Class to perform lemmatization (inflection reduction)
class InflectionReduction:

    def reduce(self, text):
        """
        Lemmatization: Converts words to their base/dictionary form.
        This method uses POS tagging to choose the correct base form (e.g., 'running' → 'run' if verb).
        
        Args:
            text (list of list of str): Tokenized sentences (e.g., [['He', 'is', 'running'], ...])
        
        Returns:
            reducedText (list of list of str): Lemmatized text
        """
        lemmatizer = WordNetLemmatizer()  # Initialize lemmatizer

        # Helper function to map NLTK POS tags to WordNet POS tags
        def get_wordnet_pos(tag):
            if tag.startswith('J'):
                return wordnet.ADJ
            elif tag.startswith('V'):
                return wordnet.VERB
            elif tag.startswith('N'):
                return wordnet.NOUN
            elif tag.startswith('R'):
                return wordnet.ADV
            else:
                return wordnet.NOUN  # Default to noun if tag is unknown

        reducedText = []  # Final result: list of lemmatized sentences
        for sentence in text:
            pos_tags = pos_tag(sentence)  # Get POS tags for each word
            # Lemmatize each word using its POS tag
            lemmatized = [lemmatizer.lemmatize(word, get_wordnet_pos(tag)) for word, tag in pos_tags]
            reducedText.append(lemmatized)  # Add lemmatized sentence to result

        return reducedText  # Return the list of lemmatized sentences
