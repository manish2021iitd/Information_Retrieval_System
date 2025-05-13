# Add your import statements here

import re  # Import the regular expressions module for text processing
import nltk  # Import NLTK (Natural Language Toolkit)

# Sentence Segmentation: Import the Punkt sentence tokenizer to split text into sentences
from nltk.tokenize import sent_tokenize  

# Tokenization: Import the Penn Treebank tokenizer for word tokenization
from nltk.tokenize import TreebankWordTokenizer  

# Stopword Removal: Import the NLTK stopword list to filter out common words (e.g., "the", "is", "and")
from nltk.corpus import stopwords  

# Inflection Reduction: Import the Porter Stemmer to perform stemming (reduce words to their base form)
from nltk.stem import PorterStemmer  

# IR
import math
from collections import defaultdict, Counter
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
# Add any utility functions here