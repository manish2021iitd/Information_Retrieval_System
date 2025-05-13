import nltk
import string
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sentenceSegmentation import SentenceSegmentation
from tokenization import Tokenization
from inflectionReduction import InflectionReduction
from nltk.corpus import stopwords
import argparse
import json
from sys import version_info


# Input compatibility for Python 2 and Python 3
if version_info.major == 3:
    pass
elif version_info.major == 2:
    try:
        input = raw_input
    except NameError:
        pass
else:
    print ("Unknown python version - input function not safe")
	

# Function to detect stop-words in bottom up approach.
def custom_stopwords(tokenized_corpus):
    documents = [" ".join([" ".join(sentence) for sentence in doc]) for doc in tokenized_corpus]
	
    # Flatten the list to count word frequency
    all_tokens = [word for doc in documents for word in doc]
    word_freq = Counter(all_tokens)

    # Identify top frequent words (Threshold: Top 20%)
    top_n = int(len(word_freq) * 0.2)
    most_common_words = [word for word, freq in word_freq.most_common(top_n)]

    # Compute TF-IDF scores to refine stopwords
    vectorizer = TfidfVectorizer(stop_words=None)
    X = vectorizer.fit_transform(documents)
    idf_scores = dict(zip(vectorizer.get_feature_names_out(), vectorizer.idf_))

    # Words with lowest IDF values (high document frequency) are potential stopwords
    low_idf_threshold = sorted(idf_scores.values())[top_n]
    low_idf_words = [word for word, score in idf_scores.items() if score <= low_idf_threshold]

    # Final custom stopword list
    custom_stopwords = set(most_common_words) | set(low_idf_words)
    return custom_stopwords


class StopWordEngine:

	def __init__(self, args):
		self.args = args
		self.tokenizer = Tokenization()
		self.sentenceSegmenter = SentenceSegmentation()
		self.inflectionReducer = InflectionReduction()

	def segmentSentences(self, text):
		"""
		Return the required sentence segmenter
		"""
		if self.args.segmenter == "naive":
			return self.sentenceSegmenter.naive(text)
		elif self.args.segmenter == "punkt":
			return self.sentenceSegmenter.punkt(text)

	def tokenize(self, text):
		"""
		Return the required tokenizer
		"""
		if self.args.tokenizer == "naive":
			return self.tokenizer.naive(text)
		elif self.args.tokenizer == "ptb":
			return self.tokenizer.pennTreeBank(text)

	def reduceInflection(self, text):
		"""
		Return the required stemmer/lemmatizer
		"""
		return self.inflectionReducer.reduce(text)


	def preprocessDocs(self, docs):
		"""
		Preprocess the documents
		"""
		
		# Segment docs
		segmentedDocs = []
		for doc in docs:
			segmentedDoc = self.segmentSentences(doc)
			segmentedDocs.append(segmentedDoc)
		json.dump(segmentedDocs, open(self.args.out_folder + "segmented_docs.txt", 'w'))

		# Tokenize docs
		tokenizedDocs = []
		for doc in segmentedDocs:
			tokenizedDoc = self.tokenize(doc)
			tokenizedDocs.append(tokenizedDoc)
		json.dump(tokenizedDocs, open(self.args.out_folder + "tokenized_docs.txt", 'w'))

		# Stem/Lemmatize docs
		reducedDocs = []
		for doc in tokenizedDocs:
			reducedDoc = self.reduceInflection(doc)
			reducedDocs.append(reducedDoc)
			
		json.dump(reducedDocs, open(self.args.out_folder + "reduced_docs.txt", 'w'))
		custom_stop_words = custom_stopwords(reducedDocs)
		print(custom_stop_words)
		json.dump(list(custom_stop_words), open(self.args.out_folder + "customized_stop_words.txt", 'w'))
		stop_words = set(stopwords.words('english')) 
		json.dump(list(stop_words), open(self.args.out_folder + "nltk_stop_words.txt", 'w'))



	def evaluateDataset(self):
		"""
		Evaluate document-query relevances for all document-query pairs
		"""

		# Read documents
		docs_json = json.load(open(args.dataset + "cran_docs.json", 'r'))[:]
		docs = [item["body"] for item in docs_json]
		# Process documents
		processedDocs = self.preprocessDocs(docs)

		# Remaning code will be added later



if __name__ == "__main__":

	# Create an argument parser
	parser = argparse.ArgumentParser(description='custom_stopword.py')

	# Tunable parameters as external arguments
	parser.add_argument('-dataset', default = "cranfield/", 
						help = "Path to the dataset folder")
	parser.add_argument('-out_folder', default = "output/", 
						help = "Path to output folder")
	parser.add_argument('-segmenter', default = "punkt",
	                    help = "Sentence Segmenter Type [naive|punkt]")
	parser.add_argument('-tokenizer',  default = "ptb",
	                    help = "Tokenizer Type [naive|ptb]")
	parser.add_argument('-custom', action = "store_true", 
						help = "Take custom query as input")
	
	# Parse the input arguments
	args = parser.parse_args()

	# Create an instance of the Search Engine
	searchEngine = StopWordEngine(args)

	# Either handle query from user or evaluate on the complete dataset 
	if args.custom:
		searchEngine.handleCustomQuery()
	else:
		searchEngine.evaluateDataset()
