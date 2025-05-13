from util import *  # Import all functions and variables from util.py
import ast  # For safely evaluating strings as Python literals
import re  # For using regular expressions

# Read the tokenized queries from file
with open('output/tokenized_queries.txt', 'r', encoding='utf-8') as f:
    data_queries_str = f.read()  # Read file content as a string

data_queries = ast.literal_eval(data_queries_str)  # Convert string to Python list object

# Build vocabulary from query data
vocab_1 = set()  # Initialize an empty set to store unique words
for doc in data_queries:  # Loop through each document
    for sentence in doc:  # Loop through each sentence
        for word in sentence:  # Loop through each word
            vocab_1.add(word)  # Add word to vocabulary set

# Read the tokenized documents from file
with open('output/tokenized_docs.txt', 'r', encoding='utf-8') as f:
    data_docs_str = f.read()  # Read file content as a string

data_docs = ast.literal_eval(data_docs_str)  # Convert string to Python list object

# Build vocabulary from document data
vocab_2 = set()  # Initialize another set for doc words
for doc in data_docs:
    for sentence in doc:
        for word in sentence:
            vocab_2.add(word)  # Add word to document vocabulary

# Combine vocabularies from queries and documents
combined_vocab = vocab_1 | vocab_2  # Union of both vocab sets
combined_vocab_list = list(combined_vocab)  # Convert set to list

# Clean each word: remove all characters except letters and hyphens
cleaned_list = [re.sub(r'[^a-zA-Z-]', '', word) for word in combined_vocab_list]

# Convert words to lowercase and prepare final vocab list
# (Optional filtering lines commented out)
# correct_word_list = [word for word in cleaned_list if word.lower() in vocab]
# correct_words = list(combined_vocab)
# correct_word_list = [word.lower() for word in correct_words]
correct_word_list = [word.lower() for word in cleaned_list]  # Final cleaned, lowercased words

print(type(correct_word_list))  # Print the type of the word list (should be <class 'list'>)

count = 0  # Initialize a counter
# Write the final vocabulary words to a file
with open('vocab_words.txt', 'w') as f:
    for word in correct_word_list:
        if word:  # Skip empty strings
            f.write(word + '\n')  # Write word to file
            count += 1  # Increment word count
            
print(f"Saved {count} words to vocab_words.txt")  # Show total saved words
