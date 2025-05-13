from util import *
from nltk.metrics.distance import edit_distance

def filter_candidates_by_distance(candidates, error_word, max_distance=2):
    """
    Filters candidate words based on edit distance from the error word.
    
    Args:
        candidates (list of str): List of valid vocabulary words.
        error_word (str): The word to compare against.
        max_distance (int): Maximum allowed edit distance.
        
    Returns:
        list of str: Filtered candidates within the allowed edit distance.
    """
    return [w for w in candidates if edit_distance(w, error_word) <= max_distance]
