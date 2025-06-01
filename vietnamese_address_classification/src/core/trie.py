import re
from rapidfuzz import fuzz, process
from vietnamese_address_classification.src.utils.normalization import remove_diacritics

class TrieNode:
    """
    Represents a single node within a Trie.
    
    Attributes:
        children (dict): A dictionary mapping a character to a TrieNode.
        is_terminal (bool): Whether this node represents the end of a valid word.
        full_words (list): List of original words (with diacritics) if this node is the end of a valid entry.
    """
    def __init__(self):
        self.children = {}
        self.is_terminal = False
        self.full_words = []  # Changed from full_word to full_words list


class Trie:
    """
    A Trie (prefix tree) for storing and searching strings.
    
    Methods:
        insert(word, original_name=None):
            Insert a normalized 'word' into the trie. 
            'original_name' can store a more complete or diacritics version.
        search_exact(word) -> bool:
            Return True if 'word' exists exactly as a terminal node in the trie.
        get_full_word(word) -> str or None:
            Return the stored original form if 'word' is found, else None.
        collect_all_words() -> list[str]:
            Collect all valid words stored in the trie (in normalized form).
    """
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word, original_name=None):
        """
        Insert a 'word' into the trie (usually already normalized).
        
        :param word: The normalized string to insert.
        :param original_name: Optionally store the original form (with diacritics).
        """
        try:
            current = self.root
            for char in word:
                if char not in current.children:
                    current.children[char] = TrieNode()
                current = current.children[char]
            current.is_terminal = True
            # Store the original name if provided
            if original_name:
                # Only add if not already in the list to avoid duplicates
                if original_name not in current.full_words:
                    current.full_words.append(original_name)
        except Exception as e:
            print(f"Error inserting word '{word}': {str(e)}")

    def search_exact(self, word):
        """
        Check if 'word' exists in the trie as a terminal node.
        
        :param word: The string to search for (normalized).
        :return: True if found exactly, False otherwise.
        """
        try:
            current = self.root
            for char in word:
                if char not in current.children:
                    return False
                current = current.children[char]
            return current.is_terminal
        except Exception as e:
            print(f"Error in exact search for '{word}': {str(e)}")
            return False

    def get_full_word(self, word, original_text=None):
        """
        Retrieve the original form of 'word' if it exists in the trie.
        If original_text is provided, try to find the best matching diacritics version.
        
        Args:
            word: The normalized string to look up
            original_text: Optional original text with diacritics to use as a hint
        
        Returns:
            The best matching original form if found, otherwise None
        """
        try:
            # If we have original text, first try to find a longer form
            if original_text and ' ' in original_text:
                # Try to find the longer form first
                longer_word = remove_diacritics(original_text.lower())
                current = self.root
                for char in longer_word:
                    if char not in current.children:
                        break
                    current = current.children[char]
                else:  # If we found the longer form
                    if current.is_terminal and current.full_words:
                        # Use rapidfuzz to find the closest match
                        result = process.extractOne(
                            original_text,
                            current.full_words,
                            scorer=fuzz.ratio
                        )
                        if result:
                            return result[0]
            
            # If longer form not found or no original text, try the original word
            current = self.root
            for char in word:
                if char not in current.children:
                    return None
                current = current.children[char]
            
            if current.is_terminal and current.full_words:
                # If we have original text, use it to find the best match
                if original_text:
                    # Use rapidfuzz to find the closest match
                    result = process.extractOne(
                        original_text,
                        current.full_words,
                        scorer=fuzz.ratio
                    )
                    if result:
                        return result[0]
                
                # If no original text or no good match found, return the first version
                return current.full_words[0]
                
            return None
            
        except Exception as e:
            print(f"Error getting full word for '{word}': {str(e)}")
            return None

    def collect_all_words(self):
        """
        Collect all valid (normalized) words stored in the trie.
        
        :return: A list of normalized strings representing all valid entries.
        """
        try:
            results = []

            def dfs(node, path):
                if node.is_terminal:
                    results.extend(node.full_words)
                for c, child in node.children.items():
                    dfs(child, path + c)

            dfs(self.root, "")
            return results
        except Exception as e:
            print(f"Error collecting words from trie: {str(e)}")
            return []
        

def build_trie_from_list(words_list):
    """
    Build a trie from a list of location names.
    Each location name is normalized for insertion, 
    but we store the original form (with diacritics) in the node.
    
    :param words_list: List of strings representing location names.
    :return: A Trie object containing the location data.
    """
    t = Trie()
    for w in words_list:
        w_clean = w.strip()
        # normalized version
        w_norm = remove_diacritics(w_clean.lower())
        t.insert(w_norm, original_name=w_clean)
    return t

def is_ambiguous_name(name, district_trie, ward_trie):
    """
    Check if a name appears in both district and ward lists.
    
    Args:
        name: The normalized name to check
        district_trie: Trie containing district data
        ward_trie: Trie containing ward data
    Returns:
        bool: True if the name appears in both lists, False otherwise
    """
    try:
        return (district_trie.search_exact(name) and 
                ward_trie.search_exact(name))
    except Exception as e:
        print(f"Error in is_ambiguous_name: {str(e)}")
        return False