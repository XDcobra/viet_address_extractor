import re

def expand_abbreviations(text):
    """
    Detect specific patterns like 'p1' -> 'phuong 1', 'q3' -> 'quan 3', etc.
    Also handles variations with dots and spaces like 'p.1', 'q. 3'
    
    :param text: A single token (already lowercase, punctuation removed).
    :return: Expanded string if matched, otherwise the original.
    """
    # Clean up extra spaces
    text = text.strip()
    
    # Handle patterns with optional dot and spaces
    # p1, p.1, p. 1 => phuong 1
    match_p = re.match(r'^p\.?\s*(\d+)$', text)
    if match_p:
        return f"phuong {match_p.group(1)}"

    # q1, q.1, q. 1 => quan 1
    match_q = re.match(r'^q\.?\s*(\d+)$', text)
    if match_q:
        return f"quan {match_q.group(1)}"

    # If no match, return original
    return text


def build_abbreviation_map(provinces, districts, wards):
    """
    Build a mapping of abbreviations to their full names.
    Only includes unambiguous mappings (where the abbreviation matches exactly one name).
    """
    abbrev_map = {}
    
    # Helper function to add abbreviation if unambiguous
    def add_if_unambiguous(abbrev, full_name):
        if abbrev in abbrev_map:
            # If we've seen this abbreviation before, it's ambiguous
            del abbrev_map[abbrev]
        else:
            abbrev_map[abbrev] = full_name
    
    # Process all names
    for name_list in [provinces, districts, wards]:
        for name in name_list:
            # Skip names that are too short
            if len(name) <= 3:
                continue
                
            # Try different abbreviation patterns
            words = name.split()
            if len(words) >= 2:
                # Try first letters of each word
                abbrev = ''.join(w[0] for w in words)
                if 2 <= len(abbrev) <= 3:
                    add_if_unambiguous(abbrev, name)
            
            # Try first 2-3 letters of the name
            for length in [2, 3]:
                abbrev = name[:length]
                if abbrev.isalpha():
                    add_if_unambiguous(abbrev, name)
    
    return abbrev_map