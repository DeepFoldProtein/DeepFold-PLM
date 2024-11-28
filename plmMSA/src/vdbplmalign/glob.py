import glob
import itertools
import re

def expand_braces(pattern):
    match = re.search(r"\{([^}]+)\}", pattern)
    if not match:
        return [pattern]
    
    options = match.group(1).split(',')
    
    expanded_patterns = []
    for option in options:
        expanded_patterns.extend(expand_braces(pattern[:match.start()] + option + pattern[match.end():]))
    
    return expanded_patterns

def glob_with_braces(pattern):
    expanded_patterns = expand_braces(pattern)
    return list(itertools.chain.from_iterable(glob.glob(p) for p in expanded_patterns))
