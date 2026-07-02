#!/usr/bin/env python
"""
This script reads lines from STDIN and returns a list of
all words and the count of how many times they occurred.

INPUT:
    a text file
OUTPUT FORMAT:
    word \t count
USAGE:
    python wordCount.py < yourTextFile.txt

Instructions:
    Fill in the missing code below so that the script
    prints tab separated word counts to Standard Output.
    NOTE: we have performed the tokenizing for you, please
    don't modify the provided code or you may fail unit tests.
"""

# imports
import sys
import re
from collections import defaultdict

counts = defaultdict(int)

# stream over lines from Standard Input
for line in sys.stdin:

    # tokenize
    line = line.strip()
    words = re.findall(r'[a-z]+', line.lower())

############ YOUR CODE HERE #########
    # count each word
    for word in words:
        counts[word] += 1
        
for word, count in counts.items():
    print(f"{word}\t{count}")


############ (END) YOUR CODE #########
