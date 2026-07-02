#!/usr/bin/env python
"""
This script reads word counts from STDIN and aggregates
the counts for any duplicated words.

INPUT & OUTPUT FORMAT:
    word \t count
USAGE (standalone):
    python aggregateCounts_v2.py < yourCountsFile.txt

Instructions:
    For Q7 - Your solution should not use a dictionary or store anything   
             other than a single total count - just print them as soon as  
             you've added them. HINT: you've modified the framework script 
             to ensure that the input is alphabetized; how can you 
             use that to your advantage?
"""

# imports
import sys


################# YOUR CODE HERE #################
current_word = None
current_total = 0

for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    try:
        word, count = line.split("\t", 1)
        count = int(count)
    except ValueError:
        # skip malformed lines
        continue

    if current_word == word:
        current_total += count
    else:
        # output previous word before moving on
        if current_word is not None:
            print(f"{current_word}\t{current_total}")
        current_word = word
        current_total = count
# remove the last word
if current_word is not None:
    print(f"{current_word}\t{current_total}")












################ (END) YOUR CODE #################
