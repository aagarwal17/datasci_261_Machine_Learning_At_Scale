#!/usr/bin/env python
"""
Reducer takes words with their class and partial counts and computes totals.
INPUT:
    word \t class \t partialCount 
OUTPUT:
    word \t class \t totalCount  
"""
import re
import sys

# initialize trackers
current_word = None
spam_count, ham_count = 0,0

# read from standard input
for line in sys.stdin:
    # parse input
    word, is_spam, count = line.split('\t')
    
############ YOUR CODE HERE #########
    count = int(count)
    if current_word is None:
        current_word = word
    if word != current_word:
        # output both counts for the finished word
        if spam_count > 0:
            print(f"{current_word}\t1\t{spam_count}")
        if ham_count > 0:
            print(f"{current_word}\t0\t{ham_count}")
        # reset trackers
        current_word = word
        spam_count, ham_count = 0, 0
    # add to the counters
    if is_spam == "1":
        spam_count += count
    else:
        ham_count += count
# flush the last word
if current_word is not None:
    if spam_count > 0:
        print(f"{current_word}\t1\t{spam_count}")
    if ham_count > 0:
        print(f"{current_word}\t0\t{ham_count}")
############ (END) YOUR CODE #########