#!/usr/bin/env python3

import os, sys
from clarity_score import ClarityScore

# Extract terms from query

with open(sys.argv[1]) as fh:
	query = fh.read().strip()

query_terms = []
token_start_idx = 0
while True:
	token_end_idx = query[token_start_idx + 1:].find('"') + token_start_idx + 1
	query_terms.append(query[token_start_idx + 1: token_end_idx])
	if token_end_idx == len(query) - 1:
		break
	else:
		assert(token_end_idx + 2 < len(query) and \
		       query[token_end_idx + 1] == ',' and query[token_end_idx + 2] == '"')
		token_start_idx = token_end_idx + 2

# Stem terms
query_terms = ClarityScore._stem_terms(query_terms)

# Remove emptry string terms
query_terms = [term for term in query_terms if len(term) > 0]

print(len(query_terms))
