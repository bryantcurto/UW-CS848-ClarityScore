#!/usr/bin/env python3

import os, sys
import re
import numpy as np

old_ap = {}
new_ap = {}
with open(sys.argv[1]) as fh:
	for line in fh:
		vals = line.strip().split(' ')
		res = re.match("bm25-count1000_trec6_([0-9]*)_title$", vals[0])
		if res is not None:
			old_ap[res.groups(0)[0]] = float(vals[1])
		else:
			res = re.match("bm25-count1000_trec6_([0-9]*)_bm25-count1000-trec6-cs1-title$", vals[0])
			if res is not None:
				new_ap[res.groups(0)[0]] = float(vals[1])

#print(old_ap)
#print(new_ap)
assert(list(sorted(old_ap.keys())) == list(sorted(new_ap.keys())))

diffs = []
for key in old_ap.keys():
	diffs.append(new_ap[key] - old_ap[key])

diffs = np.array(diffs)
print(len(diffs), np.mean(diffs), np.std(diffs))
