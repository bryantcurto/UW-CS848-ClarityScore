#!/usr/bin/env python

import sys, os
from scipy.stats import kendalltau

if len(sys.argv) != 2:
	print("USAGE: %s SCORES_FILE" % sys.argv[0])
	exit(-1)

scores = []
with open(sys.argv[1]) as fh:
	for line in fh:
		query_id, average_precision, clarity_score = line.strip().split(' ')
		if average_precision == 'nan' or clarity_score == 'nan':
			print("%s has nan score!" % query_id)
			continue
		scores.append((query_id, float(average_precision), float(clarity_score)))

qids_by_ap = [v[0] for v in sorted(scores, key=lambda x: x[1])]
qids_by_cs = [v[0] for v in sorted(scores, key=lambda x: x[2])]

print(qids_by_ap)
print(qids_by_cs)

correlation, pvalue = kendalltau(qids_by_ap, qids_by_cs)
print("correlation=%f p-value=%f" %(correlation, pvalue))
