#!/usr/bin/env python

import sys, os
from scipy.stats import kendalltau

if len(sys.argv) != 3:
	print("USAGE: %s FILEPATH_TO_LIST FILEPATH_TO_LIST" % sys.argv[0])
	exit(-1)

def parse_file(filepath):
	with open(filepath) as fh:
		lines = [line.strip() for line in fh]
	return lines

list1 = parse_file(sys.argv[1])
list2 = parse_file(sys.argv[2])

correlation, pvalue = kendalltau(list1, list2)
print("correlation=%f p-value=%f" %(correlation, pvalue))
