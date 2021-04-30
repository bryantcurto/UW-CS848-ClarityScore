#!/usr/bin/env python3

import sys, os
import numpy as np
from fractions import Fraction

# Adapted from:
# https://github.com/benhamner/Metrics/blob/9a637aea795dc6f2333f022b0863398de0a1ca77/Python/ml_metrics/average_precision.py
# I just maked it a bit more efficient.
def apk(actual, predicted, k=10):
    """
    Computes the average precision at k.

    This function computes the average prescision at k between two lists of
    items.

    Parameters
    ----------
    actual : list
             A list of elements that are to be predicted (order doesn't matter)
    predicted : list
                A list of predicted elements (order does matter)
    k : int, optional
        The maximum number of predicted elements

    Returns
    -------
    score : double
            The average precision at k over the input lists

    """
    if len(predicted) > k:
        predicted = predicted[:k]

    score = Fraction(0)
    num_hits = 0

    actual_set = set()
    for item in actual:
        actual_set.add(item)

    predicted_set = set()
    for i,p in enumerate(predicted):
        if p in actual_set and p not in predicted_set:
            num_hits += 1
            score += Fraction(num_hits, i + 1)
        predicted_set.add(p)

    if not actual:
        return 0.

    return float(score / min(len(actual), k))


def parse_file(filepath, items_per_line):
    # Return None if file is empty
    if os.stat(filepath).st_size == 0:
        print("%s is empty!" % filepath)
        return None

    # Helper function for easily splitting elements in a line of the file
    # Only spaces and tabs are valid delimiters
    spaces = False
    def parse_line(line):
        items = line.strip().split(' ' if spaces else '\t')
        assert(len(items) == items_per_line)
        return items

    items = []
    with open(filepath) as fh:
        line = fh.readline().strip()
        if len(line.split(' ')) == items_per_line:
            spaces = True
        elif len(line.split('\t')) == items_per_line:
            pass
        else:
            print("Delimiters not found! %s could not be parsed." % filepath)
            return None

        items.append(parse_line(line))
        for line in fh:
            items.append(parse_line(line))
    return items


if len(sys.argv) != 3:
	print("USAGE: %s QRELS RUNFILE" % sys.argv[0])
	exit(-1)

qrels_items = parse_file(sys.argv[1], 4)
runfile_items = parse_file(sys.argv[2], 6)

if qrels_items is None or runfile_items is None:
    exit(-1)

unique_runfile_topics = list(set([i[0] for i in runfile_items]))
if len(unique_runfile_topics) != 1:
    print("Runfile can only contain retrevals for one topic: found %s" % ','.join(unique_runfile_topics))
    exit(-1)
topic = unique_runfile_topics[0]

relevant_docs = [item[2] for item in qrels_items if (item[0] == topic and item[-1] != '0')]
retrieved_docs = [item[2] for item in runfile_items]

print(apk(relevant_docs, retrieved_docs, len(retrieved_docs)))
