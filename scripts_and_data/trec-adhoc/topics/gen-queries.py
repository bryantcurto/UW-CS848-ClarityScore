#!/usr/bin/env python3

import os, sys
import enum
import re
import nltk

if len(sys.argv) != 2:
	print("USAGE: %s INPUT_FILEPATH OUTPUT_DIR", sys.argv[0])

input_filepath = sys.argv[1]
output_dir = sys.argv[2]

topics = []

class State(enum.IntEnum):
	NUM = 0
	TITLE = 1
	DESC = 2
	NARR = 3
	NONE = 4

topic_re = re.compile("^ *</top> *")
num_re = re.compile("^ *<num> *([Nn]umber: *)?")
title_re = re.compile("^ *<title> *")
desc_re = re.compile("^ *<desc> *([Dd]escription: *)?")
narr_re = re.compile("^ *<narr> *([Nn]arrative: *)?")

state_re_pairs = [(State.NUM, num_re), (State.TITLE, title_re), (State.DESC, desc_re), (State.NARR, narr_re)]

with open(input_filepath) as fh:
	extracted = [""] * 4
	state = State.NONE
	for line in fh:
		
		if topic_re.match(line):
			topics.append([v.strip() for v in extracted])
			extracted = [""] * 4
			state = State.NONE
			continue
		for test_state, test_re in state_re_pairs:
			match = test_re.match(line)
			if match:
				line = line[len(match.group(0)):]
				state = test_state

		line = line.strip()
		if State.NONE != state and len(line) > 0:
			extracted[state] += line + " "

for topic in topics:
	topicnum = int(topic[State.NUM])

	for idx, part in [(State.TITLE, "title"), (State.DESC, "desc"), (State.NARR, "narr")]:
		with open(os.path.join(output_dir, "%d_%s.txt" % (topicnum, part)), 'w') as fh:
			fh.write(','.join(['"%s"' % s for s in nltk.tokenize.word_tokenize(topic[idx])]))

for topic in topics:
	print("Topic:", int(topic[State.NUM]))
	print("Title:", topic[State.TITLE])
	print("  %s" % ','.join(['"%s"' % s for s in nltk.tokenize.word_tokenize(topic[State.TITLE])]))
	print("Description:", topic[State.DESC])
	print("  %s" % ','.join(['"%s"' % s for s in nltk.tokenize.word_tokenize(topic[State.DESC])]))
	print("Narrative:", topic[State.NARR])
	print("  %s" % ','.join(['"%s"' % s for s in nltk.tokenize.word_tokenize(topic[State.NARR])]))
	print("======================================================")

