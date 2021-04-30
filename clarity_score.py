#!/usr/bin/env python

import sys, os
import nltk
import math, base64, pickle
import numpy as np
import multiprocessing
from collections import Counter
import argparse
import tqdm
import subprocess
from fractions import Fraction

LINES_CHUNK_SIZE=1000
CLARITY_SCORE_TERM_CONTRIB_CHUNK_SIZE = 10000

class BadTermError(RuntimeError):
	def __init__(self, term):
		self.term = term

class ClarityScore(object):
	def __init__(self, collection_filepath, smoothing_constant=0.6, workers=8):
		self.smoothing_constant = Fraction(smoothing_constant)
		self.workers = workers

		# I only really want restore passing None to constructor
		if collection_filepath is not None:
			self.collection_filepath = collection_filepath
			self.collection_language_model = \
					ClarityScore.gen_language_model((self.collection_filepath, self.workers, "Collection LM"))
		else:
			self.collection_filepath = None
			self.collection_language_model = None

	@staticmethod
	def _stem_terms(terms):
		proc = subprocess.Popen(args=["handyman", "STEMMING"], stdout=subprocess.PIPE, stdin=subprocess.PIPE, bufsize=1, universal_newlines=True)
		for term in terms:
			term = term.lower()
			proc.stdin.write(term + '\n')
		stemmed_terms = proc.communicate()[0].split('\n')[:-1]
		assert(len(stemmed_terms) == len(terms))
		return stemmed_terms

	@staticmethod
	def _parse_handyman_langmodel_lines(lines):
		counter = Counter()
		term_data = []
		for line in lines:
			term, _, collection_freq, _ = line.split(' ')
			term_data.append((term, int(collection_freq)))
		stemmed_terms = ClarityScore._stem_terms(list(zip(*term_data))[0])
		for (term, collection_freq), stemmed_term in zip(term_data, stemmed_terms):
			counter[stemmed_term] += collection_freq
		return counter

	@staticmethod
	def _iter_handyman_langmodel_output(filepath):
		with open(filepath) as fh:
			line = fh.readline().strip()
			while not line.startswith("# All following lines: TERM STEMMED_FORM CORPUS_FREQUENCY DOC_FREQUENCY"):
				line = fh.readline().strip()
			lines = []
			for i, line in enumerate(fh):
				if i > 0 and 0 == i % LINES_CHUNK_SIZE:
					yield lines
					lines = []
				lines.append(line.strip())
			if len(lines) > 0:
				yield lines

	@staticmethod
	def _merge(pair_merge_func, map_func, items, desc):
		# Create progress logging utilities
		task_tqdm = tqdm.tqdm(total=len(items) - 1, desc=desc)
		def update_tqdm_callback(arg):
			task_tqdm.update()
			return arg

		# Pair up items to be merged
		half_len = len(items) // 2
		unmerged_items = items[2 * half_len:]
		item_pairs = list(zip(items[:half_len], items[half_len:2 * half_len]))
		while len(item_pairs) > 0:
			# Sum pairs and generate a new list of items to be merged
			items = list(map(update_tqdm_callback, map_func(pair_merge_func, item_pairs))) + \
					   unmerged_items
			# Pair up items to be merged
			half_len = len(items) // 2
			unmerged_items = items[2 * half_len:]
			item_pairs = list(zip(items[:half_len], items[half_len:2 * half_len]))
		return unmerged_items[0]

	@staticmethod
	def _counter_add(counter_pair):
		c1, c2 = counter_pair
		return c1 + c2

	@staticmethod
	def gen_language_model(argspack): # There's no multiprocessing.Pool.startimap
		tokencounts_filepath, workers, desc = argspack
		#def merge_counters(map_func, counters):
		#	# Create progress logging utilities
		#	task_tqdm = tqdm.tqdm(total=len(counters) - 1, desc="%s Counter Accumulation" % desc)
		#	def update_tqdm_callback(arg):
		#		task_tqdm.update()
		#		return arg
		#
		#	# Pair up counters to be summed
		#	half_len = len(counters) // 2
		#	unsummed_counters = counters[2 * half_len:]
		#	counter_pairs = list(zip(counters[:half_len], counters[half_len:2 * half_len]))
		#	while len(counter_pairs) > 0:
		#		# Sum pairs and generate a new list of counters to be summed
		#		counters = list(map(update_tqdm_callback, map_func(ClarityScore._counter_add, counter_pairs))) + \
		#				   unsummed_counters
		#		# Pair up counters to be summed
		#		half_len = len(counters) // 2
		#		unsummed_counters = counters[2 * half_len:]
		#		counter_pairs = list(zip(counters[:half_len], counters[half_len:2 * half_len]))
		#	return unsummed_counters[0]

		# Get term count for each document in collection
		tokenizing_desc = "%s Token Counting" % desc
		merging_desc = "%s Counter Accumulation" % desc
		if workers > 1:
			with multiprocessing.Pool(workers) as pool:
				counters = list(tqdm.tqdm(pool.imap(ClarityScore._parse_handyman_langmodel_lines,
							ClarityScore._iter_handyman_langmodel_output(tokencounts_filepath)),
							desc=tokenizing_desc))

			with multiprocessing.Pool(workers) as pool:
				counter = ClarityScore._merge(ClarityScore._counter_add, pool.imap, counters, desc=merging_desc)
		else:
			counters = [ClarityScore._parse_handyman_langmodel_lines(lines) \
					for lines in tqdm.tqdm(ClarityScore._iter_handyman_langmodel_output(tokencounts_filepath), \
							       desc=tokenizing_desc)]
			counter = ClarityScore._merge(ClarityScore._counter_add, map, counters, desc=merging_desc)

		# Generate probability distribution from counts by dividing counts
		# by total number of terms
		term_count = sum(counter.values())
		for term in tqdm.tqdm(counter.keys(), desc="%s Token Count Normalization" % desc):
			counter[term] = Fraction(counter[term], term_count)
		assert(sum(counter.values()) == 1)
		return counter


	def __call__(self, query, doc_langmodel_filepaths_file):
		# Generate lanuage model for each document listed in docfile
		document_language_models = []
		with open(doc_langmodel_filepaths_file) as fh, multiprocessing.Pool(self.workers) as pool:
			for lm in pool.map(ClarityScore.gen_language_model, \
					map(lambda x: (x.strip(), 1, os.path.basename(x.strip()) + " LM"), fh)):
				document_language_models.append(lm)
		#print("document_language_models=", document_language_models)
		sys.stdout.flush()
		sys.stderr.flush()

		# Stem query
		# Query is assumed to be in format: "term0","term1",...,"termN"
		query = query.strip()
		print(query)
		assert(len(query) > 0 and query[0] == '"' and query[-1] == '"')

		# Extract terms from query
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
		print("stemmed_query_terms=", query_terms)

		# Remove emptry string terms
		query_terms = [term for term in query_terms if len(term) > 0]
		print("query_terms=", query_terms)

		# Precompute some values used for computing each of P(w|Q)
		while True:
			try:
				values_for_P_of_w_given_Q = self._values_for_P_of_w_given_Q(query_terms, document_language_models)
				break
			except BadTermError as e:
				idx = query_terms.index(e.term)
				print("Bad Term '%s'!" % e.term)
				if ' ' in e.term:
					print("  Splitting spaces and retrying")
					query_terms.pop(idx)
					for subterm in reversed(e.term.split(' ')):
						query_terms.insert(idx, subterm)
				else:
					print("  Removing from query")
					query_terms.pop(idx)
				print("Updated Query: %s" % ','.join(query_terms))

		#https://stackoverflow.com/a/312464/6573510
		def chunks(lst, n):
			rval = []
			"""Yield successive n-sized chunks from lst."""
			for i in range(0, len(lst), n):
				#yield lst[i:i + n]
				rval.append(lst[i:i + n])
			return rval

		# Compute clarity score term by term
		#   clarity score = \sum_{w \in V} P(w|Q) log2( P(w|Q) / P_{coll}(w) )
		#     w: term, V: entire vocabulary of collection, Q: query,
		#     P(w|Q): query language model function,
		#     P_{coll}(w): collection language model function,
		with multiprocessing.Pool(self.workers) as pool:
			term_contributions = \
					list(tqdm.tqdm(pool.imap(ClarityScore._compute_terms_contribution,
								 map(lambda ws: (ws, document_language_models, values_for_P_of_w_given_Q,
										 self.collection_language_model, self.smoothing_constant),
								     chunks(list(self.collection_language_model.keys()),
									    CLARITY_SCORE_TERM_CONTRIB_CHUNK_SIZE))),
						       desc="Computing Clarity Score Term Contrib",
						       total=int(math.ceil(len(self.collection_language_model) / CLARITY_SCORE_TERM_CONTRIB_CHUNK_SIZE))))
		term_contributions = [item for sublist in term_contributions for item in sublist]

		term_info = {}
		with multiprocessing.Pool(self.workers) as pool:
			clarity_score, term_info = ClarityScore._merge(ClarityScore._merge_term_contributions, pool.imap,
					term_contributions, desc="Merging Clarity Score Term Contrib")
		sys.stdout.flush()
		sys.stderr.flush()
		return clarity_score, term_info, query_terms

	@staticmethod
	def _merge_term_contributions(argspack):
		contrib1, contrib2 = argspack
		partial_clarity_score = contrib1[0] + contrib2[0]

		partial_info = dict()
		partial_info.update(contrib1[1])
		partial_info.update(contrib2[1])

		return (partial_clarity_score, partial_info)


	def _values_for_P_of_w_given_Q(self, query_terms, document_language_models):
		# Precompute [P(Q|D) P(D)]s for each document for computation of P(D|Q)
		# 1) Compute priors, P(D)s
		priors = [Fraction(1, len(document_language_models))] * len(document_language_models)
		#print("priors -", priors)
		assert(sum(priors) == 1)

		# 2) Compute P(Q|D)s
		P_of_Q_given_D_list = [ClarityScore._P_of_Q_given_D(query_terms, doc_lm,
				self.collection_language_model, self.smoothing_constant) \
					for doc_lm in document_language_models]
		#print("P(Q|D) list -", P_of_Q_given_D_list)

		# 3) Compute [P(Q|D) P(D)]s
		P_of_D_given_Q_list = [v1 * v2 for (v1, v2) in zip(P_of_Q_given_D_list, priors)]
		#print("P(Q|D) P(D) list -", P_of_D_given_Q_list)

		# 4) Compute sum of [P(Q|D) P(D)]s
		P_of_D_given_Q_sum = sum(P_of_D_given_Q_list)
		#print("[P(Q|D) P(D)]s sum -", P_of_D_given_Q_sum)

		return (P_of_D_given_Q_list, P_of_D_given_Q_sum)

	@staticmethod
	def _compute_terms_contribution(argspack):
		ws = argspack[0]
		return [ClarityScore._compute_term_contribution(w, *argspack[1:]) for w in ws]

	@staticmethod
	def _compute_term_contribution(w, document_language_models, values_for_P_of_w_given_Q, collection_language_model, smoothing_constant):
		P_of_w_given_Q = ClarityScore._P_of_w_given_Q(w, document_language_models, *values_for_P_of_w_given_Q, \
				collection_language_model, smoothing_constant)
		P_coll_of_w = ClarityScore._term_frequency(w, collection_language_model, \
				collection_language_model, smoothing_constant)

		# At this point in the code, all values used to compute the term contribution should be Python Fractions
		assert(isinstance(P_of_w_given_Q, Fraction) and isinstance(P_coll_of_w, Fraction))
		# Because of the operations performed, the term contribution will have type float

		term_contribution = P_of_w_given_Q * math.log2(P_of_w_given_Q / P_coll_of_w)

		return (term_contribution, {w: (term_contribution, math.log2(P_of_w_given_Q), math.log2(P_coll_of_w))})
		#term_info[w] = (term_contribution, math.log2(P_of_w_given_Q), math.log2(P_coll_of_w))
		#clarity_score += term_contribution

		#print("Clarity Score Term \"%s\" Contribution:" %(w))
		#print("    log2(P(\"%s\"|Q))=" % w, math.log2(P_of_w_given_Q))
		#print("    log2(P_{coll}(%s))=" % w, math.log2(P_coll_of_w))
		#print("  P(\"%s\"|Q) log2( P(\"%s\"|Q) / P_{coll}(%s)) =" % (w, w, w), term_contribution)
		#print("")


	@staticmethod
	def _P_of_Q_given_D(query_terms, document_language_model, collection_language_model, smoothing_constant):
		# P(Q|D) = \prod_{w \in Q} P(w|D)
		#print("Computing P(Q|D)")
		P_of_Q_given_D = Fraction(1.)
		for term in query_terms:
			term_frequency = ClarityScore._term_frequency(term, document_language_model, \
					collection_language_model, smoothing_constant)
			if 0 == term_frequency:
				raise BadTermError(term)
			#print("  tf(%s) =" % term, term_frequency)
			P_of_Q_given_D *= term_frequency
			#print("  P_of_Q_given_D=", P_of_Q_given_D)
		return P_of_Q_given_D


	@staticmethod
	def _P_of_w_given_Q(w, document_language_models, P_of_D_given_Q_list, P_of_D_given_Q_sum, \
			    collection_language_model, smoothing_constant):
		#print("_P_of_w_given_Q:", '"'+w+'"', query_terms, document_language_models)

		# Estimate query language model function at term w, P(w|Q), term by term
		#   P(w|Q) = \sum_{D \in R} P(w|D) P(D|Q)
		#     w: some term, Q: query string,
		#     D: document or model estimated from the corresponding single document,
		#     R: set of containing at least one query term (documents retrieved),
		#     P(w|D): frequency of term w in document D,
		#     P(D|Q): probability of document D given query Q
		probability = Fraction(0)
		for i, (P_of_D_given_Q, document_language_model) in enumerate(zip(P_of_D_given_Q_list, document_language_models)):
			#print("Document %d P(\"%s\"|Q)" % (i, w))

			# Compute P(w|D)
			P_of_w_given_D = ClarityScore._term_frequency(w, document_language_model, \
					collection_language_model, smoothing_constant)
			#print("    P(\"%s\"|D_%d)=" % (w, i), P_of_w_given_D)

			# Compute P(D|Q)
			#   P(D|Q) is derived from P(Q|D) using Bayesian inverse with
			#   uniform prior probabilities for documents in R and zero prior
			#   for documents that contain no query terms
			#   P(D|Q) =          P(Q|D) P(D)
			#            -----------------------------
			#            \sum_{D' \in R} P(Q|D') P(D')
			#
			# see Estimating the Query Difficulty for Information Retrieval
			P_of_D_given_Q = P_of_D_given_Q / P_of_D_given_Q_sum
			#print("    P(D_%d|Q)=" % i, P_of_D_given_Q)

			term_contribution = P_of_w_given_D * P_of_D_given_Q
			#print("  P(\"%s\"|Q)=" % w, term_contribution)

			probability += term_contribution
		return probability


	@staticmethod
	def _term_frequency(w, language_model, collection_language_model, smoothing_constant):
		# Compute relative frequeny of term w in language model.
		# Result is smoothed if langauge model is not the collection
		# language model.
		frequency = language_model[w]
		if language_model is not collection_language_model:
			# P(w|D) = \lambda P_{ml}(w|D) + (1 - \lambda) P_{coll}(w)
			# \lambda = 0.6
			frequency = smoothing_constant * frequency + \
						(1- smoothing_constant) * collection_language_model[w]
		return frequency

	@staticmethod
	def _parse_docfile(filepath):
		with open(filepath) as fh:
			documents_filepaths = [line.strip() for line in fh]
		return documents_filepaths


	def save(self, cache_filepath):
		obj_dict = {
			"collection_filepath": self.collection_filepath,
			"collection_language_model": base64.b64encode(pickle.dumps(self.collection_language_model)),
			"smoothing_constant": self.smoothing_constant
		}
		with open(cache_filepath, 'wb') as fh:
			pickle.dump(obj_dict, fh)

	@staticmethod
	def restore(cache_filepath):
		with open(cache_filepath, 'rb') as fh:
			obj_dict = pickle.load(fh)
		obj_dict["collection_language_model"] = pickle.loads(base64.b64decode(obj_dict["collection_language_model"]))

		cs = ClarityScore(None, smoothing_constant=obj_dict["smoothing_constant"])
		cs.collection_filepath = obj_dict["collection_filepath"]
		cs.collection_language_model = obj_dict["collection_language_model"]
		return cs


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('-q', '--query', required=False)

	group = parser.add_mutually_exclusive_group(required=True)
	group.add_argument('-L', '--collection_langmodel_filepath',
			   help='File should contain output from wumpus\' handyman operation BUILD_LM')
	group.add_argument('-A', '--cache_input_filepath')

	parser.add_argument('-r', '--retrieval_langmodel_filepaths_file', required=False,
			    help='File should contain output from wumpus\' handyman operation BUILD_LM')
	parser.add_argument('-o', '--cache_output_filepath', required=False)
	parser.add_argument('--smoothing', default=0.6, type=float, required=False)
	parser.add_argument('--workers', default=8, type=int, required=False)
	parser.add_argument('--max_top_terms', default=5, type=int, required=False)
	parser.add_argument('--augmented_queries', default=0, type=int, required=False)
	args = parser.parse_args()

	if args.collection_langmodel_filepath is not None:
		cs = ClarityScore(args.collection_langmodel_filepath, smoothing_constant=args.smoothing, workers=args.workers)
	else:
		cs = ClarityScore.restore(args.cache_input_filepath)
		cs.workers = args.workers

	if args.cache_output_filepath is not None:
		cs.save(args.cache_output_filepath)

	if args.retrieval_langmodel_filepaths_file is not None:
		clarity_score, term_info, stemmed_query_terms = cs(args.query, args.retrieval_langmodel_filepaths_file)
		print("Clarity Score=", clarity_score)
		print("Top Contributing Terms:")
		sorted_contributing_terms = sorted(term_info.items(), key=lambda v: -v[1][0])
		top_contributing_terms = sorted_contributing_terms[:min(args.max_top_terms, len(term_info))]
		for term, (term_contribution, log2_P_of_w_given_Q, log2_P_coll_of_w) in top_contributing_terms:
			print("  %s:" % str(term), term_contribution)

		if args.augmented_queries > 0:
			augmented_query = args.query
			added_terms = 0

			for term, _ in sorted_contributing_terms:
				if term not in stemmed_query_terms:
					augmented_query += ',"%s"' % term
					added_terms += 1
					print("Augmented Query %d=%s" % (added_terms, augmented_query))
					if added_terms == args.augmented_queries:
						break

if __name__ == "__main__":
	main()
