#!/usr/bin/env python

import nltk
import math, base64, pickle
import numpy as np
import multiprocessing
from collections import Counter
from nltk.stem.porter import PorterStemmer
import argparse
import tqdm


EPSILON = 1e-10

class ClarityScore(object):
	#https://www.nltk.org/_modules/nltk/tokenize.html#word_tokenize
	__internal_tokenize = nltk.tokenize.word_tokenize
	__internal_stemmer = PorterStemmer()


	def __init__(self, collection_filepath, smoothing_constant=0.6, workers=8):
		self.smoothing_constant = smoothing_constant
		self.workers = workers

		# I only really want restore passing None to constructor
		if collection_filepath is not None:
			self.collection_filepath = collection_filepath
			self.collection_language_model = \
					self.gen_language_model(ClarityScore._parse_docfile(self.collection_filepath))
		else:
			self.collection_filepath = None
			self.collection_language_model = None


	@staticmethod
	def _tokenize(text, stem_tokens=True):
		tokens = ClarityScore.__internal_tokenize(text)
		if stem_tokens:
			# Stem tokens, checking for and removing empty strings just in case
			# Let's do this in pace to reduce memory working set size
			next_slot_idx = 0
			for cur_elem_idx in range(len(tokens)):
				stemmed_token = ClarityScore.__internal_stemmer.stem(tokens[cur_elem_idx])
				if '' != stemmed_token:
					tokens[next_slot_idx] = stemmed_token
					next_slot_idx += 1
			if len(tokens) > next_slot_idx:
				print("Removed %d empty-strings generated from stemming" % \
					  (len(tokens) - next_slot_idx))
			tokens = tokens[:next_slot_idx]
		#unigrams = nltk.ngrams(tokens, 1)
		return tokens

	# Get term counts for a specific document
	@staticmethod
	def _file_token_counts(filepath, stem):
		with open(filepath, errors='backslashreplace') as fh:
			try:
				unigrams = ClarityScore._tokenize(fh.read(), stem)
			except Exception as e:
				print("Error encountered when parsing %s" % filepath)
				raise e
		print("Finished tokenizing %s" % filepath)
		return Counter(unigrams)

	def gen_language_model(self, document_filepaths, stem_tokens=True):
		#print("gen_language_model:", document_filepaths, stem_tokens)

		# Get term counts for all documents specified
		workers = min(self.workers, len(document_filepaths))
		if workers > 1:
			with multiprocessing.Pool(workers) as pool:
				document_counters = pool.starmap(ClarityScore._file_token_counts, \
						[(d, stem_tokens) for d in document_filepaths])
		else:
			document_counters = [ClarityScore._file_token_counts(f, stem_tokens) \
					for f in tqdm.tqdm(document_filepaths)]

		# Sum term counts from each counter
		counter = Counter()
		while len(document_counters) > 0:
			counter += document_counters.pop()

		# Generate probability distribution from counts by dividing counts
		# by total number of terms
		term_count = float(sum(counter.values()))
		for term in counter.keys():
			counter[term] /= term_count
		assert(1. - EPSILON <= sum(counter.values()) <= 1. + EPSILON)
		return counter


	def __call__(self, query, docfile_filepath):
		#print("__call__:", '"'+query+'"', docfile_filepath)

		# Generate lanuage model for each document listed in docfile
		document_language_models = []
		for filepath in ClarityScore._parse_docfile(docfile_filepath):
			document_language_models.append(self.gen_language_model([filepath]))
		#print("document_language_models=", document_language_models)

		# Tokenize query
		query_terms = self._tokenize(query)
		#print("query_terms=", query_terms)

		# Precompute some values used for computing each of P(w|Q)
		values_for_P_of_w_given_Q = self._values_for_P_of_w_given_Q(query_terms, document_language_models)

		# Compute clarity score term by term
		#   clarity score = \sum_{w \in V} P(w|Q) log2( P(w|Q) / P_{coll}(w) )
		#     w: term, V: entire vocabulary of collection, Q: query,
		#     P(w|Q): query language model function,
		#     P_{coll}(w): collection language model function,
		clarity_score = 0.
		term_info = {}
		for w in tqdm.tqdm(self.collection_language_model.keys()):
			P_of_w_given_Q = self._P_of_w_given_Q(w, document_language_models, *values_for_P_of_w_given_Q)
			P_coll_of_w = self._term_frequency(w, self.collection_language_model)
			term_contribution = P_of_w_given_Q * math.log2(P_of_w_given_Q / P_coll_of_w)

			term_info[w] = (term_contribution, math.log2(P_of_w_given_Q), math.log2(P_coll_of_w))
			clarity_score += term_contribution

			#print("Clarity Score Term \"%s\" Contribution:" %(w))
			#print("    log2(P(\"%s\"|Q))=%f" % (w, math.log2(P_of_w_given_Q)))
			#print("    log2(P_{coll}(%s))=%f" % (w, math.log2(P_coll_of_w)))
			#print("  P(\"%s\"|Q) log2( P(\"%s\"|Q) / P_{coll}(%s)) =%f" % (w, w, w, term_contribution))
			#print("")
		return clarity_score, term_info


	def _values_for_P_of_w_given_Q(self, query_terms, document_language_models):
		# Precompute [P(Q|D) P(D)]s for each document for computation of P(D|Q)
		# 1) Compute priors, P(D)s
		priors = np.array([0.] * len(document_language_models))
		for i, lm in enumerate(document_language_models):
			for term in query_terms:
				if term in lm:
					priors[i] = 1.
					break
		priors = priors / np.sum(priors)
		#print("priors -", priors)
		assert(1. - EPSILON <= np.sum(priors) <= 1 + EPSILON or 0 == np.sum(priors)) 

		# 2) Compute P(Q|D)s
		P_of_Q_given_D_list = [self._P_of_Q_given_D(query_terms, doc_lm) \
							   for doc_lm in document_language_models]
		#print("P(Q|D) list -", P_of_Q_given_D_list)

		# 3) Compute [P(Q|D) P(D)]s
		P_of_D_given_Q_list = [v1 * v2 for (v1, v2) in zip(P_of_Q_given_D_list, priors)]
		#print("P(Q|D) P(D) list -", P_of_D_given_Q_list)

		# 4) Compute sum of [P(Q|D) P(D)]s
		P_of_D_given_Q_sum = sum(P_of_D_given_Q_list)
		#print("[P(Q|D) P(D)]s sum -", P_of_D_given_Q_sum)

		return (P_of_D_given_Q_list, P_of_D_given_Q_sum)


	def _P_of_Q_given_D(self, query_terms, document_language_model):
		# P(Q|D) = \prod_{w \in Q} P(w|D)
		P_of_Q_given_D = 1.
		for term in query_terms:
			P_of_Q_given_D *= self._term_frequency(term, document_language_model)
		return P_of_Q_given_D


	def _P_of_w_given_Q(self, w, document_language_models, P_of_D_given_Q_list, P_of_D_given_Q_sum):
		#print("_P_of_w_given_Q:", '"'+w+'"', query_terms, document_language_models)

		# Estimate query language model function at term w, P(w|Q), term by term
		#   P(w|Q) = \sum_{D \in R} P(w|D) P(D|Q)
		#     w: some term, Q: query string,
		#     D: document or model estimated from the corresponding single document,
		#     R: set of containing at least one query term (documents retrieved),
		#     P(w|D): frequency of term w in document D,
		#     P(D|Q): probability of document D given query Q
		probability = 0.
		for i, (P_of_D_given_Q, document_language_model) in enumerate(zip(P_of_D_given_Q_list, document_language_models)):
			#print("Document %d P(\"%s\"|Q)" % (i, w))

			# Compute P(w|D)
			P_of_w_given_D = self._term_frequency(w, document_language_model)
			#print("    P(\"%s\"|D_%d)=%f" % (w, i, P_of_w_given_D))

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
			#print("    P(D_%d|Q)=%f" % (i, P_of_D_given_Q))

			term_contribution = P_of_w_given_D * P_of_D_given_Q
			#print("  P(\"%s\"|Q)=%f" % (w, term_contribution))

			probability += term_contribution
		return probability


	def _term_frequency(self, w, language_model):
		# Compute relative frequeny of term w in language model.
		# Result is smoothed if langauge model is not the collection
		# language model.
		frequency = language_model[w]
		if language_model is not self.collection_language_model:
			# P(w|D) = \lambda P_{ml}(w|D) + (1 - \lambda) P_{coll}(w)
			# \lambda = 0.6
			frequency = self.smoothing_constant * frequency + \
						(1. - self.smoothing_constant) * self.collection_language_model[w]
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
	parser.add_argument('-q', '--query', required=True)

	group = parser.add_mutually_exclusive_group(required=True)
	group.add_argument('-L', '--collection_filepath')
	group.add_argument('-A', '--cache_input_filepath')

	parser.add_argument('-r', '--retrieval_filepath')
	parser.add_argument('-o', '--cache_output_filepath', required=False)
	parser.add_argument('--smoothing', default=0.6, type=float, required=False)
	parser.add_argument('--workers', default=8, type=int, required=False)
	parser.add_argument('--max_top_terms', default=5, type=int, required=False)
	args = parser.parse_args()

	if args.collection_filepath is not None:
		cs = ClarityScore(args.collection_filepath, smoothing_constant=args.smoothing, workers=args.workers)
	else:
		cs = ClarityScore.restore(args.cache_input_filepath)
		cs.workers = args.workers

	if args.cache_output_filepath is not None:
		cs.save(args.cache_output_filepath)

	clarity_score, term_info = cs(args.query, args.retrieval_filepath)
	print("Clarity Score: %f" % clarity_score)
	print("Top Contributing Terms:")
	top_contributing_terms = sorted(term_info.items(), key=lambda v: -v[1][0])[:min(args.max_top_terms, len(term_info))]
	for term, (term_contribution, log2_P_of_w_given_Q, log2_P_coll_of_w) in top_contributing_terms:
		print("  %s: %f" % (str(term), term_contribution))


if __name__ == "__main__":
	main()
