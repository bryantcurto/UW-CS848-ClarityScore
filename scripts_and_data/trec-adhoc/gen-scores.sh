#!/bin/bash

QRELS_DIR="./qrels"
COLLECTIONS_DIR="./collections"
QUERIES_DIR="./queries"
CLARITY_SCORE="../clarityscore/clarity_score.py"
COMPUTE_AP="./compute-ap.py"

if [[ $# != 3 ]]; then
	echo "USAGE: $0 trecN RETRIEVALS_DIR RUNFILE_FILENAME_GLOB"
	echo "  RETRIEVALS_DIR: e.g., retrievals/bm25_count600"
	echo "  RUNFILE_FILENAME_GLOB: e.g., *_title_runfile"
	exit -1
fi

trec="$1"
retrievals_dir="$2"
filename_glob="$3"

trec_dir="$retrievals_dir"/"$trec"
runfile_dir="$trec_dir"/runfile
retrieved_doclangmodel_dir="$trec_dir"/retrieved-doclangmodel-paths

trec_eval_output_dir="$trec_dir"/trec_eval
clarity_score_output_dir="$trec_dir"/clarity_score
mkdir -p "$trec_eval_output_dir" "$clarity_score_output_dir"
#mkdir -p "$clarity_score_output_dir"

collection_langmodel_cache_filepath="$trec_dir"/collection_langmodel_cache

# Process and cache collection language model
if ! $CLARITY_SCORE -L "$COLLECTIONS_DIR"/"$trec"-collection-languagemodel.txt -o "$collection_langmodel_cache_filepath" --workers 40; then
	echo "Failed to generate collection language model"
	return -1
fi

# Get path where scores will be stored
clarity_scores_filepath="$trec_dir"/clarity_scores
ap_scores_filepath="$trec_dir"/ap_scores
scores_filepath="$trec_dir"/scores
touch "$clarity_scores_filepath" "$ap_scores_filepath" "$scores_filepath"

for runfile_filepath in $(ls "$runfile_dir"/$filename_glob); do
	query_id="$(basename "$runfile_filepath" | cut -d '_' -f 3-4)"
	extended_query_id="$(basename "$runfile_filepath" | cut -d '_' -f -4)"
	echo "Computing scores for $query_id"

	# Continue if scores already computed
	if [[ $(grep "^$extended_query_id " "$scores_filepath" | wc -l) == 1 ]]; then
		echo "Scores for $query_id already computed"
		continue
	fi

	# Get path containing paths to language models of retrieved docs
	retrieved_doclangmodel_filepath="$retrieved_doclangmodel_dir"/"$extended_query_id"_retrieved-doclangmodel-paths
	if [[ ! -f "$retrieved_doclangmodel_filepath" ]]; then
		echo "ERROR: Retrieved Document Language Model not found for $query_id"
		continue
	fi

	# Get path to file containing query terms
	query_filepath="$QUERIES_DIR"/"$trec"-queries/"$query_id".txt
	if [[ ! -f "$query_filepath" ]]; then
		echo "ERROR: Something went wrong finding query file for $query_id"
		continue
	fi

	# Compute clarity score
	if [[ "$(grep "^$extended_query_id " "$clarity_scores_filepath" | wc -l)" == 1 ]]; then
		cs="$(grep "^$extended_query_id " "$clarity_scores_filepath" | cut -d ' ' -f 2)"
	else
		# Compute clarity score using top 500 retrieved documents
		clarity_score_output_filepath="$clarity_score_output_dir"/"$extended_query_id"_clarity-score.txt
		retrieved_doclangmodel_top500_filepath="$retrieved_doclangmodel_filepath"_top500
		cat "$retrieved_doclangmodel_filepath" | head -n 500 > "$retrieved_doclangmodel_top500_filepath"
		if ! $CLARITY_SCORE -q "$(cat "$query_filepath")" -A "$collection_langmodel_cache_filepath" \
				    -r "$retrieved_doclangmodel_top500_filepath" --max_top_terms 500 \
				    --augmented_queries 100 --workers 40 > "$clarity_score_output_filepath"; then
			echo "Clarity score computation failed for $query_id"
			continue
		fi
		rm "$retrieved_doclangmodel_top500_filepath"
		cs="$(cat "$clarity_score_output_filepath" | grep '^Clarity Score=' | cut -d '=' -f 2)"

		echo "$extended_query_id $cs" >> "$clarity_scores_filepath"
	fi

	# Compute average precision score
	if [[ "$(grep "^$extended_query_id " "$ap_scores_filepath" | wc -l)" == 1 ]]; then
		ap="$(grep "^$extended_query_id " "$ap_scores_filepath" | cut -d ' ' -f 2)"
	else
		# Run custom script to compute average precision since map is doesn't have the characteristics we want
		ap="$($COMPUTE_AP "$QRELS_DIR"/"$trec"-qrels "$runfile_filepath")"
		if [[ $? != 0 ]]; then
			echo "ERROR: Something went wrong with computing average precision for $query_id"
			continue
		fi

		#trec_eval_output_filepath="$trec_eval_output_dir"/"$extended_query_id"_trec-eval.txt
		#if ! trec_eval "$QRELS_DIR"/"$trec"-qrels "$runfile_filepath" > "$trec_eval_output_filepath"; then
		#	echo "ERROR: Something went wrong with trec_eval for $query_id"
		#	continue
		#fi
		#ap="$(grep '^map' "$trec_eval_output_filepath" | rev | cut -d $'\t' -f 1 | rev)"

		echo "$extended_query_id $ap" >> "$ap_scores_filepath"
	fi

	echo "$extended_query_id $ap $cs" >> "$scores_filepath"
done
