#!/bin/bash


WUMPUS="/u1/bcurto/wumpus/bin/wumpus --config=/u1/bcurto/wumpus/wumpus.cfg"
HANDYMAN="/u1/bcurto/wumpus/bin/handyman"
WUMPUS_WORKSPACE_DIR="./wumpus-workspaces"

sugar="$(date +%s-%N)"
TMP1_FILE="tmp1-$sugar"
TMP2_FILE="tmp2-$sugar"

if [[ 5 != $# ]]; then
	echo "USAGE: $0 trecN WUMPUS_OP QUERY_FILES_GLOB OUTPUT_DIR DEBUG_STR"
	echo "Example: trec5 '@bm25[count=600]' '$(pwd)/queries/trec5-queries/*_desc.txt' $(pwd)/retrievals/ 'bm25-count600'"
	exit -1
fi

trec="$1"
wumpus_op="$2"
query_files_glob="$3"
output_dir="$4"
debug_str="$5"


trec_output_dir="$output_dir"/"$trec"							# base directory of all output
wumpus_output_dir="$trec_output_dir"/wumpus						# holds output of retrievals using wumpus
runfile_output_dir="$trec_output_dir"/runfile						# holds runfile generated using wumpus output
doc_output_dir="$trec_output_dir"/doc							# holds documents retrieved
doclangmodel_output_dir="$trec_output_dir"/doclangmodel					# holds document languages of documents retrieved
retrieved_doclangmodel_paths_output_dir="$trec_output_dir"/retrieved-doclangmodel-paths	# for each query, holds file containing paths to langs of retrieved docs
mkdir -p "$trec_output_dir" "$wumpus_output_dir" "$doclangmodel_output_dir" "$doc_output_dir" "$runfile_output_dir" "$retrieved_doclangmodel_paths_output_dir"

doc_cache_filepath="$trec_output_dir"/doc-cache	# cache for storing mapping doc content range (from wumpus) to document id
touch "$doc_cache_filepath"

wumpus_database_dir="$WUMPUS_WORKSPACE_DIR"/"$trec"-workspace/database	
for filepath in $(ls $query_files_glob); do
	echo "Processing $filepath..."

	# Get filename without extension: expected format is [TOPIC_ID]_[TOPIC_SRC(e.g., title)]
	query_id="$(basename $filepath | rev | cut -d '.' -f 2 | rev)"
	extended_query_id="$debug_str"_"$trec"_"$query_id"
	topicnum="$(echo "$query_id" | cut -d '_' -f 1)"

	# Figure out where we're writing output of wumpus rank query command
	wumpus_output_filepath="$wumpus_output_dir"/"$extended_query_id"_wumpus-output.txt

	# Execute wumpus rank query command
	wumpus_query="$wumpus_op"'[docid] "<doc>".."</doc>" by '"$(cat "$filepath")"
	echo "$wumpus_query" > "$wumpus_output_filepath" # add query to top of file for reference
	if ! echo "$wumpus_query" | $WUMPUS --DIRECTORY="$wumpus_database_dir" 2> /dev/null >> "$wumpus_output_filepath"; then
		echo "Something went wrong..."
		exit -1
	fi

	# Path to runfile that can be used with trec_eval/dyn_eval
	runfile_output_filepath="$runfile_output_dir"/"$extended_query_id"_runfile
	rm -f "$runfile_output_filepath"

	# Path holding paths to language models of retrieved documents
	retrieved_doclangmodel_paths_filepath="$retrieved_doclangmodel_paths_output_dir"/"$extended_query_id"_retrieved-doclangmodel-paths
	rm -f "$retrieved_doclangmodel_paths_filepath"

	# Iterate through lines of wumpus output containing retrieved documents
	# Creating documents and document laguage models of retrieved docs.
	# Also, generate runfile for query as well as document paths to document language models.
	rank=1
	while IFS= read -r line; do
		# Process retrieved documents
		echo "$line"

		# Get score of retrieved document
		score="$(echo "$line" | cut -d ' ' -f 2)"

		# Retrieve contents of retrieved document
		doc_filepath_template="$doc_output_dir"/"%s_doc.txt"
		docrange="$(echo "$line" | cut -d ' ' -f 3-4)"
		if [[ $(grep "^$docrange#" "$doc_cache_filepath" | wc -l) != 1 ]]; then
			echo "Extracting document (range=$docrange)"

			echo "@get $docrange" | $WUMPUS --DIRECTORY="$wumpus_database_dir" 2> /dev/null | tail -n +2 | head -n -1 > "$TMP1_FILE"

			docno="$(echo "$line" | cut -d ' ' -f 5 | sed 's/"//g')"
			# Extract document number and validate that it's sane
			##docno="$(python -c "from xml.etree import cElementTree as ET; print(ET.parse('$TMP1_FILE').find('DOCNO').text.strip())")"
			#docno="$(cat "$TMP1_FILE" | \
			#		sed 's/\(<DOCNO>\)/\1\n/' | sed 's/\(<\/DOCNO>\)/\n\1/' | \
			#		sed '0,/<DOCNO>/d' | tac | sed '0,/<\/DOCNO>/d' | \
			#		tr -d ' ' | tr -d '\n')"
			#if (( "$(echo "$docno" | grep -vE '^[-_a-zA-Z0-9]+$' | wc -l)" != 0 )); then
			#	echo "ERROR: failed to extract document number from "
			#	exit -1
			#fi

			# Store mapping from docrange to docno in cache
			echo -e "$docrange#$docno" >> "$doc_cache_filepath"

			# Rename document file to file with name contain doc number
			doc_filepath="$(printf "$doc_filepath_template" "$docno")"
			mv "$TMP1_FILE" "$doc_filepath"
		else
			echo "Reading document (range=$docrange) from cache"

			docno="$(grep "^$docrange#" "$doc_cache_filepath" | cut -d '#' -f 2)"
			doc_filepath="$(printf "$doc_filepath_template" "$docno")"
		fi

		# Generate language model from document if it doesn't alreay exist
		doclangmodel_filepath="$doclangmodel_output_dir"/"$docno"_doclangmodel.txt
		if [[ ! -f "$doclangmodel_filepath" ]]; then
			echo "$doc_filepath" > "$TMP2_FILE"
			$HANDYMAN BUILD_LM "$TMP2_FILE" "$doclangmodel_filepath"
		fi

		# Add entry to runfile
		echo "$topicnum q0 $docno $rank $score curtorun" >> "$runfile_output_filepath"
		rank=$(($rank + 1))

		# Add filepath of document language model to ongoing list
		echo "$doclangmodel_filepath" >> "$retrieved_doclangmodel_paths_filepath"

		# Cleanup
		rm -f "$TMP1_FILE" "$TMP2_FILE"
	done <<< "$(cat "$wumpus_output_filepath" | tail -n +3 | head -n -1)"
	echo
	echo
done

