#!/bin/bash

DISK_BASE_DIR=../../trec-disks
DISK2_FILEPATH="$DISK_BASE_DIR"/disk2-filelist.txt
DISK4_FILEPATH="$DISK_BASE_DIR"/disk4-filelist.txt
DISK5_FILEPATH="$DISK_BASE_DIR"/disk5-filelist.txt

#document set is TIPSTER disk 2 and TREC disk 4
TREC5_COLLECTION_FILEPATH=trec5-collection-filelist.txt
cat "$DISK2_FILEPATH" "$DISK4_FILEPATH" > "$TREC5_COLLECTION_FILEPATH"
handyman BUILD_LM "$TREC5_COLLECTION_FILEPATH" trec5-collection-languagemodel.txt --count=5000000 |& tee trec5-handyman-output.txt

# document set is TREC disks 4&5
TREC6_COLLECTION_FILEPATH=trec6-collection-filelist.txt
cat "$DISK4_FILEPATH" "$DISK5_FILEPATH" > "$TREC6_COLLECTION_FILEPATH"
handyman BUILD_LM "$TREC6_COLLECTION_FILEPATH" trec6-collection-languagemodel.txt --count=5000000 |& tee trec6-handyman-output.txt

# document set is TREC disks 4&5 minus the Congressional Record
TREC7_COLLECTION_FILEPATH=trec7-collection-filelist.txt
cat "$DISK4_FILEPATH" "$DISK5_FILEPATH" | grep -v '/disk4/cr/' > "$TREC7_COLLECTION_FILEPATH"
handyman BUILD_LM "$TREC7_COLLECTION_FILEPATH" trec7-collection-languagemodel.txt --count=5000000 |& tee trec7-handyman-output.txt

# document set is TREC disks 4&5 minus the Congressional Record
TREC8_COLLECTION_FILEPATH=trec8-collection-filelist.txt
cat "$DISK4_FILEPATH" "$DISK5_FILEPATH" | grep -v '/disk4/cr/' > "$TREC8_COLLECTION_FILEPATH"
#handyman BUILD_LM "$TREC8_COLLECTION_FILEPATH" trec8-collection-languagemodel.txt --count=5000000 |& tee trec8-handyman-output.txt
cp trec7-collection-languagemodel.txt trec8-collection-languagemodel.txt
