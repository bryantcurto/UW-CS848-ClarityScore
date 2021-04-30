#!/bin/bash

DISK1_OUTPUT=disk1-filelist.txt
DISK2_OUTPUT=disk2-filelist.txt
DISK4_OUTPUT=disk4-filelist.txt
DISK5_OUTPUT=disk5-filelist.txt


rm -f "$DISK1_OUTPUT"
for dirpath in $(find '/tuna1/collections/newswire/disk12_uncompress/disk12/' -mindepth 1 -maxdepth 1 -type d -name 'disk1*'); do
    find "$dirpath" -type f >> "$DISK1_OUTPUT"
done

rm -f "$DISK2_OUTPUT"
for dirpath in $(find '/tuna1/collections/newswire/disk12_uncompress/disk12/' -mindepth 1 -maxdepth 1 -type d -name 'disk2*'); do
    find "$dirpath" -type f >> "$DISK2_OUTPUT"
done

### Code for copying and cleaning up disks 4 & 5 ###
#cp -r /tuna1/collections/newswire/disk45 ./
#chmod +w -R disk45
#
#for f in $(find disk45 -type f -name '*.z'); do uncompress --stdout "$f" > "$(echo "$f" | sed 's/.z$//')"; rm "$f"; done
#for f in $(find disk45/disk4/fr94 -type f -regex '.*\.[0-9][0-9]*z'); do uncompress --stdout "$f" > "$(echo "$f" | sed 's/z$//')"; rm "$f"; done
#
#mkdir -p extra/dtds/disk45/
#mv disk45/disk4/dtds extra/dtds/disk45/disk4
#mv disk45/disk5/dtds extra/dtds/disk45/disk5
#
#mkdir -p extra/aux/disk45/disk4
#mv disk45/disk4/fr94/aux extra/aux/disk45/disk4/fr94
#
#for f in $(find disk45 -type f -name 'read*'); do mkdir -p "extra/read/$(dirname $f)"; mv "$f" "extra/read/$f"; done
#
#chmod -w -R disk45
#chmod -w -R extra

find "$(pwd)/disk45/disk4" -type f > "$DISK4_OUTPUT"
find "$(pwd)/disk45/disk5" -type f > "$DISK5_OUTPUT"
