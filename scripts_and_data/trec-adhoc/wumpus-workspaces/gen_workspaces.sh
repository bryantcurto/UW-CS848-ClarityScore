#!/bin/bash

for i in $(seq 5 8); do
    dir=trec"$i"-workspace
    mkdir $dir
    pushd $dir
    cat ../../collections/trec"$i"-collection-filelist.txt | sed 's/^/@addfile /' | wumpus --config=/u1/bcurto/wumpus/wumpus.cfg
    popd
done
