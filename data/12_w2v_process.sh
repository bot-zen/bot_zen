#!/usr/bin/env bash
#
#
#
#
#

set -e

[ -d ./tmp/word2vec.svn ] || svn co http://word2vec.googlecode.com/svn/trunk ./tmp/word2vec.svn
[ -x ./tmp/word2vec.svn/word2vec ] || (pushd ./tmp/word2vec.svn; make word2vec; popd)

DATA_EMPIRIST=w2v_empirist.txt
DATA_WIKIPEDIA=w2v_de.wikipedia.org.txt

THREADS=${THREADS:-4}

mkdir -p word2vec
./tmp/word2vec.svn/word2vec -train ./tmp/$DATA_EMPIRIST -output word2vec/empirist.bin -cbow 0 -size 500 -window 10 -hs 1 -negative 5 -min-count 3 -threads $THREADS -debug 2 -save-vocab ./word2vec/empirist-vocab.txt

./tmp/word2vec.svn/word2vec -train ./tmp/$DATA_WIKIPEDIA -output ./word2vec/bigdata.bin -cbow 1 -size 500 -window 10 -hs 1 -negative 3 -min-count 25 -threads $THREADS -debug 2 -save-vocab ./word2vec/bigdata-vocab.txt
