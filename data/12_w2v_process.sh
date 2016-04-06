#!/usr/bin/env bash
#
#
#
#
#

set -e

[ -d ./tmp/word2vec.svn ] || svn co http://word2vec.googlecode.com/svn/trunk ./tmp/word2vec.svn
[ -x ./tmp/word2vec.svn/word2vec ] || (pushd ./tmp/word2vec.svn; make word2vec; popd)

mkdir -p word2vec
./tmp/word2vec.svn/word2vec -train ./tmp/empirist.txt -output word2vec/empirist.bin -cbow 0 -size 500 -window 10 -hs 1 -negative 5 -min-count 3 -threads 4 -debug 2 -save-vocab ./word2vec/empirist-vocab.txt

./tmp/word2vec.svn/word2vec -train ./tmp/bigdata.txt -output ./word2vec/bigdata.bin -cbow 1 -size 500 -window 10 -hs 1 -negative 3 -min-count 25 -threads 4 -debug 2 -save-vocab ./wordvec/bigdata-vocab.txt
