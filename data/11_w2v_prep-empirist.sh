#!/usr/bin/env bash
#
# pre-process data to be fed to word2vec
# * EmpiriST 2016 data
#

set -e
. 10_w2v_prep_data

cat <(cat ./empirist/empirist_training_{cmc,web}/tokenized/*.txt \
      | sed -e 's#^<posting.*#<s>#' -e 's#<article.*#<s>#' -e 's#^$#</s>#') \
    > ./tmp/pre-w2v-${CLEAN_VERSION}_empirist.txt
clean < ./tmp/pre-w2v-${CLEAN_VERSION}_empirist.txt > ./tmp/w2v-${CLEAN_VERSION}_empirist.txt
