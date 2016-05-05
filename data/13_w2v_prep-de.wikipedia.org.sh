#!/usr/bin/env bash
#
# pre-process data to be fed to word2vec
# * de.wikipedia.org data
#

set -e
. 10_w2v_prep_data

cat <(bzip2 -c -d ./tmp/de.wikipedia.org/*.tt.bz2) \
    > ./tmp/pre-w2v-${CLEAN_VERSION}_de.wikipedia.org.txt
clean < ./tmp/pre-w2v-${CLEAN_VERSION}_de.wikipedia.org.txt > ./tmp/w2v-${CLEAN_VERSION}_de.wikipedia.org.txt
