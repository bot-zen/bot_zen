#!/usr/bin/env bash
#
# pre-process data to be fed to word2vec
# * de.wikipedia.org data
#

set -e

cat <(bzip2 -c -d ./tmp/de.wikipedia.org/*.tt.bz2) \
    > ./tmp/pre-w2v_de.wikipedia.org.txt
./_w2v_prep.sh < ./tmp/pre-w2v_de.wikipedia.org.txt > ./tmp/w2v_de.wikipedia.org.txt
