#!/usr/bin/env bash
#
#
#
#
#

set -e

[ -d ./tmp/word2vec.svn ] || svn co http://word2vec.googlecode.com/svn/trunk ./tmp/word2vec.svn
[ -x ./tmp/word2vec.svn/word2vec ] || (pushd ./tmp/word2vec.svn; make word2vec; popd)
