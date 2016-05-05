#!/usr/bin/env bash
#
# Get training, test, etc. data for EmpiriST 2016
# * the Tiger Corpus

set -e
. 00_get_data

#
### Tiger Corpus
#
# *important* http://www.ims.uni-stuttgart.de/forschung/ressourcen/korpora/TIGERCorpus/license/index.html
# http://www.ims.uni-stuttgart.de/forschung/ressourcen/korpora/TIGERCorpus/download/tigercorpus-2.2.conll09.tar.gz
#
# * download tigercorpus-2.2, conll format 
# * extract vertical file from conll format, i.e. lines of the form:
#   token\tlemma\tPOS
# * introduce <s> </s> marks
# * subst PROAV with PAV and NNE with NE
download tmp/tigercorpus-2.2.conll09.tar.gz 'http://www.ims.uni-stuttgart.de/forschung/ressourcen/korpora/TIGERCorpus/download/tigercorpus-2.2.conll09.tar.gz'
[ -e tiger_release_aug07.corrected.16012013-empirist.vrt.bz2 ] \
    || tar -xOzf tmp/tigercorpus-2.2.conll09.tar.gz tiger_release_aug07.corrected.16012013.conll09 \
    | cut -f2,3,5 \
    | sed -e '1{s#\(.*\)#<s>\n\1#;}' -e '${s#^$#</s>#;}' -e 's#^$#</s>\n<s>#' \
    | sed -e 's/PROAV$/PAV/' -e 's/NNE$/NE/' \
    | bzip2 -c -9 \
    > tiger_release_aug07.corrected.16012013-empirist.vrt.bz2
