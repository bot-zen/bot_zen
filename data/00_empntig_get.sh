#!/usr/bin/env bash
#
# Get training, test, etc. data for EmpiriST 2016
# * the EmpiriST 2016 data
# * the Tiger Corpus

set -e

download() {
    DESTDIR=$(dirname "$1")
    DESTFN=$(basename "$1")
    mkdir -p "${DESTDIR}"

    [ -e "${DESTDIR}/${DESTFN}" ] || (wget --continue -O "${DESTDIR}/.${DESTFN}" "$2" && mv "${DESTDIR}/.${DESTFN}" "${DESTDIR}/${DESTFN}")
}


#
### EmpiriST
#
# cf. https://sites.google.com/site/empirist2015/home/shared-task-data
#
# * download the EmpiriST 2015 data
# * remove utf-8 BOMs (supposedly there are none...)
SUBDIR=empirist
mkdir -p ${SUBDIR}
for file in empirist_test_pos_cmc.zip empirist_test_pos_web.zip \
    empirist_test_tok_cmc.zip empirist_test_tok_web.zip \
    empirist_training_cmc.zip empirist_training_web.zip \
    empirist_trial_cmc.zip empirist_trial_web.zip; do
    
    download "${SUBDIR}"/"${file}" 'https://sites.google.com/site/empirist2015/home/shared-task-data/'"${file}"'/?attredirects=0&d=1'
done

pushd "${SUBDIR}"
unzip -tq "*.zip" && unzip "*.zip"
find ./ -name '*.txt' -exec sed -i -e '1{s/^\xef\xbb\xbf//;}' "{}" \;
popd


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
