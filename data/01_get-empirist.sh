#!/usr/bin/env bash
#
# Get training, test, etc. data for EmpiriST 2016
# * the EmpiriST 2016 data

set -e
. 00_get_data

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
