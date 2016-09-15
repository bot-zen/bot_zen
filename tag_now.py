from tagger import logger, network
from tagger.representation import postags
from tagger.utils import (
    load_tagged_files,
    all_tggd_flocs,
    all_gold_flocs,
    load_tiger_vrt_file,
    process_test_data_tagging)
# import numpy as np


logger.info('Starting experient.')
retres = []

postagstype_ibk = postags.PosTagsType(feature_type="ibk")
postagstype_ibk_used = postags.PosTagsType(feature_type="ibk_used")
postagstype_tiger_used = postags.PosTagsType(feature_type="1999_used")

toks, tags = load_tagged_files(all_tggd_flocs)
# toks_trial, tags_trial = load_tagged_files(all_trial_tggd_flocs)
toks_gold, tags_gold = load_tagged_files(all_gold_flocs)
toks_tig, tags_tig = load_tiger_vrt_file()
dropout = 0.1
nb_epoch = 20

batch_size = 20
model = network.build_nn(output_dim=postagstype_ibk.feature_length,
                         lstm_output_dim=1024, dropout=dropout)
# model.save_weights('/tmp/emptig_plain.hdf5', overwrite=True)

network.train_nn(model, toks, tags, batch_size=batch_size,
                 nb_epoch=nb_epoch, postagstype=postagstype_ibk)
# model.save_weights('/tmp/emptig_trained-0.hdf5', overwrite=True)
res_emp2 = network.eval_nn(model, toks_gold, tags_gold,
                           postagstype=postagstype_ibk)
logger.info(network.compact_res(res_emp2))
retres.append(('emp', res_emp2))


# ##

batch_size = 50
network.train_nn(model, toks_tig, tags_tig, batch_size=batch_size,
                 nb_epoch=nb_epoch, postagstype=postagstype_ibk)
# model.save_weights('/tmp/emptig_trained-1.hdf5', overwrite=True)
res_emptig = network.eval_nn(model, toks_gold, tags_gold,
                             postagstype=postagstype_ibk)
logger.info(network.compact_res(res_emptig))

# process_test_data_tagging(model, extension=".emptig",
#                           postagstype=postagstype_ibk)
retres.append(('emptig', res_emptig))

print(retres)
exit(0)
