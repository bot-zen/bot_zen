from tagger import logger, network
from tagger.representation import postags
from tagger.utils import (
    load_tagged_files,
    all_postwita_tggd_flocs,
    all_postwita_tst_flocs,
    # all_gold_flocs,
    load_tiger_vrt_file,
    process_test_data_tagging
)
# import numpy as np


logger.info('Starting experient.')
retres = []

# postagstype_ibk = postags.PosTagsType(feature_type="ibk")
# postagstype_ibk_used = postags.PosTagsType(feature_type="ibk_used")
# postagstype_tiger_used = postags.PosTagsType(feature_type="1999_used")
postagstype = postags.PosTagsType(feature_type="postwita")

toks, tags = load_tagged_files(all_postwita_tggd_flocs)
# toks_trial, tags_trial = load_tagged_files(all_trial_tggd_flocs)
# toks_gold, tags_gold = load_tagged_files(all_gold_flocs)
toks_tig, tags_tig = load_tiger_vrt_file(
    fileloc='../data/postwita/ud-treebanks-v1.3-it.vrt.bz2')
dropout = 0.1
nb_epoch = 20

batch_size = 20
model = network.build_nn(output_dim=postagstype.feature_length,
                         lstm_output_dim=1024, dropout=dropout)
# model.save_weights('/tmp/emptig_plain.hdf5', overwrite=True)

network.train_nn(model, toks, tags, batch_size=batch_size,
                 nb_epoch=nb_epoch, postagstype=postagstype)
model.save_weights('/tmp/emp_trained.hdf5', overwrite=True)
process_test_data_tagging(model, postagstype, all_postwita_tst_flocs,
                          extension=".emp")
# res_emp2 = network.eval_nn(model, toks_gold, tags_gold,
#                            postagstype=postagstype)
# logger.info(network.compact_res(res_emp2))
# retres.append(('emp', res_emp2))


# ##

batch_size = 50
network.train_nn(model, toks_tig, tags_tig, batch_size=batch_size,
                 nb_epoch=nb_epoch, postagstype=postagstype)
model.save_weights('/tmp/emptig_retrained-0.hdf5', overwrite=True)
process_test_data_tagging(model, postagstype, all_postwita_tst_flocs,
                          extension=".emptig")
# res_emptig = network.eval_nn(model, toks_gold, tags_gold,
#                              postagstype=postagstype)
# logger.info(network.compact_res(res_emptig))

# process_test_data_tagging(model, extension=".emptig",
#                           postagstype=postagstype_ibk)
# retres.append(('emptig', res_emptig))

# ## ###
batch_size = 20
network.train_nn(model, toks, tags, batch_size=batch_size,
                 nb_epoch=nb_epoch, postagstype=postagstype)
model.save_weights('/tmp/emptig_retrained-1.hdf5', overwrite=True)
process_test_data_tagging(model, postagstype, all_postwita_tst_flocs,
                          extension=".emptigemp1")
# res_emptigemp = network.eval_nn(model, toks_gold, tags_gold,
#                                 postagstype=postagstype)
# logger.info(network.compact_res(res_emptigemp))
# retres.append(('emptigemp', res_emptigemp))

network.train_nn(model, toks, tags, batch_size=batch_size,
                 nb_epoch=nb_epoch, postagstype=postagstype)
model.save_weights('/tmp/emptig_retrained-2.hdf5', overwrite=True)
process_test_data_tagging(model, postagstype, all_postwita_tst_flocs,
                          extension=".emptigemp2")
# res_emptigempemp = network.eval_nn(model, toks_gold, tags_gold,
#                                    postagstype=postagstype)
# logger.info(network.compact_res(res_emptigempemp))
# retres.append(('emptigempemp', res_emptigempemp))

network.train_nn(model, toks, tags, batch_size=batch_size,
                 nb_epoch=nb_epoch, postagstype=postagstype)
model.save_weights('/tmp/emptig_retrained-3.hdf5', overwrite=True)
process_test_data_tagging(model, postagstype, all_postwita_tst_flocs,
                          extension=".emptigemp3")
# res_emptigempempemp = network.eval_nn(model, toks_gold, tags_gold,
#                                       postagstype=postagstype)
# logger.info(network.compact_res(res_emptigempempemp))
# retres.append(('emptigempempemp', res_emptigempempemp))

network.train_nn(model, toks, tags, batch_size=batch_size,
                 nb_epoch=nb_epoch, postagstype=postagstype)
model.save_weights('/tmp/emptig_retrained-4.hdf5', overwrite=True)
process_test_data_tagging(model, postagstype, all_postwita_tst_flocs,
                          extension=".emptigemp4")
# res_emptigemp4 = network.eval_nn(model, toks_gold, tags_gold,
#                                  postagstype=postagstype)
# logger.info(network.compact_res(res_emptigemp4))
# retres.append(('emptigemp4', res_emptigemp4))

network.train_nn(model, toks, tags, batch_size=batch_size,
                 nb_epoch=nb_epoch, postagstype=postagstype)
model.save_weights('/tmp/emptig_retrained-5.hdf5', overwrite=True)
process_test_data_tagging(model, postagstype, all_postwita_tst_flocs,
                          extension=".emptigemp5")
# res_emptigemp5 = network.eval_nn(model, toks_gold, tags_gold,
#                                  postagstype=postagstype)
# logger.info(network.compact_res(res_emptigemp5))
# retres.append(('emptigemp5', res_emptigemp5))
# ## ###

print(retres)
exit(0)
