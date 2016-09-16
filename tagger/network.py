import numpy as np

from . import utils, logger


def build_nn(input_dim=1800, output_dim=None, lstm_output_dim=512, dropout=0.5):
    logger.info("building nn...")
    from keras.models import Sequential

    from keras.layers.core import Dropout, TimeDistributedDense
    from keras.layers.recurrent import LSTM

    from .representation.postags import PosTagsType
    if output_dim is None:
        postagstype = PosTagsType()
        output_dim = postagstype.feature_length

    return_sequence = True

    model = Sequential()
    model.add(LSTM(lstm_output_dim, input_dim=input_dim,
                   return_sequences=True, stateful=False))
    model.add(Dropout(0.5))
    model.add(LSTM(lstm_output_dim, input_dim=lstm_output_dim,
                   return_sequences=True, stateful=False,
                   activation='sigmoid', inner_activation='hard_sigmoid'))
    model.add(Dropout(dropout))
    model.add(TimeDistributedDense(output_dim, activation="softmax"))
    model.compile(loss="categorical_crossentropy", optimizer="rmsprop")
    logger.info("building nn...done.")
    return model


def train_nn(model, toks, tags, batch_size=10,
             nb_epoch=10, verbose=1, show_accuracy=True, postagstype=None):
    # toks, tags = utils.load_tagged_files(utils.all_tggd_flocs)
    x, _, _, _ = utils.training_data_tagging(toks, tags,
                                             postagstype=postagstype)
    xlens = list(set([len(_x) for _x in x]))
    for xlenid, xlen in enumerate(xlens):
        _x, _y, _, _yorg = utils.training_data_tagging(toks, tags,
                                                       seqlen=int(xlen),
                                                       postagstype=postagstype)
        print("len:%i (%i/%i), x:%i" % (xlen, xlenid, len(xlens), len(_x)))
        if len(_x) < batch_size:
            use_batch_size = 1
        else:
            use_batch_size = batch_size

        if xlen > 1:
            model.fit(np.array(_x), np.array(_y), batch_size=use_batch_size,
                      nb_epoch=nb_epoch, verbose=verbose,
                      show_accuracy=show_accuracy)


def eval_nn(model, toks, tags, verbose=1, postagstype=None):
    retres = []
    xtst, _, _, _ = utils.training_data_tagging(toks, tags,
                                                postagstype=postagstype)
    for xtstlenid, xtstlen in enumerate(sorted(list(set([len(_x) for _x in
                                                         xtst])))):
        xtstseq, ytstseq, _, _ = utils.training_data_tagging(
            toks, tags, seqlen=xtstlen, postagstype=postagstype)
        res = model.evaluate(np.array(xtstseq), np.array(ytstseq),
                             batch_size=len(xtstseq), verbose=verbose,
                             show_accuracy=True)
        retres.append((xtstlen, len(xtstseq), res))
    return retres


def compact_res(res):
    return sum([r[2][1] for r in res])/len(res)


def qgrid_search():
    lstm_output_dims = [128, 256, 512, 1024]
    dropouts = [0.1, 0.25, 0.5, 0.75]
    nb_epochs = [5, 10, 20]

    retres = []
    for lstm_output_dim in lstm_output_dims:
        for dropout in dropouts:
            model = build_nn(lstm_output_dim=lstm_output_dim, dropout=dropout)
            model.save_weights('/tmp/tmpmodel.hdf5', overwrite=True)
            for nb_epoch in nb_epochs:
                train_nn(model, nb_epoch=nb_epoch, verbose=0)
                res = eval_nn(model, xtstflocs=utils.all_trial_tggd_flocs,
                              verbose=0)
                print("lstm_od:%i, drpt:%0.2f, nb_epc:%i, acc:%f" %
                      (lstm_output_dim, dropout, nb_epoch, compact_res(res)))
                retres.append((lstm_output_dim, dropout, nb_epoch,
                               compact_res(res)))
                model.load_weights('/tmp/tmpmodel.hdf5')
    return retres
