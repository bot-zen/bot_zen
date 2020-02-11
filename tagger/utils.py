# -*- coding: UTF-8 -*-

import bz2
import re

from itertools import islice
from os import path

import numpy as np

from . import logger
from .representation import postags as _postags
from .representation import qonehotchars as _qonehotchars
from .slow_utils import w2v_big, w2v_small

np.random.seed(0)  # For reproducability

_cmc_fnames = ["cmc_train_blog_comment.txt",
               "cmc_train_professional_chat.txt",
               "cmc_train_social_chat.txt",
               "cmc_train_twitter_1.txt",
               "cmc_train_twitter_2.txt",
               "cmc_train_whats_app.txt",
               "cmc_train_wiki_discussion_1.txt",
               "cmc_train_wiki_discussion_2.txt"]

_web_names = ["web_train_%03d.txt" % (i) for i in range(1, 12)]

cmc_raw_flocs = [
    path.join('../data/empirist/empirist_training_cmc/raw/', cmc_fname)
    for cmc_fname in _cmc_fnames]
cmc_tokd_flocs = [
    path.join('../data/empirist/empirist_training_cmc/tokenized/', cmc_fname)
    for cmc_fname in _cmc_fnames]
cmc_tggd_flocs = [
    path.join('../data/empirist/empirist_training_cmc/tagged/', cmc_fname)
    for cmc_fname in _cmc_fnames]

web_raw_flocs = [
    path.join('../data/empirist/empirist_training_web/raw/', web_name)
    for web_name in _web_names]
web_tokd_flocs = [
    path.join('../data/empirist/empirist_training_web/tokenized/', web_name)
    for web_name in _web_names]
web_tggd_flocs = [
    path.join('../data/empirist/empirist_training_web/tagged/', web_name)
    for web_name in _web_names]

all_raw_flocs = cmc_raw_flocs + web_raw_flocs
all_tokd_flocs = cmc_tokd_flocs + web_tokd_flocs
all_tggd_flocs = cmc_tggd_flocs + web_tggd_flocs
all_postwita_tggd_flocs = (['../data/postwita/postwita.vrt',
                            '../data/postwita/didi.vrt'])
all_postwita_tst_flocs = (['../data/postwita/tst.txt'])

_cmc_trial_fnames = ["professional_chat.txt",
                     "social_chat.txt",
                     "tweets.txt",
                     "wikipedia_talk_pages.txt"]

_web_trial_fnames = ["trial006_hobby.txt",
                     "trial007_reisen.txt",
                     "trial008_lifestyle.txt",
                     "trial009_beruf.txt",
                     "trial010_sonstige.txt"]

cmc_trial_raw_flocs = [
    path.join('../data/empirist/empirist_trial_cmc/raw/', cmc_fname)
    for cmc_fname in _cmc_trial_fnames]
cmc_trial_tokd_flocs = [
    path.join('../data/empirist/empirist_trial_cmc/tokenized/', cmc_fname)
    for cmc_fname in _cmc_trial_fnames]
cmc_trial_tggd_flocs = [
    path.join('../data/empirist/empirist_trial_cmc/tagged/', cmc_fname)
    for cmc_fname in _cmc_trial_fnames]

web_trial_raw_flocs = [
    path.join('../data/empirist/empirist_trial_web/raw/', web_name)
    for web_name in _web_trial_fnames]
web_trial_tokd_flocs = [
    path.join('../data/empirist/empirist_trial_web/tokenized/', web_name)
    for web_name in _web_trial_fnames]
web_trial_tggd_flocs = [
    path.join('../data/empirist/empirist_trial_web/tagged/', web_name)
    for web_name in _web_trial_fnames]

all_trial_raw_flocs = cmc_trial_raw_flocs + web_trial_raw_flocs
all_trial_tokd_flocs = cmc_trial_tokd_flocs + web_trial_tokd_flocs
all_trial_tggd_flocs = cmc_trial_tggd_flocs + web_trial_tggd_flocs

_cmc_tst_fnames = ["cmc_test_blog_comment.txt",
                   "cmc_test_professional_chat.txt",
                   "cmc_test_social_chat.txt",
                   "cmc_test_twitter.txt",
                   "cmc_test_whatsapp.txt",
                   "cmc_test_wiki_discussion.txt"]
_web_tst_names = ["web_test_%03d.txt" % (i) for i in range(1, 13)]

cmc_tst_tokd_flocs = [
    path.join('../data/empirist/empirist_test_pos_cmc/tokenized/', cmc_fname)
    for cmc_fname in _cmc_tst_fnames]
web_tst_tokd_flocs = [
    path.join('../data/empirist/empirist_test_pos_web/tokenized/', web_name)
    for web_name in _web_tst_names]
all_tst_tokd_flocs = cmc_tst_tokd_flocs + web_tst_tokd_flocs

cmc_gold_flocs = [
    path.join('../data/empirist/empirist_gold_cmc/tagged/', cmc_fname)
    for cmc_fname in _cmc_tst_fnames]
web_gold_flocs = [
    path.join('../data/empirist/empirist_gold_web/tagged/', web_name)
    for web_name in _web_tst_names]
all_gold_flocs = cmc_gold_flocs + web_gold_flocs


def load_raw_file(fileloc):
    """
    Loads a raw empirist file.

    Args:
        fileloc: string

    Returns:
        list of lists
    """
    retlist = list()  # the list of lists to return
    tbuffy = list()   # tmp buffer
    with open(fileloc) as fp:
        for line in fp:
            # deal with XML meta-data lines:
            if line.startswith('<') and line.strip().endswith('/>'):
                if len(tbuffy) > 0:
                    # ...this was not the first XML meta-data line we've come
                    # accross:
                    # save the recently seen content and empty the tmp buffer.
                    retlist.append(tbuffy)
                    tbuffy = list()
                # in any case, don't forget to process the XML meta-data line
                tbuffy.append(line.strip())
            else:
                # ah, not a XML meta-data line.
                # now, normalise this:
                #    - strip
                #    - reduce more than 2 whitespace to one token marker
                tbuffy.append(re.sub('\s{2,}', '\n', line.strip()))
    if len(tbuffy) > 0:
        # don't forget to empty the tmp buffer
        retlist.append(tbuffy)
    return retlist


def load_raw_files(filelocs):
    """
    Load multiple raw empirist files.
    """
    return sum([load_raw_file(fileloc) for fileloc in filelocs], [])


def _load_tagdtokd_file(fileloc):
    """
    Load a tokenized or tokenized+tagged empirist file.

    Args:
        fileloc: string

    Returns:
        list of lists
    """
    def process_tbuffy(tbuffy):
        """
        Process the tmp buffer:
            - be aware of XML meta-data lines

        Args:
            tbuffy: the temporary buffer (consisting of lines) to process

        """
        retlist = list()
        if len(tbuffy) > 1:
            emptytail = False
            #
            if tbuffy[-1] == "":
                emptytail = True
                tbuffy = tbuffy[:-1]
            if tbuffy[0].startswith('<') and tbuffy[0].strip().endswith('/>'):
                jtbuffy = '\n'.join(tbuffy[1:]).split('\n\n')
                retlist.append([tbuffy[0]] + [element for tuple in
                                              zip(jtbuffy[:-1],
                                                  (len(jtbuffy)-1)*['']) for
                                              element in tuple] + [jtbuffy[-1]])
            else:
                jtbuffy = '\n'.join(tbuffy[0:]).split('\n\n')
                retlist.append([element for tuple in zip(jtbuffy[:-1],
                                                         (len(jtbuffy)-1)*[''])
                                for element in tuple] + [jtbuffy[-1]])

            if emptytail is True:
                retlist[-1] = retlist[-1]+['']
        elif len(tbuffy) > 0:
            retlist.appen(tbuffy)
        return retlist

    retlist = list()
    tbuffy = list()
    with open(fileloc) as fp:
        for line in fp:
            if line.startswith('<') and line.strip().endswith('/>'):
                retlist += process_tbuffy(tbuffy)
                tbuffy = list()
            tbuffy.append(line.strip())
    retlist += process_tbuffy(tbuffy)

    return retlist


def load_tokenized_file(fileloc):
    """
    Load a tokenized empirist file.

    Args:
        fileloc: location of the file to load

    Returns:
        List of lists of tokenized empirist elements

        --- 8< ---
        [
         ['<posting author="B. Tovar" date="23. Januar 2013 um 15:21" />',
          'Wusstet\nIhr\n,\ndass\nes\neine\nSeite\n“\nFettehenne.info\n”\
              \nim\nNetz\ngibt\n?',
          ''],
         ['<posting author="Carl Frederick Luthin" date="23. Januar 2013 um \
             15:24" />',
          'Danke\nfür\ndeinen\nKommentar\n!\nJa\n,\ndie\nSeite\nkenne\nich\
              \n…\n;-)',
          '']
        ]
        --- 8< ---
    """
    return _load_tagdtokd_file(fileloc)


def load_tokenized_files(filelocs):
    """
    Load multiple tokenized empirist files.
    """
    return sum([load_tokenized_file(fileloc) for fileloc in filelocs], [])


def _process_tagged_elems(elements):
    """
    FIXME:
    """
    tokelems, tagelems = list(), list()
    for elem in elements:
        toklines, taglines = list(), list()
        for line in elem:
            tokline, tagline = "", ""
            if line.startswith('<') and line.endswith('/>'):
                toklines.append(line)
                taglines.append(line)
            else:
                pairs = line.split('\n')
                if all(['\t' in pair for pair in pairs]):
                    for tok, tag in [pair.split('\t') for pair in pairs]:
                        tokline = '\n'.join([tokline, tok]).strip()
                        tagline = '\n'.join([tagline, tag]).strip()
                    toklines.append(tokline)
                    taglines.append(tagline)
                else:
                    toklines.append(line)
                    taglines.append(line)
        tokelems.append(toklines)
        tagelems.append(taglines)
    return tokelems, tagelems


def load_tagged_file(fileloc):
    """
    Load a tagged empirist file.

    Args:
        fileloc: location of the file to load

    Retruns:
        pair of list of lists of tokenized and tagged elements

        --- 8< ---
        (
        [
         ['<posting author="B. Tovar" date="23. Januar 2013 um 15:21" />',
          'Wusstet\nIhr\n,\ndass\nes\neine\nSeite\n“\nFettehenne.info\n”\
              \nim\nNetz\ngibt\n?',
          ''],
         ['<posting author="Carl Frederick Luthin" date="23. Januar 2013 um \
             15:24" />',
          'Danke\nfür\ndeinen\nKommentar\n!\nJa\n,\ndie\nSeite\nkenne\nich\
              \n…\n;-)',
          '']
        ]
        ,
        [
         ['<posting author="B. Tovar" date="23. Januar 2013 um 15:21" />',
          'VVFIN\nPPER\n$,\nKOUS\nPPER\nART\nNN\n$(\nURL\n$(\nAPPRART\nNN\n\
              VVFIN\n$.',
          ''],
         ['<posting author="Carl Frederick Luthin" date="23. Januar 2013 um \
             15:24" />',
          'PTKANT\nAPPR\nPPOSAT\nNN\n$.\nPTKANT\n$,\nART\nNN\nVVFIN\nPPER\n\
              $.\nEMOASC',
          '']
        ]
        )
        --- 8< ---
    """
    return _process_tagged_elems(_load_tagdtokd_file(fileloc))


def load_tagged_files(filelocs):
    """
    Load multiple tagged empirist files.
    """
    rettoks, rettggs = list(), list()
    for fileloc in filelocs:
        toks, tggs = load_tagged_file(fileloc)
        rettoks.extend(toks)
        rettggs.extend(tggs)
    return rettoks, rettggs


def filter_elem(elem):
    offset = 0
    if elem[0].startswith('<') and elem[0].strip().endswith('/>'):
        offset = 1
    return [line for line in elem[offset:] if line]


def filter_elems(elems):
    # r=utils.load_raw_files(utils.web_raw_flocs)
    # t=utils.load_tokenized_files(utils.web_tokd_flocs)
    # x,y= utils.load_tagged_files(utils.web_tggd_flocs)
    retlist = list()
    for elem in elems:
        retlist.append(filter_elem(elem))
    return retlist


def load_tiger_vrt_file(
        fileloc='../data/tiger_release_aug07.corrected.16012013-empirist.vrt.bz2'):
    """
    Load a bz2 compressed tiger vrt (tok\tlem\tpos) file.

    Args:
        fileloc: location of the compressed vertical file

    """
    # utils.load_tiger_vrt_file('../data/tiger/tiger_release_aug07.corrected.16012013.vrt.bz2')
    def process_tbuffy(tbuffy):
        retlist = list()
        if len(tbuffy) > 0:
            tmplist = list()
            for line in tbuffy:
                tok, lem, pos = line.split('\t')
                tmplist.append('\t'.join([tok, pos]))
            retlist.append('\n'.join(tmplist))
        return retlist

    retlist = list()
    tbuffy = list()
    with bz2.open(fileloc, mode='rt') as fp:
        for lid, line in enumerate([line.strip() for line in fp]):
            if line.startswith('<s>'):
                retlist += process_tbuffy(tbuffy)
                tbuffy = list()
            elif line.startswith('</s>'):
                pass
            else:
                logger.debug("%8d\t%s" % (lid+1, line))
                tbuffy.append(line)
    retlist += process_tbuffy(tbuffy)
    return _process_tagged_elems([retlist])


def _sanitize_tok(tok):
    if tok == '“': tok = '"'
    elif tok == '”': tok = '"'
    tok = re.sub('\d', '0', tok)
    return tok


def _encode_tok(tok, suffix_length=8):
    toklen = len(tok)
    stoken = ''.join([" " * (suffix_length - toklen), tok[0:suffix_length],
                      tok[-suffix_length:], " " * (suffix_length - toklen)])
    return _qonehotchars.encode(stoken).reshape(1280,)


def _nearby_tok(w2v, tok, tokid, tokens):
    if tok in w2v:
        retmat = w2v[tok]
    else:
        nearby_before = [_tok for _tok in tokens[:tokid] if _tok in w2v]
        nearby_after = [_tok for _tok in tokens[tokid+1:] if _tok in w2v]
        before_idx = 0
        if len(nearby_before) > 5:
            before_idx = len(nearby_before)-6
        nearby = ((nearby_before + nearby_after)[before_idx:])[0:10]

        if nearby:
            retmat = w2v[w2v.most_similar(nearby, topn=1)[0][0]]
            logger.debug("\ntokens: %s\n|%s #%s# %s\nnearby tokens: %s" %
                         (str(tokens), str(nearby_before), tok,
                          str(nearby_after), nearby))
        else:
            retmat = w2v.seeded_vector(tok)
            logger.debug("no tokens: %s" % (str(tokens)))
    return retmat


def training_data_tagging(toks, tags, sample_size=-1, seqlen=None,
                          padding=False, shuffle=False, postagstype=None):
    """
    Load Training and Testing Data from the empirist data.

    What we want is something like:
    --- 8< ---
    In : X1[0]
    Out: ['…das hört sich ja nach einem spannend-amüsanten Ausflug an….super!']

    In : y1[0]
    Out: ['…\ndas\nhört\nsich\nja\nnach\neinem\n\\
          spannend-amüsanten\nAusflug\nan\n…\n.\nsuper\n!']

    In : X2[0]
    Out:
    ['6 Tipps zum Fotografieren',
     'In diesem Video seht Ihr 6 Tipps, die beim Fotografieren helfen könnten.',
     'Und das sind die Tipps aus dem Video:',...

    In : y2[0]
    Out:
    ['6\nTipps\nzum\nFotografieren',
     'In\ndiesem\nVideo\nseht\nIhr\n6\nTipps\n,\ndie\nbeim\nFotografieren\nhelfen\nkönnten\n.',
     'Und\ndas\nsind\ndie\nTipps\naus\ndem\nVideo\n:',...

    # and the following special case:
    In : X1[174]
    Out: ['tag quaki : )']

    In : y1[174]
    Out: ['tag\nquaki\n:)']
    --- 8< ---

    Args:
       toks, tags = load_tagged_files(flocs)

    Returns:
        x, y, xorg, yorg:
    """
    all_tggd = (filter_elems(toks), filter_elems(tags))
    tok_elems = all_tggd[0]
    tag_elems = all_tggd[1]

    if sample_size > 0:
        tok_elems = tok_elems[0:sample_size]
        tag_elems = tag_elems[0:sample_size]

    w2v_empirist = w2v_small.data
    w2v_bigdata = w2v_big.data
    x, y, xorg, yorg = [], [], [], []

    for eid, elem in enumerate(tok_elems):
        tokelem = elem
        tagelem = tag_elems[eid]

        for tokline in tokelem:
            toksline = []
            tokens = tokline.split('\n')
            if seqlen is not None and len(tokens) != seqlen:
                continue

            for tokid, tok in enumerate([_sanitize_tok(tok) for tok in tokens]):
                tok_encd = _encode_tok(tok)
                tok_l = tok.lower()
                x_emp = _nearby_tok(w2v_empirist, tok_l, tokid,
                                    [_sanitize_tok(tok) for tok in tokens])
                x_big = _nearby_tok(w2v_bigdata, tok_l, tokid,
                                    [_sanitize_tok(tok) for tok in tokens])
                toksline.append(np.concatenate((x_emp, x_big, tok_encd)))
                dummy_emp = np.zeros(
                    w2v_empirist.seeded_vector(_postags.padding_tag).shape)
                dummy_big = np.zeros(
                    w2v_bigdata.seeded_vector(_postags.padding_tag).shape)
                dummy = np.concatenate((dummy_emp, dummy_big,
                                        np.zeros(tok_encd.shape)))
            if padding:
                dummies = [dummy for did in range(seqlen-len(tokens))]
                toksline.extend(dummies)
            x.append(toksline)
            xorg.append(tokline)

        for line in tagelem:
            postags = line.split('\n')
            if seqlen is not None and len(postags) != seqlen:
                continue

            if padding:
                postags_enc = _postags.encode(postags + [_postags.padding_tag] *
                                              (seqlen-len(postags)),
                                              postagstype=postagstype)
            else:
                postags_enc = _postags.encode(postags, postagstype=postagstype)
            y.append(postags_enc)
            yorg.append(line)

    return x, y, xorg, yorg


def get_test_data_tagging(flocs=all_tst_tokd_flocs):
    toks, tags = load_tagged_files(flocs)
    all_tggd = (filter_elems(toks), filter_elems(tags))
    tok_elems = all_tggd[0]
    w2v_empirist = w2v_small.data
    w2v_bigdata = w2v_big.data
    x, xorg = [], []

    for eid, elem in enumerate(tok_elems):
        tokelem = elem

        for tokline in tokelem:
            toksline = []
            tokens = tokline.split('\n')

            for tokid, tok in enumerate([_sanitize_tok(tok) for tok in tokens]):
                tok_encd = _encode_tok(tok)
                tok_l = tok.lower()
                x_emp = _nearby_tok(w2v_empirist, tok_l, tokid,
                                    [_sanitize_tok(tok) for tok in tokens])
                x_big = _nearby_tok(w2v_bigdata, tok_l, tokid,
                                    [_sanitize_tok(tok) for tok in tokens])
                toksline.append(np.concatenate((x_emp, x_big, tok_encd)))
            x.append(toksline)
            xorg.append(tokline)

    return x, xorg


def process_test_data_tagging(model, postagstype, flocs, extension=".done"):
    # for floc in all_tst_tokd_flocs:
    for floc in flocs:
        elems, elems_org = get_test_data_tagging(flocs=[floc])
        prcd_floc = floc + extension
        with open(prcd_floc, 'w') as prcdh:
            for elemid, elem in enumerate(elems):
                preds = model.predict_classes(np.array([elem]), batch_size=1,
                                              verbose=0)[0]
                tags = _postags.decode_oh(preds, postagstype)
                lines = [''] + ['\t'.join(x) for x in
                                zip(elems_org[elemid].split('\n'), tags)]
                prcdh.writelines('\n'.join(lines)+'\n')
                if 'cmc/' in floc:
                    prcdh.write('\n')

    # for file in empirist_test_pos_{cmc,web}/tokenized/*.done; do paste
    # $(dirname $file)/$(basename $file .done) <(cut -f2 $file); done | less


def run_experiments():
    # from tagger import slow_utils
    # from tagger import utils
    from tagger import network
    # import numpy as np
    from tagger.representation import postags

    retres = []

    postagstype_ibk = postags.PosTagsType(feature_type="ibk")
    postagstype_ibk_used = postags.PosTagsType(feature_type="ibk_used")
    postagstype_tiger_used = postags.PosTagsType(feature_type="1999_used")

    toks, tags = load_tagged_files(all_tggd_flocs)
    toks_trial, tags_trial = load_tagged_files(all_trial_tggd_flocs)
    toks_tig, tags_tig = load_tiger_vrt_file()
    dropout = 0.1
    batch_size = 10
    nb_epoch = 20

    ###
    # QUICK TESTING
    # toks = toks[0:100]
    # tags = tags[0:100]

    # dropout = 0.1
    # batch_size = 2
    # nb_epoch = 2
    ###

    model = network.build_nn(output_dim=postagstype_ibk_used.feature_length,
                             lstm_output_dim=1024, dropout=dropout)
    model.save_weights('/tmp/emp_plain.hdf5', overwrite=True)
    network.train_nn(model, toks, tags, batch_size=batch_size,
                     nb_epoch=nb_epoch, postagstype=postagstype_ibk_used)
    model.save_weights('/tmp/emp_trained.hdf5', overwrite=True)
    res_emp = network.eval_nn(model, toks_trial, tags_trial,
                              postagstype=postagstype_ibk_used)
    network.compact_res(res_emp)
    process_test_data_tagging(model, extension=".emp",
                              postagstype=postagstype_ibk_used)
    retres.append(('emp', res_emp))

    batch_size = 50
    model = network.build_nn(output_dim=postagstype_tiger_used.feature_length,
                             lstm_output_dim=1024, dropout=dropout)
    model.save_weights('/tmp/tig_plain.hdf5', overwrite=True)
    network.train_nn(model, toks_tig, tags_tig, batch_size=batch_size,
                     nb_epoch=nb_epoch, postagstype=postagstype_tiger_used)
    model.save_weights('/tmp/tig_trained.hdf5', overwrite=True)
    res_tig = network.eval_nn(model, toks_trial, tags_trial,
                              postagstype=postagstype_tiger_used)
    network.compact_res(res_tig)
    process_test_data_tagging(model, extension=".tig",
                              postagstype=postagstype_tiger_used)
    retres.append(('tig', res_tig))

    batch_size = 20
    model = network.build_nn(output_dim=postagstype_ibk.feature_length,
                             lstm_output_dim=1024, dropout=dropout)
    model.save_weights('/tmp/emptig_plain.hdf5', overwrite=True)

    network.train_nn(model, toks, tags, batch_size=batch_size,
                     nb_epoch=nb_epoch, postagstype=postagstype_ibk)
    model.save_weights('/tmp/emptig_trained-0.hdf5', overwrite=True)
    res_emp2 = network.eval_nn(model, toks_trial, tags_trial,
                               postagstype=postagstype_ibk)
    network.compact_res(res_emp2)
    batch_size = 50
    network.train_nn(model, toks_tig, tags_tig, batch_size=batch_size,
                     nb_epoch=nb_epoch, postagstype=postagstype_ibk)
    model.save_weights('/tmp/emptig_trained-1.hdf5', overwrite=True)
    res_emptig = network.eval_nn(model, toks_trial, tags_trial,
                                 postagstype=postagstype_ibk)
    network.compact_res(res_emptig)
    process_test_data_tagging(model, extension=".emptig",
                              postagstype=postagstype_ibk)
    retres.append(('emptig', res_emptig))

    return retres


def sliding_window(seq, n=2, return_shorter_seq=True):
    """
    Returns a sliding window (of width n) over data from the iterable
    s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...

    inspired by https://docs.python.org/release/2.3.5/lib/itertools-example.html
    """
    if n > len(seq) and return_shorter_seq:
        n = len(seq)

    it = iter(seq)
    result = tuple(islice(it, n))
    if len(result) == n:
        yield result
    for elem in it:
        result = result[1:] + (elem,)
        yield result


def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    """
    inspired by https://github.com/Lasagne/Lasagne/blob/master/examples/mnist.py
    """
    assert len(inputs) == len(targets)
    if shuffle:
        raise Exception('CRAZY!')
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        indices = np.arange(len(inputs))
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]
