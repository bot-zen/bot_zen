# -*- coding: UTF-8 -*-

import bz2
import re

from itertools import islice
from os import path

import numpy as np

from gensim.models.word2vec import Word2Vec

# when using ipython
# %load_ext autoreload
# %autoreload 2

_cmc_fnames = ["cmc_train_blog_comment.txt", "cmc_train_professional_chat.txt",
               "cmc_train_social_chat.txt", "cmc_train_twitter_1.txt",
               "cmc_train_twitter_2.txt", "cmc_train_whats_app.txt",
               "cmc_train_wiki_discussion_1.txt",
               "cmc_train_wiki_discussion_2.txt"]

_web_names = ["web_train_%03d.txt" % (i) for i in range(1, 12)]

cmc_raw_flocs = [
    path.join('../data/empirist_training_cmc/raw/', cmc_fname)
    for cmc_fname in _cmc_fnames]
cmc_tokd_flocs = [
    path.join('../data/empirist_training_cmc/tokenized/', cmc_fname)
    for cmc_fname in _cmc_fnames]
cmc_tggd_flocs = [
    path.join('../data/empirist_training_cmc/tagged/', cmc_fname)
    for cmc_fname in _cmc_fnames]

web_raw_flocs = [
    path.join('../data/empirist_training_web/raw/', web_name)
    for web_name in _web_names]
web_tokd_flocs = [
    path.join('../data/empirist_training_web/tokenized/', web_name)
    for web_name in _web_names]
web_tggd_flocs = [
    path.join('../data/empirist_training_web/tagged/', web_name)
    for web_name in _web_names]

all_raw_flocs = cmc_raw_flocs + web_raw_flocs
all_tokd_flocs = cmc_tokd_flocs + web_tokd_flocs
all_tggd_flocs = cmc_tggd_flocs + web_tggd_flocs


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


def filter_elems(elems):
    # r=utils.load_raw_files(utils.web_raw_flocs)
    # t=utils.load_tokenized_files(utils.web_tokd_flocs)
    # x,y= utils.load_tagged_files(utils.web_tggd_flocs)
    retlist = list()
    for elem in elems:
        offset = 0
        if elem[0].startswith('<') and elem[0].strip().endswith('/>'):
            offset = 1
        retlist.append([line for line in elem[offset:] if line])
    return retlist


# def load_all_tok_trntstd():
#     """
#     Load Training and Testing Data from the empirist data.
#
#
#     What we want is something like:
#     --- 8< ---
#     In : X1[0]
#     Out: ['…das hört sich ja nach einem spannend-amüsanten Ausflug an….super!']
#
#     In : y1[0]
#     Out: ['…\ndas\nhört\nsich\nja\nnach\neinem\n\\
#           spannend-amüsanten\nAusflug\nan\n…\n.\nsuper\n!']
#
#     In : X2[0]
#     Out:
#     ['6 Tipps zum Fotografieren',
#      'In diesem Video seht Ihr 6 Tipps, die beim Fotografieren helfen könnten.',
#      'Und das sind die Tipps aus dem Video:',...
#
#     In : y2[0]
#     Out:
#     ['6\nTipps\nzum\nFotografieren',
#      'In\ndiesem\nVideo\nseht\nIhr\n6\nTipps\n,\ndie\nbeim\nFotografieren\nhelfen\nkönnten\n.',
#      'Und\ndas\nsind\ndie\nTipps\naus\ndem\nVideo\n:',...
#
#     # and the following special case:
#     In : X1[174]
#     Out: ['tag quaki : )']
#
#     In : y1[174]
#     Out: ['tag\nquaki\n:)']
#     --- 8< ---
#
#     """
#     X1, y1 = load_tok_trntstd('../data/empirist_training_cmc/raw/',
#                               '../data/empirist_training_cmc/tokenized/',
#                               cmc_names)
#     X2, y2 = load_tok_trntstd('../data/empirist_training_web/raw/',
#                               '../data/empirist_training_web/tokenized/',
#                               web_names)
#
#     X = [snt for txt in X1+X2 for snt in txt]
#     y = [snt for txt in y1+y2 for snt in txt]
#     return X, y


def load_tiger_vrt_file(
        fileloc='../data/tiger/tiger_release_aug07.corrected.16012013.vrt.bz2'):
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
        for line in fp:
            if line.startswith('<s>'):
                retlist += process_tbuffy(tbuffy)
                tbuffy = list()
            elif line.startswith('</s>'):
                pass
            else:
                tbuffy.append(line.strip())
    retlist += process_tbuffy(tbuffy)
    return _process_tagged_elems([retlist])


def load_w2vs():
    """
    Load the preprocessed word2vec data for the empirist task.

    Returns:
        w2v_emp, w2v_big

        w2v_emp: trained on the empirist trainig data
        w2v_big: trained on big wikipedia data
    """
    w2v_emp = Word2Vec.load_word2vec_format('empirist.bin')
    w2v_big = Word2Vec.load_word2vec_format('bigdata.bin')
    return w2v_emp, w2v_big


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
