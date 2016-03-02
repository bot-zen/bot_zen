import re

from itertools import islice
from os import path

import numpy as np

# when using ipython
# %load_ext autoreload
# %autoreload 2


def load_raw_file(fileloc):
    """
    Loads an empirist raw file

    :paramters:
        - path : string

    :returns:
        - retlist
            list of lists
    """
    retlist = list()  # the list of lists to return
    tbuffy = list()   # tmp buffer
    with open(fileloc) as fp:
        for line in fp:
            # deal with XML meta-data lines:
            if line.startswith('<') and line.strip().endswith('/>'):
                if len(tbuffy) > 0:
                    # ...this was not the first XML meta-data line we've coma
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


def load_tokd_file(fileloc):
    """
    Loads a tokenized empirist file

    :parameters:
        - path : string

    :return:
        -retlist
            list of lists
    """
    def process_tbuffy(tbuffy):
        """
        Process the tmp buffer:
            - be aware of XML meta-data lines

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


def load_tok_trntstd(trainloc, trgloc, names):
    X = list()
    y = list()
    for fn in names:
        for tr in load_raw_file(path.join(trainloc, fn)):
            offset = 0
            if tr[0].startswith('<') and tr[0].strip().endswith('/>'):
                offset = 1
            X.append([x.strip() for x in ' '.join(tr[offset:]).split('  ')])
        for tr in load_tokd_file(path.join(trgloc, fn)):
            offset = 0
            if tr[0].startswith('<') and tr[0].strip().endswith('/>'):
                offset = 1
            y.append([x for x in tr[offset:] if x != ''])

    return X, y


def load_all_tok_trntstd():
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

    """
    cmc_names = ["cmc_train_blog_comment.txt",
                 "cmc_train_professional_chat.txt", "cmc_train_social_chat.txt",
                 "cmc_train_twitter_1.txt", "cmc_train_twitter_2.txt",
                 "cmc_train_whats_app.txt", "cmc_train_wiki_discussion_1.txt",
                 "cmc_train_wiki_discussion_2.txt"]

    web_names = ["web_train_%03d.txt" % (i) for i in range(1, 12)]

    X1, y1 = load_tok_trntstd('../data/empirist_training_cmc/raw/',
                              '../data/empirist_training_cmc/tokenized/',
                              cmc_names)
    X2, y2 = load_tok_trntstd('../data/empirist_training_web/raw/',
                              '../data/empirist_training_web/tokenized/',
                              web_names)

    X = [snt for txt in X1+X2 for snt in txt]
    y = [snt for txt in y1+y2 for snt in txt]
    return X, y


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
