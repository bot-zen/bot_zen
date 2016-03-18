from gensim.models.word2vec import Word2Vec


w2v_empirist_floc = "../data/word2vec/empirist.bin"
w2v_bigdata_floc = "../data/word2vec/bigdata.bin"
_w2v_empirist = None
_w2v_bigdata = None


def _get_w2v_empirist():
    global _w2v_empirist
    if _w2v_empirist is None:
        _w2v_empirist = Word2Vec.load_word2vec_format(w2v_empirist_floc)
    return _w2v_empirist


def _get_w2v_bigdata():
    global _w2v_bigdata
    if _w2v_bigdata is None:
        _w2v_bigdata = Word2Vec.load_word2vec_format(w2v_bigdata_floc)
    return _w2v_bigdata


def w2vs():
    """
    Load the preprocessed word2vec data for the empirist task.

    Returns:
        w2v_emp, w2v_big

        w2v_emp: trained on the empirist trainig data
        w2v_big: trained on big wikipedia data
    """
    return _get_w2v_empirist(), _get_w2v_bigdata()
