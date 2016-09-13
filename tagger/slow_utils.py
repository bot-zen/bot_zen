import logging

from . import config, logger, CONFIG_TASK, CONFIG_ENV_DEFAULTS

# don't spam logging with gensim warnings (during initial import)
logging.disable(logging.INFO)
from gensim.models.word2vec import Word2Vec
logging.disable(logging.NOTSET)


class W2V():
    """
    helper class to read word2vec files.
    """
    def __init__(self, floc):
        self.floc = floc
        self._data = None
        logger.debug(floc)

    @property
    def data(self):
        if self._data is None:
            self._data = Word2Vec.load_word2vec_format(self.floc)
        return self._data

w2v_small = W2V(config.get(CONFIG_TASK, 'w2v_small_floc',
                           vars=CONFIG_ENV_DEFAULTS))
w2v_big = W2V(config.get(CONFIG_TASK, 'w2v_big_floc',
                         vars=CONFIG_ENV_DEFAULTS))
