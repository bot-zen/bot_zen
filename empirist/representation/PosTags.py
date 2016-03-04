
from . import RepresentationError

from .. import utils

class PosTags(object):
    def _get_feature_names():
        _, empirist_tggs = utils.load_tagged_files(
            utils.cmc_tggd_flocs+utils.web_tggd_flocs)
        _, tiger_tggs = utils.load_tiger_vrt_bz2file()

        return set([pos for elem in empirist_tggs+tiger_tggs for line in elem
                    for pos in line.split('\n') if pos.isupper()])

    feature_names = _get_feature_names()

    feature_length = len(feature_names)
