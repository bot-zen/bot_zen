from . import RepresentationError

from .. import utils


class PosTags(object):
    stts_ibk_floc = "../data/stts_ibk.txt"
    stts_1999_floc = "../data/stts_1999.txt"

    def __init__(self, feature_type="ibk"):
        self._feature_type = feature_type
        self._feature_names = None
        self._feature_length = None

    @property
    def feature_type(self):
        return self._feature_type

    @feature_type.setter
    def feature_type(self, feature_type):
        self._feature_type = feature_type
        self._set_feature_names()

    @property
    def feature_names(self):
        if not self._feature_names:
            self._set_feature_names()
        return self._feature_names

    @property
    def feature_length(self):
        return len(self.feature_names)

    def _set_feature_names(self):
        if self.feature_type == "ibk":
            self._feature_names = self.get_stts_ibk()
        elif self.feature_type == "1999":
            self._feature_names = self.get_stts_1999()
        elif self.feature_type == "ibk_used":
            self._feature_names = self.get_stts_ibk_used()
        elif self.feature_type == "1999_used":
            self._feature_names = self.get_stts_1999_used()

    @staticmethod
    def get_stts_ibk(fileloc=stts_ibk_floc):
        return sorted([line.strip() for line in open(fileloc).readlines() if
                       line])

    @staticmethod
    def get_stts_1999(fileloc=stts_1999_floc):
        return sorted([line.strip() for line in open(fileloc).readlines() if
                       line])

    @staticmethod
    def get_stts_ibk_used():
        _, yc = utils.load_tagged_files(utils.cmc_tggd_flocs)
        _, yw = utils.load_tagged_files(utils.web_tggd_flocs)
        return sorted(set([pos for elem in utils.filter_elems(yc+yw) for line in
                           elem for pos in line.split('\n')]))

    @staticmethod
    def get_stts_1999_used():
        _, tiger_tggs = utils.load_tiger_vrt_file()
        return sorted(set([pos for elem in tiger_tggs for line in elem for pos
                           in line.split('\n')]))
