import string

import numpy as np


class RepresentationError(Exception):
    """
    Exception raised for Errors in the Representation.
    """
    pass


class QOneHotChars(object):
    """
    FIXME:

    Args:
        chars:
        matrix:

    Raises:
        RepresentationError: chars and matrix were both given as args but
            did not match.
    """
    # this is what will mark a 'token boundary'
    new_token_char = '\n'

    # one hot dims for common classes
    _string_feature_names = ['ascii_lowercase', 'digits', 'punctuation',
                             'whitespace']

    # extra dims: is-uppercase, is-digit, is-punctutation, is-whitespace,
    # new token char, unknown char
    _other_feature_names = ['is_uppercase', 'is_digit', 'is_punctuation',
                            'is_whitespace', 'is_new_token_char', 'unknown']

    _feature_string = ''.join(
        [getattr(string, feature_name) for feature_name in
            _string_feature_names] +
        [' ' for _ in _other_feature_names])

    feature_length = len(_feature_string)

    feature_names = (["string."+_str for _str in _string_feature_names] +
                     _other_feature_names)

    @staticmethod
    def _encode_char(char):
        """
        Returns the quasi-one-hot index vector for a character.
            - alpha-characters are mapped to lower-case one-hot + 'is-uppercase'
            - digits are mapped to one-hot + 'is-digit'
            - punctuation marks are mapped to one-hot + 'is-punctuation'
            - whitespace (ecxept '\n') characters are mapped to one-hot +
            'is-whitespace'
            - unknowns have their own one-hot
            * '\n' is treated as new-token-character

        Args:
            char: string
            Character to index

        Returns:
            index : np.ndarray, dtype=bool, shape=(~106,1)
                Index vector of character
        """
        # make sure to process a single character
        if len(char) > 1:
            raise RepresentationError('can only cope with single a char.')

        index = np.zeros((1, QOneHotChars.feature_length)).astype(bool)

        if (char.lower() in
                QOneHotChars._feature_string[
                    0:-len(QOneHotChars._other_feature_names)]
                or char in string.ascii_uppercase):
            index[0, QOneHotChars._feature_string.index(char.lower())] = True
        else:
            index[0, QOneHotChars.feature_length-1] = True

        if char in string.ascii_uppercase:
            index[0, QOneHotChars.feature_length-6] = True
        elif char in string.digits:
            index[0, QOneHotChars.feature_length-5] = True
        elif char in string.punctuation:
            index[0, QOneHotChars.feature_length-4] = True
        elif char in string.whitespace or char == QOneHotChars.new_token_char:
            if char == QOneHotChars.new_token_char:
                index[0, QOneHotChars.feature_length-2] = True
            else:
                index[0, QOneHotChars.feature_length-3] = True

        return index

    @staticmethod
    def _decode_matrix(matrix):
        """
        Inverse of _encode().

        Args:
            index: np.ndarray, dtype=bool, shape=(~100,1)
                Index vector of character

        Returns:
            char: string
                Character of index

        """
        chars = ''
        for row in matrix:
            if row[QOneHotChars.feature_length-1]:
                char = '?'
            elif row[QOneHotChars.feature_length-2]:
                char = '\n'
            else:
                char = QOneHotChars._feature_string[row.tolist().index(True)]
                if row[QOneHotChars.feature_length-6]:
                    char = char.upper()
            chars = ''.join(chars+char)
        return chars

    def __init__(self, chars=None, matrix=None):
        self._chars = chars
        self._matrix = matrix

        if chars is not None and matrix is not None:
            if matrix != self._encode() and chars != self._decode():
                raise RepresentationError('chars and matrix do not match.')

        if chars is not None:
            self._matrix = self._encode()

        if matrix is not None:
            self._chars = self._decode()

    @property
    def chars(self):
        return self._chars

    @property
    def matrix(self):
        return self._matrix

    def _encode(self):
        """
        Converts self.chars to the quasi-one-hot matrix representation.

        Returns:
            quasi one-hot matrix of self.chars.
            np.ndarray, dtype=bool, shape=(len(self.chars),self.feature_length)
        """
        matrix = np.zeros((len(self.chars),
                           QOneHotChars.feature_length)).astype(bool)
        for rdx in range(len(self._chars)):
            matrix[rdx, ] = self._encode_char(self._chars[rdx])
        return matrix

    def _decode(self):
        """
        Convert self.matrix to the corresponding character representation.

        Returns:
            chars representation of self.matrix.
        """
        self._chars = QOneHotChars._decode_matrix(self._matrix)
        return self._chars
