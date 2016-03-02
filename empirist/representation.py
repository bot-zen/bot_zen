import string

import numpy as np


class Error(Exception):
    """Base class for exceptions in this module."""
    pass


class RepresenationError(Error):
    """
    Exception raised for errors in the input.

    Attributes:
        expr -- input expression in which the error occurred
        msg  -- explanation of the error
    """

    def __init__(self, msg):
        self.msg = msg


class QOneHotChars(object):

    def __init__(self, chars=None, matrix=None):
        # this is what will mark a 'token boundary'
        self.new_token_char = '\n'

        # one hot dims for common classes
        self._string_feature_names = ['ascii_lowercase', 'digits',
                                      'punctuation', 'whitespace']

        # extra dims: is-uppercase, is-digit, is-punctutation, is-whitespace,
        # new token char, unknown char
        self._other_feature_names = ['is_uppercase', 'is_digit',
                                     'is_punctuation', 'is_whitespace',
                                     'is_new_token_char', 'unknown']

        self._feature_string = ''.join(
            [getattr(string, feature_name) for feature_name in
             self._string_feature_names] +
            [' ' for _ in self._other_feature_names])

        self._feature_length = len(self._feature_string)

        self._feature_names = (["string."+_str for _str in
                                self._string_feature_names] +
                               self._other_feature_names)

        self.chars = chars
        self.matrix = matrix

        if chars is not None and matrix is not None:
            if matrix != self._encode() and chars != self._decode():
                raise RepresenationError('chars and matrix do not match!')

        if chars:
            self.matrix = self._encode()

        if matrix is not None:
            self.chars = self._decode()

    @property
    def feature_names(self):
        return self._feature_names

    @property
    def feature_length(self):
        return self._feature_length

    def _encode_char(self, char):
        """
        Returns the quasi-one-hot index vector for a character.
            - alpha-characters are mapped to lower-case one-hot + 'is-uppercase'
            - digits are mapped to one-hot + 'is-digit'
            - punctuation marks are mapped to one-hot + 'is-punctuation'
            - whitespace (ecxept '\n') characters are mapped to one-hot +
            'is-whitespace'
            - unknowns have their own one-hot
            * '\n' is treated as new-token-character

        :paramtetrs:
            - char : string
                Character to index

        :returns:
            - index : np.ndarray, dtype=bool, shape=(~106,1)
                Index vector of character
        """
        # make sure to process a single character
        if len(char) > 1:
            return None

        index = np.zeros((1, self.feature_length)).astype(bool)

        if (char.lower() in
                self._feature_string[0:-len(self._other_feature_names)] or char
                in string.ascii_uppercase):
            index[0, self._feature_string.index(char.lower())] = True
        else:
            index[0, self.feature_length-1] = True

        if char in string.ascii_uppercase:
            index[0, self._feature_length-6] = True
        elif char in string.digits:
            index[0, self.feature_length-5] = True
        elif char in string.punctuation:
            index[0, self.feature_length-4] = True
        elif char in string.whitespace or char == self.new_token_char:
            if char == self.new_token_char:
                index[0, self.feature_length-2] = True
            else:
                index[0, self.feature_length-3] = True

        return index

    def _decode_matrix(self, matrix):
        """
        Inverse of _encode()

        :paramtetrs:
            - index : np.ndarray, dtype=bool, shape=(~100,1)
                Index vector of character

        :returns:
            - char : string
                Character of index

        """
        chars = ''
        for row in matrix:
            if row[self.feature_length-1] is True:
                char = '?'
            elif row[self.feature_length-2] is True:
                char = '\n'
            else:
                char = self._feature_string[row.tolist().index(True)]
                if row[self.feature_length-6] is True:
                    char = char.upper()
            chars = ''.join(chars+char)
        return chars

    def _encode(self):
        """
        Converts an array of characters to a quasi-one-hot matrix using
        ``_encode_char()``

        :parameters:
            - chars : string

        :returns:
            - one_hot : np.ndarray, dtype=bool,
            shape=(len(chars),len(qone_hot_char()))
                One-hot matrix of the input
        """
        matrix = np.zeros((len(self.chars), self.feature_length)).astype(bool)
        for rdx in range(len(self.chars)):
            matrix[rdx, ] = self._encode_char(self.chars[rdx])
        return matrix

    def _decode(self):
        self.chars = self._decode_matrix(self.matrix)
        return self.chars

    def encoded(self):
        return self.matrix

    def decoded(self):
        return self.chars
