import logging
import math
import numpy as np
import scipy.sparse
from scipy.stats import entropy
import scipy.linalg
from scipy.linalg.lapack import get_lapack_funcs
from scipy.linalg.special_matrices import triu
from scipy.special import psi  # gamma function utils

class Sparse2Corpus:
    """Convert a matrix in scipy.sparse format into a streaming Gensim corpus.
    See Also
    --------
    :func:`~gensim.matutils.corpus2csc`
        Convert gensim corpus format to `scipy.sparse.csc` matrix
    :class:`~gensim.matutils.Dense2Corpus`
        Convert dense matrix to gensim corpus.
    """
    def __init__(self, sparse, documents_columns=True):
        """
        Parameters
        ----------
        sparse : `scipy.sparse`
            Corpus scipy sparse format
        documents_columns : bool, optional
            Documents will be column?
        """
        if documents_columns:
            self.sparse = sparse.tocsc()
        else:
            self.sparse = sparse.tocsr().T  # make sure shape[1]=number of docs (needed in len())

    def __iter__(self):
        """
        Yields
        ------
        list of (int, float)
            Document in BoW format.
        """
        for indprev, indnow in zip(self.sparse.indptr, self.sparse.indptr[1:]):
            yield list(zip(self.sparse.indices[indprev:indnow], self.sparse.data[indprev:indnow]))

    def __len__(self):
        return self.sparse.shape[1]

    def __getitem__(self, key):
        """
        Retrieve a document vector or subset from the corpus by key.
        Parameters
        ----------
        key: int, ellipsis, slice, iterable object
            Index of the document retrieve.
            Less commonly, the key can also be a slice, ellipsis, or an iterable
            to retrieve multiple documents.
        Returns
        -------
        list of (int, number), Sparse2Corpus
            Document in BoW format when `key` is an integer. Otherwise :class:`~gensim.matutils.Sparse2Corpus`.
        """
        sparse = self.sparse
        if isinstance(key, int):
            iprev = self.sparse.indptr[key]
            inow = self.sparse.indptr[key + 1]
            return list(zip(sparse.indices[iprev:inow], sparse.data[iprev:inow]))

        sparse = self.sparse.__getitem__((slice(None, None, None), key))
        return Sparse2Corpus(sparse)
