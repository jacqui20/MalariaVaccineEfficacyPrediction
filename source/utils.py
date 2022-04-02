# Copyright (c) 2020 Bernhard Reuter.
# ------------------------------------------------------------------------------------------------
# If you use this code or parts of it, cite the following reference:
# ------------------------------------------------------------------------------------------------
# XXXXXX
# ------------------------------------------------------------------------------------------------
# This is free software: you can redistribute it and/or modify it under the terms of the GNU
# Lesser General Public License as published by the Free Software Foundation, either version 3
# of the License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
# without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License along with this program.
# If not, see <http://www.gnu.org/licenses/>.
# ------------------------------------------------------------------------------------------------

import numpy as np
import os.path
from typing import Any, Dict, Union, Optional, Tuple, List
from sklearn.utils import check_X_y
from sklearn.utils.multiclass import unique_labels
from sklearn.model_selection._split import BaseCrossValidator
from sklearn.utils.validation import column_or_1d
from sklearn.model_selection import StratifiedKFold
import warnings
from sklearn.metrics.pairwise import rbf_kernel, sigmoid_kernel, polynomial_kernel
import pandas as pd
import os
from sklearn.preprocessing import KernelCenterer


def normalize(
    X: np.ndarray,
    clip: bool = False,
    assert_sym: bool = False,
) -> np.ndarray:
    """Utility function (loop version) to normalize a square symmetric matrix by
    normalized_element[i, j] = element[i, j] / np.sqrt(element[i, i] * element[j, j]).

    Parameters
    ----------
    X : np.ndarray
        Square symmetric Matrix to normalize.
    assert_sym : bool, default = False
        If True, assert that the supplied Gram matrix is symmetric.

    Returns
    -------
    X_norm : np.ndarray
        Normalized square symmetric matrix.
    """
    # assert that X matrix is square
    assert X.shape[0] == X.shape[1], "Matrix is not square."
    if assert_sym:
        # assert that X matrix is symmetric:
        assert np.allclose(X, X.T), "Matrix isn't symmetric."
    # check, if all diagonal elements are positive:
    if np.any(np.diag(X) <= 0.0):
        raise ValueError(
            "Some diagonal elements of the matrix are <=0. Can't normalize."
        )
    X_norm = np.zeros(X.shape, dtype=np.double)
    for i in range(X.shape[0]):
        for j in range(i, X.shape[1]):
            X_norm[i, j] = X[i, j] / np.sqrt(X[i, i] * X[j, j])
            X_norm[j, i] = X_norm[i, j]
    return X_norm


def normalize_fast(
    X: np.ndarray,
    clip: bool = False,
    assert_sym: bool = False,
) -> np.ndarray:
    """Utility function (fastest vectorized version) to normalize a square symmetric matrix by
    normalized_element[i, j] = element[i, j] / np.sqrt(element[i, i] * element[j, j]).
    This version is significantly faster than normalize_vec for large (> 1000 x 1000) matrices!

    Parameters
    ----------
    X : np.ndarray
        Square symmetric Matrix to normalize.
    assert_sym : bool, default = False
        If True, assert that the supplied Gram matrix is symmetric.

    Returns
    -------
    X_norm : np.ndarray
        Normalized square symmetric matrix.
    """
    M, N = X.shape
    # assert that X matrix is square
    assert M == N, "Matrix is not square."
    if assert_sym:
        # assert that X matrix is symmetric:
        assert np.allclose(X, X.T), "Matrix isn't symmetric."
    # check, if all diagonal elements are positive:
    if np.any(np.diag(X) <= 0.0):
        raise ValueError(
            "Some diagonal elements of the matrix are <=0. Can't normalize."
        )
    D = 1.0 / np.sqrt(np.diag(X))
    DD = D.reshape((N, 1)) @ D.reshape((1, N))
    return DD * X


def check_complex_or_negative(
    ev: np.ndarray,
    warn: bool = True,
) -> Tuple[bool, bool]:
    complex = False
    negative = False
    if np.any(np.iscomplex(ev)):
        if np.max(np.abs(ev.imag)) > np.finfo(ev.dtype).eps:
            if warn:
                print(
                    'Complex eigenvalues with significant imaginary part found:\n'
                    f'{ev[np.abs(ev.imag) > np.finfo(ev.dtype).eps]}'
                )
            complex = True
    if np.any(ev.real < 0.0):
        if warn:
            print(
                'Eigenvalues with negative real parts found:\n'
                f'{ev[ev.real < 0.0]}'
            )
        negative = True
    return complex, negative


def make_symmetric_matrix_psd(
    X: np.ndarray,
    epsilon_factor: float = 1.e3,
) -> Tuple[np.ndarray, List[float], List[str]]:
    """ Spectral Translation approach
    Tests, if a given symmetric matrix is positive semi-definite (psd) and, if not,
    a damping value c is iteratively added to the matrix diagonal, i.e. `X = X + c * I`,
    until the kernel matrix becomes positive semi-definite.
    Non-positive-semi-definiteness will be tested via a spectral criterion:
    If `X` has negative or complex eigenvalues, `X` can not be psd.
    The damping value `c` is derived from the spectrum of `X`: If negative eigenvalues exist,
    `c = max([epsilon_factor * np.finfo(X.dtype).eps, np.min(np.linalg.eigvals(X).real)])`,
    else,if complex but non-negative eigenvalues exist,
    `c = max([epsilon_factor * np.finfo(X.dtype).eps, np.abs(np.max(eigenvalues.imag))])`.

    Parameters
    ----------
    X: np.ndarray
        Symmetric matrix to test and make psd, if necessary.
    epsilon_factor : float, default = 1e3
        Factor `epsilon_factor` to multiply the machine precision `np.finfo(X.dtype).eps` with,
        resulting in `c = epsilon_factor * np.finfo(X.dtype).eps` beinig the mimimum value
        to add to the diagonal of `X`, if `X` is not psd.

    Returns
    -------
    X_psd : np.ndarray
        Symmetric positive semi-definite matrix.
    """
    epsilon = epsilon_factor * np.finfo(X.dtype).eps
    c_list = []
    info_list = []

    # check, if X is square
    if X.shape[0] != X.shape[1]:
        raise ValueError("Matrix is not square.")

    # check, if X is symmetric:
    if not np.allclose(X, X.T):
        raise ValueError("Matrix is not symmetric.")

    eigenvalues = np.linalg.eigvals(X)

    # check, if all eigenvalues are real and non-negative
    complex, negative = check_complex_or_negative(eigenvalues)

    if complex or negative:
        print(
            "Matrix is not positive semi-definite.\n"
            "Will try to make it positive semi-definite."
        )

        n_negative = 0
        n_imaginary = 0
        counter = 0
        while (complex or negative) and counter <= 1000:
            counter += 1
            if negative:
                n_negative += 1
                info = 'negative'
                c = np.abs(np.min(eigenvalues).real)
            else:
                n_imaginary += 1
                info = 'imaginary'
                c = np.max(np.abs(eigenvalues.imag))
            if c < epsilon:
                info += '_epsilon'
                c = epsilon
            info_list.append(info)
            c_list.append(c)
            np.fill_diagonal(X, np.diag(X) + c)
            eigenvalues = np.linalg.eigvals(X)
            complex, negative = check_complex_or_negative(eigenvalues, warn=False)

        complex, negative = check_complex_or_negative(eigenvalues)
        if complex or negative:
            print(
                "Couldn't make matrix positive semi-definite by adding"
                f"sum_c={np.sum(c_list)} in {len(c_list)} steps to its diagonal.\n"
                f"Damped {n_negative} times for negative eigenvalues "
                f"and {n_imaginary} times for imaginary parts."
            )
        else:
            print(
                "Made matrix positive semi-definite by adding "
                f"sum_c={np.sum(c_list)} in {len(c_list)} steps to its diagonal.\n"
                f"Damped {n_negative} times for negative eigenvalues "
                f"and {n_imaginary} times for imaginary parts."
            )
    return X, c_list, info_list


def make_symmetric_matrix_pd(
    X: np.ndarray,
    epsilon_factor: float = 1.e1,
) -> Tuple[np.ndarray, List[float]]:
    """USE WITH CAUTION (see below)
    Tests, IN THEORY, if a given symmetric matrix is positive definite (pd) and, if not,
    a damping value c is iteratively added to the matrix diagonal, i.e. `X = X + c * I`,
    until the kernel matrix becomes positive definite.
    Non-positive-definiteness will be tested via np.linalg.cholesky(Y):
    The Cholesky decomposition fails for matrices `X` that aren't positive definite.
    IN PRACTICE, np.linalg.cholesky(Y) sometimes passes for matrices with defective rank.
    This stands in contradiction to the mathematical fact that pd matrices must have full rank.
    Thus, in the best case, the resulting matrices are in fact only psd. In the worst case,
    the checks in np.linalg.cholesky(Y) (and thus LAPACK) are entirely flawed.
    THUS, THIS FUNCTION SHOULD BE USED WITH CAUTION.
    The damping value `c` is derived from the spectrum of `X`: If non-positive eigenvalues exist,
    `c = max([epsilon_factor * np.finfo(X.dtype).eps, np.min(np.linalg.eigvals(X).real)])`,
    else, if complex but positive eigenvalues exist,
    `c = max([epsilon_factor * np.finfo(X.dtype).eps, np.abs(np.max(eigenvalues.imag))])`.

    Parameters
    ----------
    X: np.ndarray
        Symmetric matrix to test and make psd, if necessary.
    epsilon_factor : float, default = 1e1
        Factor `epsilon_factor` to multiply the machine precision `np.finfo(X.dtype).eps` with,
        resulting in `c = epsilon_factor * np.finfo(X.dtype).eps` beinig the mimimum value
        to add to the diagonal of `X`, if `X` is not pd.

    Returns
    -------
    X_psd : np.ndarray
        Symmetric positive definite matrix.
    """
    epsilon = epsilon_factor * np.finfo(X.dtype).eps
    c_list = []

    # check, if X is square
    if X.shape[0] != X.shape[1]:
        raise ValueError("Matrix is not square.")

    # check, if X is symmetric:
    if not np.allclose(X, X.T):
        raise ValueError("Matrix is not symmetric.")

    # check, if matrix is pd by computing its cholesky decomposition
    is_pd = is_symmetric_positive_definite(X, warn_npd=False)

    eigenvalues = np.linalg.eigvals(X)

    # check, if all eigenvalues are real and non-negative
    complex, negative = check_complex_or_negative(eigenvalues)

    if not is_pd:
        print(
            "Matrix is not positive definite.\n"
            "Will try to make it positive definite."
        )

        while not is_pd:
            eigenvalues = np.linalg.eigvals(X)
            complex, negative = check_complex_or_negative(eigenvalues, warn=False)
            if negative:
                c = np.abs(np.min(eigenvalues.real))
            else:
                if complex:
                    c = np.max(np.abs(eigenvalues.imag))
                else:
                    c = epsilon
            if c < epsilon:
                c = epsilon
            c_list.append(c)
            np.fill_diagonal(X, np.diag(X) + c)
            is_pd = is_symmetric_positive_definite(X, warn_npd=False)

        is_pd = is_symmetric_positive_definite(X, warn_npd=False)
        if not is_pd:
            raise ValueError(
                "Couldn't make matrix positive definite by adding"
                f"sum_c={np.sum(c_list)} in {len(c_list)} steps to its diagonal."
            )
        else:
            print(
                "Made matrix positive definite by adding "
                f"sum_c={np.sum(c_list)} in {len(c_list)} steps to its diagonal."
            )
    return X, c_list


def is_symmetric_positive_semidefinite(
    X: np.ndarray,
    epsilon_factor: float = 1.e1
) -> bool:
    """Utility function to check, if a matrix is symmetric positive semidefinite (s.p.s.d).
    The check is based on np.linalg.cholesky(Y) and is thus typically faster than
    computing all eigenvalues of X.

    The Cholesky decomposition fails for matrices X that aren't positive definite.
    Since we want to accept positive semi-definite matrices X as well,
    we must regularize X by adding a small postive definite matrix,
    e.g. Y = X + 1e-14 * I (I being the (positiv definite) identity matrix).
    By doing so, Y becomes positive definite for positive semi-definite X
    and passes the Cholesky decomposition.
    This is due to the following facts:
       Scaling: If M is positive definite and r > 0 is a real number,
       then r*M is positive definite.
       Addition: If M is positive-definite and N is positive-semidefinite,
       then the sum M+N is also positive-definite.
    Interesting discussions regarding this problem:
    https://stackoverflow.com/questions/16266720/find-out-if-matrix-is-positive-definite-with-numpy
    https://scicomp.stackexchange.com/questions/12979/testing-if-a-matrix-is-positive-semi-definite

    Parameters
    ----------
    X : np.ndarray
        Square symmetric Matrix to test.
    epsilon_factor : float, default = 1e1
        Factor `f` to multiply the machine precision `eps` with resulting in
        `Y = X + f * eps * I` beinig the matrix supplied to np.linalg.cholesky()
        that is checked for positive definiteness.

    Returns
    -------
    b : bool
        True, if X is s.p.s.d.
    """
    epsilon = epsilon_factor * np.finfo(X.dtype).eps
    # check, if X is square
    if X.shape[0] != X.shape[1]:
        warnings.warn("Matrix is not square.")
        return False

    # check, if X is symmetric:
    if not np.allclose(X, X.T):
        warnings.warn("Matrix is not symmetric.")
        return False

    try:
        Y = X + np.eye(X.shape[0]) * epsilon
        np.linalg.cholesky(Y)
        return True
    except np.linalg.LinAlgError as err:
        if "Matrix is not positive definite" in str(err):
            warnings.warn(f"Matrix X + {epsilon} * I is not positive definite")
            return False
        else:
            raise


def is_symmetric_positive_definite(
    X: np.ndarray,
    warn_npd: bool = True,
) -> bool:
    """Utility function to check, if a matrix is symmetric positive definite (s.p.d.).
    The check is based on np.linalg.cholesky(Y) and is thus typically faster than
    computing all eigenvalues of X.

    The Cholesky decomposition fails for matrices X that aren't positive definite.

    Parameters
    ----------
    X : np.ndarray
        Square symmetric Matrix to test.

    Returns
    -------
    b : bool
        True, if X is s.p.d.
    """
    # check, if X is square
    if X.shape[0] != X.shape[1]:
        warnings.warn("Matrix is not square.")
        return False

    # check, if X is symmetric:
    if not np.allclose(X, X.T):
        warnings.warn("Matrix is not symmetric.")
        return False

    # The Cholesky decomposition fails for matrices X that aren't positive definite.
    try:
        np.linalg.cholesky(X)
        return True
    except np.linalg.LinAlgError as err:
        if "Matrix is not positive definite" in str(err):
            if warn_npd:
                warnings.warn("Matrix is not positive definite")
            return False
        else:
            raise


def assign_folds(
    labels: np.ndarray,
    groups: np.ndarray,
    delta: int,
    step: int,
    n_splits: int = 5,
    shuffle: bool = True,
    print_info: bool = True,
    random_state: Optional[Union[int, np.random.mtrand.RandomState]] = None,
):
    """
    Provides 1-D arrays of train/test indices used in CustomPredefinedSplit
    to split data into train/test sets.
    This routine is designed for time-ordered data that consists of samples
    taken at several timepoints that belong to the same group (e.g. patient).
    Consider, e.g., 40 patients that underwent antibody measurements against
    malaria at multiple (3) time-points after vaccination. Thus we have 120
    samples at 3 timepoints in 40 groups. We now want to split the data
    into disjunct train/test sets, such that the samples in the test sets
    are all taken from the same time point, while the samples in the train
    sets are taken from all timepoints under the constraint that patients
    (groups) appearing in the respective test set don't appear in the
    associated train set.

    CAUTION: This routine is only tested for

    Parameters
    ----------
    labels : array-like of shape (n_samples,)
        Target labels. Used for stratification.
    groups : array-like of shape (n_samples,)
        Group labels for the samples used while splitting the dataset into
        train/test set.
    delta : int
        Step-width. ``len(labels) % delta = 0`` must be ensured.
    step : int
        Step. Used together with ``delta`` to select the interval the test
        samples are selected from.
    n_splits : int, default=5
        Number of folds. Must be at least 2.
    shuffle : bool, default=True
        Whether to shuffle each class's samples before splitting into batches.
        Note that the samples within each split will not be shuffled.
    print_info : bool, default=True
        If True, printout detailed information about the train/test splits.
    random_state : int, RandomState instance or None, default=None
        When shuffle is True, random_state affects the ordering of the indices,
        which controls the randomness of each fold for each class. Otherwise,
        leave random_state as None. Pass an int for reproducible output across
        multiple function calls.

    Returns
    -------
    test_fold : np.ndarray of shape (n_samples,)
        The entry ``test_fold[i]`` represents the index of the test set that
        sample ``i`` belongs to. It is possible to exclude sample ``i`` from
        any test set by setting ``test_fold[i]`` equal to -1.
    train_fold : np.ndarray of shape (n_samples,)
        The entry ``train_fold[i]`` represents the index of the train set that
        sample ``i`` is excluded from. It is possible to include sample ``i``
        in any test set by setting ``train_fold[i]`` equal to -1.
    """
    labels = column_or_1d(labels)
    groups = column_or_1d(groups)
    assert len(labels) == len(groups), "labels and groups arrays must be of same length."
    assert len(labels) % delta == 0, "len(labels) % delta = 0 must be ensured."
    assert delta + delta * step <= len(labels), \
        "delta + delta * step <= len(labels) must be ensured"
    labels_slice = labels[delta * step: delta + delta * step]
    groups_slice = groups[delta * step: delta + delta * step]
    skf = StratifiedKFold(n_splits, shuffle=shuffle, random_state=random_state)
    test_fold = np.array([-1 for i in range(len(labels))])
    train_fold = np.array([-1 for i in range(len(labels))])
    for i, (train_index, test_index) in enumerate(
        skf.split(np.zeros((delta, delta)), labels_slice)
    ):
        exclude_groups = []
        for j in test_index:
            test_fold[j + delta * step] = i
            exclude_groups.append(groups_slice[j])
        train_fold = np.where(np.isin(groups, exclude_groups), i, train_fold)
        if print_info:
            print("TEST:", test_index + delta * step)
            print('exclude groups:', exclude_groups)
            print('train_fold:', train_fold)
    if print_info:
        print('test_fold:', test_fold)
        cps = CustomPredefinedSplit(test_fold, train_fold)
        for i, (train_index, test_index) in enumerate(cps.split()):
            print(
                f"TRAIN (len={len(train_index)}): {train_index} "
                f"TEST (len={len(test_index)}): {test_index}"
            )
        print('')
    return test_fold, train_fold


class CustomPredefinedSplit(BaseCrossValidator):
    """Predefined split cross-validator
    Provides train/test indices to split data into train/test sets using a
    predefined scheme specified by the user with the ``test_fold`` parameter.

    Parameters
    ----------
    test_fold : array-like of shape (n_samples,)
        The entry ``test_fold[i]`` represents the index of the test set that
        sample ``i`` belongs to. It is possible to exclude sample ``i`` from
        any test set by setting ``test_fold[i]`` equal to -1.
    train_fold : array-like of shape (n_samples,)
        The entry ``train_fold[i]`` represents the index of the train set that
        sample ``i`` is excluded from. It is possible to include sample ``i``
        in any test set by setting ``train_fold[i]`` equal to -1.
    """

    def __init__(self, test_fold, train_fold):
        self.test_fold = np.array(test_fold, dtype=int)
        self.test_fold = column_or_1d(self.test_fold)
        self.train_fold = np.array(train_fold, dtype=int)
        self.train_fold = column_or_1d(self.train_fold)
        assert len(self.test_fold) == len(self.train_fold), \
            "test_fold and train_fold must be of equal length."
        unique_test_folds = np.unique(self.test_fold)
        unique_test_folds = unique_test_folds[unique_test_folds != -1]
        unique_train_folds = np.unique(self.train_fold)
        unique_train_folds = unique_train_folds[unique_train_folds != -1]
        assert np.array_equal(unique_test_folds, unique_train_folds), \
            "test_fold and train fold must have the same fold indices."
        self.unique_folds = unique_test_folds

    def split(self, X=None, y=None, groups=None):
        """Generate indices to split data into training and test set.
        Parameters
        ----------
        X : object
            Always ignored, exists for compatibility.
        y : object
            Always ignored, exists for compatibility.
        groups : object
            Always ignored, exists for compatibility.
        Yields
        ------
        train : ndarray
            The training set indices for that split.
        test : ndarray
            The testing set indices for that split.
        """
        ind = np.arange(len(self.test_fold))
        for test_index, train_index in self._iter_test_masks():
            train_index = ind[train_index]
            test_index = ind[test_index]
            yield train_index, test_index

    def _iter_test_masks(self):
        """Generates boolean masks corresponding to test sets."""
        for f in self.unique_folds:
            test_index = np.where(self.test_fold == f)[0]
            test_mask = np.zeros(len(self.test_fold), dtype=bool)
            test_mask[test_index] = True
            train_index = np.where(self.train_fold == f)[0]
            train_mask = np.ones(len(self.train_fold), dtype=bool)
            train_mask[train_index] = False
            yield test_mask, train_mask

    def get_n_splits(self, X=None, y=None, groups=None):
        """Returns the number of splitting iterations in the cross-validator
        Parameters
        ----------
        X : object
            Always ignored, exists for compatibility.
        y : object
            Always ignored, exists for compatibility.
        groups : object
            Always ignored, exists for compatibility.
        Returns
        -------
        n_splits : int
            Returns the number of splitting iterations in the cross-validator.
        """
        return len(self.unique_folds)


class DataSelector:

    def __init__(
        self,
        *,
        kernel_directory: str,
        identifier: str,
        SA: Union[float, str] = 'X',
        SO: Union[float, str] = 'X',
        R0: Union[float, str] = 'X',
        R1: Union[float, str] = 'X',
        R2: Union[float, str] = 'X',
        P1: Union[float, str] = 'X',
        P2: Union[float, str] = 'X',
    ) -> None:
        self.kernel_directory = kernel_directory
        self.identifier = identifier
        self.SA = SA
        self.SO = SO
        self.R0 = R0
        self.R1 = R1
        self.R2 = R2
        self.P1 = P1
        self.P2 = P2

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> "DataSelector":
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)
        # Return the classifier
        return self

    def transform(
        self,
        X: np.ndarray,
    ) -> np.ndarray:

        if isinstance(self.R0, float):
            if isinstance(self.R1, float) and isinstance(self.R2, float):
                fn = (f"{self.identifier}_SA_{self.SA}_SO_{self.SO}_R0_{self.R0:.1E}_"
                      f"R1_{self.R1:.1E}_R2_{self.R2:.1E}_P1_{self.P1}_P2_{self.P2}.npy")
            elif isinstance(self.R1, float) and not isinstance(self.R2, float):
                fn = (f"{self.identifier}_SA_{self.SA}_SO_{self.SO}_R0_{self.R0:.1E}_"
                      f"R1_{self.R1:.1E}_R2_{self.R2}_P1_{self.P1}_P2_{self.P2}.npy")
            elif not isinstance(self.R1, float) and isinstance(self.R2, float):
                fn = (f"{self.identifier}_SA_{self.SA}_SO_{self.SO}_R0_{self.R0:.1E}_"
                      f"R1_{self.R1}_R2_{self.R2:.1E}_P1_{self.P1}_P2_{self.P2}.npy")
            else:
                fn = (f"{self.identifier}_SA_{self.SA}_SO_{self.SO}_R0_{self.R0:.1E}_"
                      f"R1_{self.R1}_R2_{self.R2}_P1_{self.P1}_P2_{self.P2}.npy")
        else:
            if isinstance(self.R1, float) and isinstance(self.R2, float):
                fn = (f"{self.identifier}_SA_{self.SA}_SO_{self.SO}_R0_{self.R0}_"
                      f"R1_{self.R1:.1E}_R2_{self.R2:.1E}_P1_{self.P1}_P2_{self.P2}.npy")
            elif isinstance(self.R1, float) and not isinstance(self.R2, float):
                fn = (f"{self.identifier}_SA_{self.SA}_SO_{self.SO}_R0_{self.R0}_"
                      f"R1_{self.R1:.1E}_R2_{self.R2}_P1_{self.P1}_P2_{self.P2}.npy")
            elif not isinstance(self.R1, float) and isinstance(self.R2, float):
                fn = (f"{self.identifier}_SA_{self.SA}_SO_{self.SO}_R0_{self.R0}_"
                      f"R1_{self.R1}_R2_{self.R2:.1E}_P1_{self.P1}_P2_{self.P2}.npy")
            else:
                fn = (f"{self.identifier}_SA_{self.SA}_SO_{self.SO}_R0_{self.R0}_"
                      f"R1_{self.R1}_R2_{self.R2}_P1_{self.P1}_P2_{self.P2}.npy")

        kernel = np.load(os.path.join(self.kernel_directory, fn))

        # kernel is a precomputed square kernel matrix
        if kernel.shape[0] != kernel.shape[1]:
            raise ValueError("Oligo kernel should be a square kernel matrix")

        return np.ascontiguousarray(kernel.flatten('C')[X.flatten('C')].reshape(X.shape))

    def get_params(
        self,
        deep: bool = True
    ) -> Dict[str, Any]:
        return {
            'kernel_directory': self.kernel_directory,
            'identifier': self.identifier,
            'SA': self.SA,
            'SO': self.SO,
            'R1': self.R1,
            'R2': self.R2,
            'P1': self.P1,
            'P2': self.P2,
        }

    def set_params(
        self,
        **parameters: Any,
    ) -> "DataSelector":
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self


def make_kernel_combinations(
    meta_data: np.ndarray,
    kernel_time_series: str,
    kernel_dosage: str,
    kernel_abSignal: str
) -> Dict[str, Union[float, str]]:
    """Make dictionary of kernel combination values

    For each kernel parameter a range of values is defined.

    Parameters
    ----------
    meta_data : np.ndarray
        Matrix of kernel parameter ranges.
    kernel_time_series : str
        Kernel function to compute relationship between individuals based on time points.
    kernel_dosage : str
        Pre-defined kernel function for to represent relationship between individuals
        based on dosage level.
    kernel_abSignal : str
        Pre-defined kernel function for to represent relationship between individuals
        based on ab signal intensity

    Returns
    -------
    kernel_comb_param : dict
        Dictionary of value ranges for each kernel parameter.

    """
    if (kernel_time_series == "sigmoid_kernel" and kernel_dosage == "rbf_kernel"
            and kernel_abSignal == "rbf_kernel"):
        kernel_comb_param = {"SA": np.arange(meta_data[1, 1], 1, meta_data[1, 1]),
                             "SO": np.arange(meta_data[2, 1], meta_data[2, 2]),
                             "R0": np.array(['X']),
                             "R1": 10 ** np.arange(meta_data[3, 1], meta_data[3, 2], dtype=float),
                             "R2": 10 ** np.arange(meta_data[3, 1], meta_data[3, 2], dtype=float),
                             "P1": np.array(['X']),
                             "P2": np.array(['X'])}
    elif (kernel_time_series == "sigmoid_kernel" and kernel_dosage == "poly_kernel"
            and kernel_abSignal == "poly_kernel"):
        kernel_comb_param = {"SA": np.arange(meta_data[1, 1], 1, meta_data[1, 1]),
                             "SO": np.arange(meta_data[2, 1], meta_data[2, 2], dtype=float),
                             "R0": np.array(['X']),
                             "R1": np.array(['X']),
                             "R2": np.array(['X']),
                             "P1": np.arange(meta_data[0, 1], meta_data[0, 2]),
                             "P2": np.arange(meta_data[0, 1], meta_data[0, 2])}
    elif (kernel_time_series == "sigmoid_kernel" and kernel_dosage == "poly_kernel"
            and kernel_abSignal == "rbf_kernel"):
        kernel_comb_param = {"SA": np.arange(meta_data[1, 1], 1, meta_data[1, 1]),
                             "SO": np.arange(meta_data[2, 1], meta_data[2, 2]),
                             "R0": np.array(['X']),
                             "R1": np.array(['X']),
                             "R2": 10 ** np.arange(meta_data[3, 1], meta_data[3, 2], dtype=float),
                             "P1": np.arange(meta_data[0, 1], meta_data[0, 2]),
                             "P2": np.array(['X'])}
    elif (kernel_time_series == "sigmoid_kernel" and kernel_dosage == "rbf_kernel"
            and kernel_abSignal == "poly_kernel"):
        kernel_comb_param = {"SA": np.arange(meta_data[1, 1], 1, meta_data[1, 1]),
                             "SO": np.arange(meta_data[2, 1], meta_data[2, 2]),
                             "R0": np.array(['X']),
                             "R1": 10 ** np.arange(meta_data[3, 1], meta_data[3, 2], dtype=float),
                             "R2": np.array(['X']),
                             "P1": np.array(['X']),
                             "P2": np.arange(meta_data[0, 1], meta_data[0, 2])}
    elif (kernel_time_series == "rbf_kernel" and kernel_dosage == "rbf_kernel"
            and kernel_abSignal == "rbf_kernel"):
        kernel_comb_param = {"SA": np.array(['X']),
                             "SO": np.array(['X']),
                             "R0": 10 ** np.arange(meta_data[3, 1], meta_data[3, 2], dtype=float),
                             "R1": 10 ** np.arange(meta_data[3, 1], meta_data[3, 2], dtype=float),
                             "R2": 10 ** np.arange(meta_data[3, 1], meta_data[3, 2], dtype=float),
                             "P1": np.array(['X']),
                             "P2": np.array(['X'])}
    elif (kernel_time_series == "rbf_kernel" and kernel_dosage == "poly_kernel"
            and kernel_abSignal == "poly_kernel"):
        kernel_comb_param = {"SA": np.array(['X']),
                             "SO": np.array(['X']),
                             "R0": 10 ** np.arange(meta_data[3, 1], meta_data[3, 2], dtype=float),
                             "R1": np.array(['X']),
                             "R2": np.array(['X']),
                             "P1": np.arange(meta_data[0, 1], meta_data[0, 2]),
                             "P2": np.arange(meta_data[0, 1], meta_data[0, 2])}
    elif (kernel_time_series == "rbf_kernel" and kernel_dosage == "poly_kernel"
            and kernel_abSignal == "rbf_kernel"):
        kernel_comb_param = {"SA": np.array(['X']),
                             "SO": np.array(['X']),
                             "R0": 10 ** np.arange(meta_data[3, 1], meta_data[3, 2], dtype=float),
                             "R1": np.array(['X']),
                             "R2": 10 ** np.arange(meta_data[3, 1], meta_data[3, 2], dtype=float),
                             "P1": np.arange(meta_data[0, 1], meta_data[0, 2]),
                             "P2": np.array(['X'])}
    elif (kernel_time_series == "rbf_kernel" and kernel_dosage == "rbf_kernel"
            and kernel_abSignal == "poly_kernel"):
        kernel_comb_param = {"SA": np.array(['X']),
                             "SO": np.array(['X']),
                             "R0": 10 ** np.arange(meta_data[3, 1], meta_data[3, 2], dtype=float),
                             "R1": 10 ** np.arange(meta_data[3, 1], meta_data[3, 2], dtype=float),
                             "R2": np.array(['X']),
                             "P1": np.array(['X']),
                             "P2": np.arange(meta_data[0, 1], meta_data[0, 2])}

    return kernel_comb_param


def multitask(
    a: np.ndarray,
    b: np.ndarray,
) -> np.ndarray:
    """ Multitask approach

    Combination of two kernel matrices are combined by
    element-wise multiplication to one kernel matrix.

    Parameters
    ----------
    a : np.ndarray
        Kernel matrix a.
    b : np.ndarray
        Kernel matrix b.

    Returns
    -------
    a * b : np.ndarray
        Element-wise multiplicated kernel matrix a * b.
    """
    return a * b


def make_kernel_matrix(
    data: pd.DataFrame,
    model: dict,
    kernel_time_series: str,
    kernel_dosage: str,
    kernel_abSignals: str,
    scale: bool = False,
) -> Tuple[np.ndarray, List[float], List[float]]:
    """ Compute combined kernel matrix.

    for each task, time, PfSPZ dose and antibody signal intensity a kernel matrix
    is pre-computed given the pre-defined kernel function and combined to one
    matrix by the multitask approach.

    Parameters
    ----------
    data : pd.Dataframe
        Pre-processed proteome data.
    model : dict
        Combination of kernel parameters.
    kernel_time_series : str
        Kernel function to compute relationship between individuals based on time points.
    kernel_dosage : str
        Kernel function to compute relationship between individuals based on dosage level.
    kernel_abSignals : str
        Kernel function to compute relationship between individuals
        based on antibody signal intensity.
    scale : bool, default=False
        If True, scale the combined matrix by first centering it followed by normalization.

    Returns
    -------
    multi_AB_signals_time_dose_kernel_matrix : np.ndarray
        Returns a combined matrix.
    """

    # get start point of antibody reactivity signals in data (n_p)
    AB_signal_start = data.columns.get_loc("TimePointOrder") + 1
    AB_signals = data[data.columns[AB_signal_start:]]

    # extract time points to vector (y_t)
    time_series = data["TimePointOrder"]

    # extract dosage to vector (y_d)
    dose = data["Dose"]

    # pre-computed kernel matrix of time points K(n_t,n_t')
    if kernel_time_series == 'sigmoid_kernel':
        time_series_kernel_matrix = sigmoid_kernel(
                time_series.values.reshape(len(time_series), 1),
                gamma=model[0],
                coef0=model[1],
        )
    elif kernel_time_series == 'rbf_kernel':
        time_series_kernel_matrix = rbf_kernel(
            time_series.values.reshape(len(time_series), 1), gamma=model[2]
        )

    # pre-computed kernel matrix of antibody reactivity K(n_p,n_p')
    if kernel_abSignals == "rbf_kernel":
        AB_signals_kernel_matrix = rbf_kernel(AB_signals, gamma=model[4])
    elif kernel_abSignals == "poly_kernel":
        AB_signals_kernel_matrix = polynomial_kernel(AB_signals, degree=model[6])

    # pre-computed kernel matrix of dosage K(n_d,n_d')
    if kernel_dosage == "rbf_kernel":
        dose_kernel_matrix = rbf_kernel(dose.values.reshape(len(dose), 1), gamma=model[3])
    elif kernel_dosage == "poly_kernel":
        dose_kernel_matrix = polynomial_kernel(dose.values.reshape(len(dose), 1), degree=model[5])

    # pre-compute multitask kernel matrix K((np, nt),(np', nt'))
    multi_AB_signals_time_series_kernel_matrix = multitask(
        AB_signals_kernel_matrix,
        time_series_kernel_matrix,
    )

    # pre-compute multitask kernel matrix K((np, nt, nd),(np', nt', nd'))
    multi_AB_signals_time_dose_kernel_matrix, c_list, info_list = make_symmetric_matrix_psd(
        multitask(
            multi_AB_signals_time_series_kernel_matrix,
            dose_kernel_matrix,
        )
    )
    if c_list:
        print(
            "multi_AB_signals_time_dose_kernel_matrix kernel had to be corrected.\n"
            f"model: {model}"
        )

    if scale:
        multi_AB_signals_time_dose_kernel_matrix, warn, _ = make_symmetric_matrix_psd(
            normalize(
                KernelCenterer().fit_transform(multi_AB_signals_time_dose_kernel_matrix)
            )
        )
        if warn:
            print(
                "Scaled multi_AB_signals_time_dose_kernel_matrix kernel had to be corrected.\n"
                f"model: {model}"
            )
        if np.max(np.diag(multi_AB_signals_time_dose_kernel_matrix)) > 1.1:
            print(
                "Scaled multi_AB_signals_time_dose_kernel_matrix "
                "kernel had diagonal elements > 1.1.\n"
                f"model: {model}"
            )

    # proof Dimension and rank of kernel matrix
    print("Dimension of final multitask kernel matrix:")
    print(multi_AB_signals_time_dose_kernel_matrix.shape)
    print("Rank of final multitask kernel matrix:")
    print(np.linalg.matrix_rank(multi_AB_signals_time_dose_kernel_matrix))
    print('\n\n')

    return multi_AB_signals_time_dose_kernel_matrix, c_list, info_list


def sort_proteome_data(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """ Sorting

    Input data is sorted by time point to keep same patient over all four time points in order.

    Parameters
    ----------
    data : pd.DataFrame
        Raw proteome data, n x m pd.DataFrame (n = samples as rows, m = features as columns)

    Returns
    -------
    data : pd.DataFrame
        Returns sorted DataFrame
    """
    data.sort_values(by=["TimePointOrder", "Patient"], inplace=True)
    data.reset_index(inplace=True)
    data.drop(columns=['index'], inplace=True)
    return data
