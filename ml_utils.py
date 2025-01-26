# imports
import os
import numpy as np
import scipy.sparse as sparse


class TanimotoKernel:
    def __init__(self, sparse_features=True):
        self.sparse_features = sparse_features

    @staticmethod
    def similarity_from_sparse(matrix_a: sparse.csr_matrix, matrix_b: sparse.csr_matrix):
        intersection = matrix_a.dot(matrix_b.transpose()).toarray()
        norm_1 = np.array(matrix_a.multiply(matrix_a).sum(axis=1))
        norm_2 = np.array(matrix_b.multiply(matrix_b).sum(axis=1))
        union = norm_1 + norm_2.T - intersection
        return intersection / union

    @staticmethod
    def similarity_from_dense(matrix_a: np.ndarray, matrix_b: np.ndarray):
        intersection = matrix_a.dot(matrix_b.transpose())
        norm_1 = np.multiply(matrix_a, matrix_a).sum(axis=1)
        norm_2 = np.multiply(matrix_b, matrix_b).sum(axis=1)
        union = np.add.outer(norm_1, norm_2.T) - intersection

        return intersection / union

    def __call__(self, matrix_a, matrix_b):
        if self.sparse_features:
            return self.similarity_from_sparse(matrix_a, matrix_b)
        else:
            raise self.similarity_from_dense(matrix_a, matrix_b)


def tanimoto_from_sparse(matrix_a: sparse.csr_matrix, matrix_b: sparse.csr_matrix):
    DeprecationWarning("Please use TanimotoKernel.sparse_similarity")
    return TanimotoKernel.similarity_from_sparse(matrix_a, matrix_b)


def tanimoto_from_dense(matrix_a: np.ndarray, matrix_b: np.ndarray):
    DeprecationWarning("Please use TanimotoKernel.sparse_similarity")
    return TanimotoKernel.similarity_from_dense(matrix_a, matrix_b)


def maxminpicker(fp_list, ntopick, seed=None):
    from rdkit import SimDivFilters
    mmp = SimDivFilters.MaxMinPicker()
    n_to_pick = round(ntopick * len(fp_list))
    picks = mmp.LazyBitVectorPick(fp_list, len(fp_list), n_to_pick, seed=seed)
    return list(picks)


def create_directory(path: str, verbose: bool = True):
    if not os.path.exists(path):

        if len(path.split("/")) <= 2:
            os.mkdir(path)
        else:
            os.makedirs(path)
        if verbose:
            print(f"Created new directory '{path}'")
    return path
#########################################################

from abc import ABC, abstractmethod

import numpy as np
import scipy as sc

from numpy.linalg import inv, det

class SearchModel(ABC):

    @abstractmethod
    def sample(self, n_sample: int, seed: int = None):
        raise NotImplementedError

    @abstractmethod
    def fit(self, X_nxp: np.array, weights: np.array = None):
        raise NotImplementedError

    @abstractmethod
    def loglikelihood(self, X_nxp: np.array) -> np.array:
        raise NotImplementedError

    @abstractmethod
    def get_parameters(self):
        raise NotImplementedError

    @abstractmethod
    def set_parameters(self, value: tuple):
        raise NotImplementedError

    parameters = property(get_parameters, set_parameters)


class MultivariateGaussian(SearchModel):

    def __init__(self, dim: int = 10):
        self.dim = dim
        self.set_parameters((np.zeros([dim]), np.eye(dim)))

    def get_initialization_kwargs(self):
        kwargs = {
            "dim": self.dim,
        }
        return kwargs

    def sample(self, n_sample: int, seed: int = None) -> np.array:
        np.random.seed(seed)
        X_nxp = np.random.multivariate_normal(self._parameters[0], self._parameters[1], size=n_sample)
        return X_nxp

    def fit(self, X_nxp: np.array, weights: np.array = None):
        if weights is None:
            weights = np.ones([X_nxp.shape[0]])
        weights_nx1 = np.reshape(weights, (X_nxp.shape[0], 1))
        
        Xweighted_nxp = weights_nx1 * X_nxp
        mean_p = np.sum(Xweighted_nxp, axis=0, keepdims=False) / np.sum(weights_nx1)
        Xcentered_nxp = X_nxp - mean_p[None, :]
    
        cov_pxp = np.dot(Xcentered_nxp.T, weights_nx1 * Xcentered_nxp) / np.sum(weights_nx1)
        self._parameters = (mean_p, cov_pxp)

    def loglikelihood(self, X_nxp: np.array) -> np.array:
        #print(np.diag(self._parameters[1]))            
        
        #print("#####det:",det(self._parameters[1]))
        ll = lambda X_p: sc.stats.multivariate_normal.logpdf(X_p, mean=self._parameters[0], cov=self._parameters[1])
        
        try:
            ll_n = np.array([ll(X_p) for X_p in X_nxp])
        except np.linalg.LinAlgError:
            print("Singular covariance matrix. Cannot evaluate log-likelihood.")
            raise np.linalg.LinAlgError
        return ll_n

    def get_parameters(self):
        return self._parameters[0], self._parameters[1]

    def set_parameters(self, value: tuple):
        if len(value) != 2:
            raise ValueError("Need to supply both the mean and covariance parameters.")
        if value[1].shape != (value[0].size, value[0].size):
            raise ValueError("Shapes of mean and covariance parameters do not match.")
        self._parameters = value

    parameters = property(get_parameters, set_parameters)

    def save(self, filename: str):
        print("Saving to {} using np.savez.".format(filename))
        parameters = self.get_parameters()
        np.savez(filename, mean_d=parameters[0], cov_dxd=parameters[1])

    def load(self, filename: str):
        d = np.load(filename)
        if 'mean_d' not in d or 'cov_dxd' not in d:
            raise ValueError('File {} is missing either the mean or covariance parameter.'.format(filename))
        self.set_parameters((d["mean_d"], d["cov_dxd"]))

def ess(lrs_xn):
    # lrs_xn = lrs_xn / np.sum(lrs_xn, axis=-1, keepdims=True)
    denom = np.sum(np.square(lrs_xn), axis=-1)
    numer = np.square(np.sum(lrs_xn, axis=-1))
    result = np.zeros(numer.shape)
    if denom.size == 1:
        print("Computing ESS: {} / {}".format(numer, denom))
    return 0 if denom == 0  else numer / denom
    result[denom > 0] = numer[denom > 0] / denom[denom > 0]
    return result

def get_data_below_percentile(X_nxp: np.array, y_n: np.array, percentile: float, n_sample: int = None, seed: int = None):
    perc = np.percentile(y_n, percentile)
    idx = np.where(y_n <= perc)[0]
    print("Max label in training data: {:.1f}. 80-th percentile label: {:.1f}".format(np.max(y_n), perc))
    if n_sample is not None and n_sample < idx.size:
        np.random.seed(seed)
        idx = np.random.choice(idx, size=n_sample, replace=False)
    Xbelow_nxm = X_nxp[idx]
    ybelow_n = y_n[idx]
    return Xbelow_nxm, ybelow_n, idx

def get_promising_candidates(oracle_m: np.array, gt_m: np.array, percentile: float = 80):
    oracle_percentile = np.percentile(oracle_m, percentile)
    candidate_idx = np.where(oracle_m >= oracle_percentile)[0]
    return oracle_m[candidate_idx], gt_m[candidate_idx], oracle_percentile

def evaluate_top_candidates(o_m: np.array, gt_m: np.array, gt0_n: np.array, percentile: float):
    o_cand, gt_cand, o_perc = get_promising_candidates(o_m, gt_m, percentile=percentile)
    rho_and_p = sc.stats.spearmanr(gt_m, o_m)
    rmse = np.sqrt(np.mean(np.square(gt_m - o_m)))
    pci = 100 * np.sum(gt_cand > np.max(gt0_n)) / float(gt_cand.size)
    return o_perc, np.median(gt_cand), np.max(gt_cand), pci, rho_and_p, rmse

def score_top_candidates(o_txm: np.array, gt_txm: np.array, gt0_n: np.array,
                         oaf_txm: np.array, gtaf_txm: np.array, gt0af_n: np.array, percentile: float):
    scores_tx = np.zeros([o_txm.shape[0], 5])
    scoresaf_tx = np.zeros([oaf_txm.shape[0], 5])
    operc_t = np.zeros([o_txm.shape[0]])
    oafperc_t = np.zeros([o_txm.shape[0]])
    for t in range(o_txm.shape[0]):
        o_perc, gt_med, gt_max, pci, rho_and_p, rmse = evaluate_top_candidates(o_txm[t], gt_txm[t], gt0_n, percentile)
        scores_tx[t] = np.array([gt_med, gt_max, pci, rho_and_p[0], rmse])
        operc_t[t] = o_perc
    for t in range(oaf_txm.shape[0]):
        oaf_perc, gtaf_med, gtaf_max, pciaf, rhoaf_and_p, rmseaf = evaluate_top_candidates(
            oaf_txm[t], gtaf_txm[t], gt0af_n, percentile)
        scoresaf_tx[t] = np.array([gtaf_med, gtaf_max, pciaf, rhoaf_and_p[0], rmseaf])
        oafperc_t[t] = oaf_perc
    t_max = np.argmax(operc_t)
    taf_max = np.argmax(oafperc_t)
    return scores_tx[t_max], scoresaf_tx[taf_max], t_max, taf_max

def compare_af(scores_trx: np.array, scoresaf_trx: np.array):
    mean_diffs = np.mean(scoresaf_trx - scores_trx, axis=0)
    p_values = [sc.stats.wilcoxon(scoresaf_trx[:, i], scores_trx[:, i])[1] for i in range(scores_trx.shape[1])]
    formatted_scores = ["{:.2f}".format(val) for val in np.mean(scores_trx, axis=0)]
    formatted_scoresaf = ["{:.2f}".format(val) for val in np.mean(scoresaf_trx, axis=0)]
    formatted_diffs = ["{:.2f}".format(val) for val in mean_diffs]
    formatted_p = ["{:.4f}".format(val) for val in p_values]
    print("            GT Median  |  GT Max  |  PCI  |  rho  |  RMSE")
    print("Original    {:<14}{:<11}{:<8}{:<8}{:<6}".format(*formatted_scores))
    print("Autofocused {:<14}{:<11}{:<8}{:<8}{:<6}".format(*formatted_scoresaf))
    print("Mean Diff.  {:<14}{:<11}{:<8}{:<8}{:<6}".format(*formatted_diffs))
    print("p-value     {:<14}{:<11}{:<8}{:<8}{:<6}".format(*formatted_p))

def iw_rmse(y1_n: np.array, y2_n: np.array, w_n: np.array = None, self_normalize=False):
    if w_n is None:
        w_n = np.ones((y1_n.size))
    if self_normalize:
        rmse = np.sqrt(np.sum(w_n * np.square(y1_n - y2_n)) / np.sum(w_n))
    else:
        rmse = np.sqrt(np.mean(w_n * np.square(y1_n - y2_n)))
    return rmse

def rmse(x1_n: np.array, x2_n: np.array):
    return np.sqrt(np.mean(np.square(x1_n - x2_n)))


