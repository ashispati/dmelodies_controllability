import numpy as np
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import mutual_info_score
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr
from sklearn import svm
from tqdm import tqdm

EVAL_METRIC_DICT = {
    'mig': 'MIG',
    'modularity_score': 'Modularity',
    'SAP_score': 'SAP',
}


def _histogram_discretize(target, num_bins=20):
    """Discretization based on histograms."""
    discretized = np.zeros_like(target)
    for i in range(target.shape[1]):
        discretized[:, i] = np.digitize(
            target[:, i], np.histogram(target[:, i], num_bins)[1][:-1]
        )
    return discretized


def normalize_data(data, mean=None, stddev=None):
    """
    Normalizes the data using a z-score normalization
    Args:
        data: np.array
        mean:
        stddev
    """
    if mean is None:
        mean = np.mean(data, axis=0)
    if stddev is None:
        stddev = np.std(data, axis=0)
    return (data - mean[np.newaxis, :]) / stddev[np.newaxis, :], mean, stddev


def discrete_mutual_info(mus, ys):
    """Compute discrete mutual information.
    Args:
        mus: np.array num_points x num_points
        ys: np.array num_points x num_attributes
    """
    num_codes = mus.shape[1]
    num_attributes = ys.shape[1]
    m = np.zeros([num_codes, num_attributes])
    for i in range(num_codes):
        for j in range(num_attributes):
            m[i, j] = mutual_info_score(ys[:, j], mus[:, i])
    return m


def continuous_mutual_info(mus, ys):
    """Compute continuous mutual information.
    Args:
        mus: np.array num_points x num_points
        ys: np.array num_points x num_attributes
    """
    num_codes = mus.shape[1]
    num_attributes = ys.shape[1]
    m = np.zeros([num_codes, num_attributes])
    for i in tqdm(range(num_attributes)):
        m[:, i] = mutual_info_regression(mus, ys[:, i])
    return m


def discrete_entropy(ys):
    """Compute discrete mutual information.
    Args:
        ys: np.array num_points x num_attributes
    """
    num_factors = ys.shape[1]
    h = np.zeros(num_factors)
    for j in range(num_factors):
        h[j] = mutual_info_score(ys[:, j], ys[:, j])
    return h


def continuous_entropy(ys):
    """Compute continuous mutual entropy
    Args:
        ys: np.array num_points x num_attributes
    """
    num_factors = ys.shape[1]
    h = np.zeros(num_factors)
    for j in tqdm(range(num_factors)):
        h[j] = mutual_info_regression(
            ys[:, j].reshape(-1, 1), ys[:, j]
        )
    return h


def compute_mig(latent_codes, attributes):
    """
    Computes the mutual information gap (MIG) metric
    Args:
        latent_codes: np.array num_points x num_codes
        attributes: np.array num_points x num_attributes
    """
    score_dict = {}
    discretized_codes = _histogram_discretize(latent_codes)
    m = discrete_mutual_info(discretized_codes, attributes)
    entropy = discrete_entropy(attributes)
    sorted_m = np.sort(m, axis=0)[::-1]
    score_dict["mig"] = np.mean(
        np.divide(sorted_m[0, :] - sorted_m[1, :], entropy[:])
    )
    score_dict["mig_factors"] = np.divide(sorted_m[0, :] - sorted_m[1, :], entropy[:])
    # score_dict["factor_idxs"] = list(np.argmax(np.divide(m, entropy), axis=0).astype(np.int32))

    return score_dict


def compute_modularity(latent_codes, attributes):
    """
    Computes the modularity metric
    Args:
        latent_codes: np.array num_points x num_codes
        attributes: np.array num_points x num_attributes
    """
    scores = {}
    # split into train and test sets
    num_datapoints = latent_codes.shape[0]
    num_train = int(0.6 * num_datapoints)
    mus_train = latent_codes[:num_train, :]
    ys_train = attributes[:num_train, :]
    mus_test = latent_codes[num_train:, :]
    ys_test = attributes[num_train:, :]

    # compute modularity
    discretized_mus = _histogram_discretize(mus_train)
    mi = discrete_mutual_info(discretized_mus, ys_train)
    scores["modularity_score"] = _modularity(mi)

    # compute explicitness
    explicitness_score_train = np.zeros([ys_train.shape[1], 1])
    explicitness_score_test = np.zeros([ys_test.shape[1], 1])
    mus_train_norm, mean_mus, stddev_mus = normalize_data(mus_train)
    mus_test_norm, _, _ = normalize_data(mus_test, mean_mus, stddev_mus)
    for i in range(ys_train.shape[1]):
        explicitness_score_train[i], explicitness_score_test[i] = explicitness_per_factor(
            mus_train_norm, ys_train[:, i], mus_test_norm, ys_test[:, i]
        )
    scores["explicitness_score_train"] = np.mean(explicitness_score_train)
    scores["explicitness_score_test"] = np.mean(explicitness_score_test)

    return scores


def explicitness_per_factor(mus_train, y_train, mus_test, y_test):
    """Compute explicitness score for a factor as ROC-AUC of a classifier.
    Args:
    mus_train: Representation for training, (num_points, num_codes)-np array.
    y_train: Ground truth factors for training, (num_points, num_factors)-np
      array.
    mus_test: Representation for testing, (num_points, num_codes)-np array.
    y_test: Ground truth factors for testing, (num_points, num_factors)-np
      array.
    Returns:
    roc_train: ROC-AUC score of the classifier on training data.
    roc_test: ROC-AUC score of the classifier on testing data.
    """
    clf = LogisticRegression(solver='lbfgs', multi_class='auto').fit(mus_train, y_train)
    y_pred_train = clf.predict_proba(mus_train)
    y_pred_test = clf.predict_proba(mus_test)
    mlb = MultiLabelBinarizer()
    roc_train = roc_auc_score(mlb.fit_transform(np.expand_dims(y_train, 1)),
                            y_pred_train)
    roc_test = roc_auc_score(mlb.fit_transform(np.expand_dims(y_test, 1)),
                           y_pred_test)
    return roc_train, roc_test


def _modularity(mutual_information):
    """
    Computes the modularity from mutual information.
    Args:
        mutual_information: np.array num_codes x num_attributes
    """
    squared_mi = np.square(mutual_information)
    max_squared_mi = np.max(squared_mi, axis=1)
    numerator = np.sum(squared_mi, axis=1) - max_squared_mi
    denominator = max_squared_mi * (squared_mi.shape[1] - 1.)
    delta = numerator / denominator
    modularity_score = 1. - delta
    index = (max_squared_mi == 0.)
    modularity_score[index] = 0.
    return np.mean(modularity_score)


def compute_sap_score(latent_codes, attributes):
    """
    Computes the separated attribute predictability (SAP) score
    Args:
        latent_codes: np.array num_points x num_codes
        attributes: np.array num_points x num_attributes
    """
    # split into train and test sets
    num_datapoints = latent_codes.shape[0]
    num_train = int(0.6 * num_datapoints)
    mus_train = latent_codes[:num_train, :]
    ys_train = attributes[:num_train, :]
    mus_test = latent_codes[num_train:, :]
    ys_test = attributes[num_train:, :]

    score_matrix = _compute_score_matrix(mus_train, ys_train, mus_test, ys_test)
    # Score matrix should have shape [num_codes, num_attributes].
    assert score_matrix.shape[0] == latent_codes.shape[1]
    assert score_matrix.shape[1] == attributes.shape[1]

    scores = {
        "SAP_score": _compute_avg_diff_top_two(score_matrix)
    }
    return scores


def _compute_score_matrix(mus, ys, mus_test, ys_test, is_continuous=False):
    """
    Compute score matrix for sap score computation.
    """
    num_latent_codes = mus.shape[1]
    num_attributes = ys.shape[1]
    score_matrix = np.zeros([num_latent_codes, num_attributes])
    for i in tqdm(range(num_latent_codes)):
        for j in range(num_attributes):
            mu_i = mus[:, i]
            y_j = ys[:, j]
            if is_continuous:
                # Attributes are considered continuous.
                cov_mu_i_y_j = np.cov(mu_i, y_j, ddof=1)
                cov_mu_y = cov_mu_i_y_j[0, 1] ** 2
                var_mu = cov_mu_i_y_j[0, 0]
                var_y = cov_mu_i_y_j[1, 1]
                if var_mu > 1e-12:
                    score_matrix[i, j] = cov_mu_y * 1. / (var_mu * var_y)
                else:
                    score_matrix[i, j] = 0.
            else:
                # Attribute is considered discrete.
                mu_i_test = mus_test[:, i]
                y_j_test = ys_test[:, j]
                classifier = svm.LinearSVC(C=0.01, class_weight="balanced")
                classifier.fit(mu_i[:, np.newaxis], y_j)
                pred = classifier.predict(mu_i_test[:, np.newaxis])
                score_matrix[i, j] = np.mean(pred == y_j_test)
    return score_matrix


def _compute_avg_diff_top_two(matrix):
    sorted_matrix = np.sort(matrix, axis=0)
    return np.mean(sorted_matrix[-1, :] - sorted_matrix[-2, :])

