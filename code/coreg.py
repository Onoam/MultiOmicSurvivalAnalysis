import pickle
import cvxpy as cp
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator
from sksurv.metrics import concordance_index_censored
var_threshold = 0


class CoxCoRegularized(BaseEstimator):
    """
    Based on a co-regularized cox model shown in:
    https://www.researchgate.net/publication/343115618_Supervised_graph_clustering_for_cancer_subtyping_based_on_survival_analysis_and_integration_of_multi-omic_tumor_data
    """

    def __init__(self, lambda_val=1, eta_val=1, eta_2_val=0, prob=None,
                 m_1=70, m_2=70, m_3=70):
        """
        :param lambda_val: tuning parameter for co-regularization penalty
        :param eta_val: tuning parameter for l1 regularization
        :param eta_2_val: tuning parameter for l2 regularization
        :param prob: CVXPY.Problem
        :param m_1: number of features in first omic
        :param m_2: number of features in second omic
        :param m_3: number of features in third omic
        """
        self.lambda_val = lambda_val
        self.eta_val = eta_val
        self.eta_2_val = eta_2_val
        self.prob = prob
        self.m_1 = m_1
        self.m_2 = m_2
        self.m_3 = m_3

    def fit(self, X, y):
        """
        Solves the cvxpy problem, setting self.prob to the solved problem.
        The format of X and y is required by the BaseEstimator interface to function with other objects
        e.g. GridSearchCV
        :param X: concatenated matrix of dim (n_samples,features_omic_1 + features_omic_2 + features_omic_3)
        :param y: pd.Series of 2-tuples, of dim (n_samples,)
        """
        X_1, X_2, X_3 = self.get_data_matrices(X)
        times, delta = self.get_times_delta(y)
        self.prob = cox_multi_omic(X_1, X_2, X_3, times, delta, self.lambda_val, self.eta_val, self.eta_2_val)
        self.prob.solve()

    def get_times_delta(self, y):
        """
        :param y: pd.Series of 2-tuples, of dim (n_samples,)
        :return: Two pd.Series, one for the time of observation, one for an indicator, delta, with delta[i]=0 indicating
        censoring, and delta[i]=1 indicating an event (death)
        """
        times = y.apply(lambda x: x[0])
        delta = y.apply(lambda x: x[1])
        y_index = y.apply(lambda x: x[2])
        times.index = y_index
        delta.index = y_index
        return times, delta

    def get_data_matrices(self, X):
        """
        Parses a concatenated matrix into three separate matrices, one for each omic.
        :param X: concatenated matrix of dim (n_samples,features_omic_1 + features_omic_2 + features_omic_3)
        :return: 3 matrices, each of dimensions n_samples, features_omic_i,
                                           n_samples, features_omic_2,
                                           n_samples, features_omic_3
        """
        first_break = self.m_1
        second_break = self.m_1 + self.m_2
        X_1, X_2, X_3 = X[:, :first_break], X[:, first_break: second_break], X[:, second_break:]
        return X_1, X_2, X_3

    def predict(self, X):
        """
        :param X: matrix of shape [n_samples X m_1+m_2+m_3]
        :return: prediction of times values in the same [n_samples,]
        """
        X_1, X_2, X_3 = self.get_data_matrices(X)
        return predict_multi_omic(self.prob, X_1, X_2, X_3)

    def score(self, X, y):
        X_1, X_2, X_3 = self.get_data_matrices(X)
        times, delta = self.get_times_delta(y)
        prediction = predict_multi_omic(self.prob, X_1, X_2, X_3)
        try:
            ci = concordance_index_censored(delta.astype(bool), times, prediction)[0]
        except ValueError:  # Happened during model-tuning if fold contained only censored data.
            ci = 0.5
        return ci


def parse_important_features_from_count(features_df, count):
    count = min(count, features_df.shape[0] - 1)
    # print(f"The cumulative importance for {count} features is: {features_df['cumulative_importance'][count]}")
    return features_df.iloc[:count, 0].to_numpy()


def transform_and_remove_low_var(df, var_threshold=1e-3):
    #assumes data_frames are as read from source (columns are observations, rows are features)
    df = df.transform(lambda x: np.log(1+x)/np.log(2))
    return df[df.var(axis='columns') > var_threshold].T


def coregularize(a, b, c, x_1, x_2, x_3):
    return cp.sum_squares(a @ x_1 - b @ x_2) + cp.sum_squares(a @ x_1 - c @ x_3) + cp.sum_squares(b @ x_2 - c @ x_3)


def cox_partial_likelihood(X, w, times, delta):
    n_samples = len(times)
    linear_predictor = X @ w
    risked_calc = np.empty(n_samples, dtype=object)
    for i in range(n_samples):
        risked_calc[i] = cp.log_sum_exp(linear_predictor[times >= times[i]])
    return cp.sum(cp.multiply(-delta, linear_predictor) + cp.multiply(delta, cp.hstack(risked_calc)))


def cox_multi_omic(X_1, X_2, X_3, times, delta, lambd_val=1, eta_val=1, eta_2_val=1):
    """
    :param X_1: matrix for first omic (methy)
    :param X_2: matrix for second omic (mirna)
    :param X_3: matrix for third omic (exp)
    :param times: list-like of event times
    :param delta: list-like of events, where 0 is censored, and 1 is death
    :param lambd_val: tuning parameter for co-regularization penalty
    :param eta_val: tuning parameter for l1 regularization
    :param eta_2_val: tuning parameter for l2 regularization
    :return:
    """
    # add decision for lambda, eta
    lambd = cp.Parameter(nonneg=True)
    eta = cp.Parameter(nonneg=True)
    eta_2 = cp.Parameter(nonneg=True)
    w_1 = cp.Variable(X_1.shape[1], name='first')
    w_2 = cp.Variable(X_2.shape[1], name='second')
    w_3 = cp.Variable(X_3.shape[1], name='third')
    X = [X_1, X_2, X_3]
    w = [w_1, w_2, w_3]
    loss = cp.sum([cox_partial_likelihood(X[i], w[i], times, delta) for i in range(len(X))])
    coreg = coregularize(X_1, X_2, X_3, w_1, w_2, w_3)
    regularization = cp.sum([cp.norm1(w[i]) for i in range(len(w))])
    reg_l2 = cp.sum([cp.norm2(w[i]) for i in range(len(w))])
    lambd.value = lambd_val
    eta.value = eta_val
    eta_2.value = eta_2_val
    obj = cp.Minimize(loss + lambd * coreg + eta * regularization + eta_2 * reg_l2)
    prob = cp.Problem(obj)
    return prob


def predict_multi_omic(prob, X_1, X_2, X_3):
    """
    :param prob: CVXPY.Problem
    :param X_1: matrix for first omic (methy)
    :param X_2: matrix for second omic (mirna)
    :param X_3: matrix for third omic (exp)
    :return: result of prediction: sum of data_matrix @ problem_optimum_values
    """
    opt_vars = prob.variables()
    first_var = opt_vars[0]
    assert first_var.name() == 'first'
    second_var = opt_vars[1]
    assert second_var.name() == 'second'
    third_var = opt_vars[2]
    assert third_var.name() == 'third'
    w_1 = first_var.value.reshape(-1, 1)
    w_2 = second_var.value.reshape(-1, 1)
    w_3 = third_var.value.reshape(-1, 1)
    # If one of our matrices is None, we predict without that part, i.e. setting that part of prediction to 0.
    if X_1 is None:
        pred1 = 0
    else:
        pred1 = X_1 @ w_1
    if X_2 is None:
        pred2 = 0
    else:
        pred2 = X_2 @ w_2
    if X_3 is None:
        pred3 = 0
    else:
        pred3 = X_3 @ w_3
    return (pred1 + pred2 + pred3).reshape(-1)


def predict_main(cancer: str, data_path: str, out_path: str):
    # load model
    model_path = f"./task_2/{cancer}.model"
    model = pickle.load(open(model_path, 'rb'))

    # read features (hard coded?)
    exp_features = pd.read_csv(f"task_2/features/{cancer}/exp.csv")
    features = parse_important_features_from_count(exp_features, count=70)

    # read data and keep important features(exp only)
    exp_df = pd.read_table(data_path) # Modified 2021/03/21 to remove the call to drop_duplicates()
    exp_df = transform_and_remove_low_var(exp_df, -1) # Modified 2021/03/21 ...(exp_df, 0) to ...(exp_df, -1)
    exp_df = exp_df[features]

    # predict
    y = predict_multi_omic(model.prob, None, None, exp_df.to_numpy())
    order = np.array([(y >= val).sum() for val in y])

    # save data (format: patient_id \t ordering\n patined_id \t ordering...)
    res_df = pd.DataFrame(data={'id': exp_df.index, 'order': order})
    res_df.to_csv(out_path, sep="\t", index=False, header=False)
