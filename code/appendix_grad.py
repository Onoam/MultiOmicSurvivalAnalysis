from processing import *
from parse_important_features import *
from sksurv.ensemble import GradientBoostingSurvivalAnalysis
import time


class EarlyStoppingMonitor:
    """
    Class used in gradient boost to facilitate early stopping.
    """

    def __init__(self, window_size, max_iter_without_improvement):
        self.window_size = window_size
        self.max_iter_without_improvement = max_iter_without_improvement
        self._best_step = -1

    def __call__(self, iteration, estimator, args):
        # continue training for first self.window_size iterations
        if iteration < self.window_size:
            return False

        # compute average improvement in last self.window_size iterations.
        # oob_improvement_ is the different in negative log partial likelihood
        # between the previous and current iteration.
        start = iteration - self.window_size + 1
        end = iteration + 1
        improvement = np.mean(estimator.oob_improvement_[start:end])

        if improvement > 1e-6:
            self._best_step = iteration
            return False  # continue fitting

        # stop fitting if there was no improvement
        # in last max_iter_without_improvement iterations
        diff = iteration - self._best_step
        return diff >= self.max_iter_without_improvement


def get_all_folds(folds_data_path):
    res = []
    for test_fold in range(5):
        train_folds = [i for i in range(5) if i != test_fold]
        train_idx = get_instances_for_folds(folds_data_path, train_folds)
        while 'nan' in train_idx:
            train_idx.remove('nan')
        test_idx = get_instances_for_folds(folds_data_path, [test_fold])
        while 'nan' in test_idx:
            test_idx.remove('nan')
        res.append((train_idx, test_idx))
    return res


def sort_and_keep_features(df, clinical_df, features):
    """
    :param df: data df
    :param clinical_df: containing censor and event time data
    :param features: features (columns of df) to keep
    :return: df sorted by event time, with only given subset of features
    """
    res_df = get_event_indicator_and_time_df(clinical_df)
    sort_idx = res_df.sort_values(by='event_days_to').index
    df = df.loc[sort_idx]
    df = df[features]
    return df


def prepare_dfs(data_path):
    """
    Basic processing pipeline for dataframes.
    """
    methy_df = pd.read_table(data_path + "/methy").drop_duplicates()
    exp_df = pd.read_table(data_path + "/exp").drop_duplicates()
    mirna_df = pd.read_table(data_path + "/mirna").drop_duplicates()
    clinical_df = pd.read_table(str("{}/clinical".format(data_path)))
    # clean omic data
    methy_df = methy_df.T
    exp_df = transform_and_remove_low_var(exp_df, 0)
    mirna_df = transform_and_remove_low_var(mirna_df, 0)
    return methy_df, exp_df, mirna_df, clinical_df


def train_grad_boost_for_df(df, clinical_df, n_estimators=100, learning_rate=1.0, subsample=1.0, monitor=None, test_size=0.25):
    X_train, X_test, y_train, y_test = create_and_split_X_y(df, clinical_df, test_size=test_size)

    est_cph_tree = GradientBoostingSurvivalAnalysis(
        n_estimators=n_estimators, learning_rate=learning_rate, max_depth=1, subsample=subsample
    )
    t0 = time.time()
    est_cph_tree.fit(X_train, y_train, monitor=monitor)
    t1 = time.time()
    print(f"Total training time was {t1 - t0} seconds")
    cindex = est_cph_tree.score(X_test, y_test)
    print("C-index:", round(cindex, 3))

    return est_cph_tree


def create_and_split_X_y(df, clinical_df, test_size=0.25):
    res_df = get_event_indicator_and_time_df(clinical_df)
    merged_df = pd.merge(df, res_df, left_index=True, right_index=True)
    X, y = get_x_y(merged_df, attr_labels=['dead_bool', 'event_days_to'], pos_label=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    return X_train, X_test, y_train, y_test


def get_important_features(df, grad_boost_model):
    """
    Returns the important features in a gradient boost model, sorted by importance.
    """
    importance = grad_boost_model.feature_importances_
    cols_cols = df.columns[importance != 0]
    sort_idx = (-importance[importance != 0]).argsort()  # argsort descending
    return cols_cols[sort_idx]


def get_important_features_as_df(df, grad_boost_model):
    """
    Parses and returns the important features in a gradient boost model, as well as the cumulative sum of importances.
    """
    importance = grad_boost_model.feature_importances_
    cols_cols = df.columns[importance != 0]
    res_df = (pd.DataFrame.from_dict({'feature_name': cols_cols, 'importance': importance[importance != 0]})
              .sort_values('importance', ascending=False))
    res_df['cumulative_importance'] = res_df['importance'].cumsum()
    return res_df

def chart_learning_rate(df, clinical_df, n_estimators, title):
    """
     Charts learning_rate (gradient boost hyperparameter) vs c-index
     """
    print("Working on " + title)
    X_train, X_test, y_train, y_test = create_and_split_X_y(df, clinical_df)
    scores_cph_tree = {}
    for i in range(1, 11):
        learning_rate = 0.1 * i
        est_cph_tree = GradientBoostingSurvivalAnalysis(
            n_estimators=n_estimators, max_depth=1
        )
        est_cph_tree.set_params(learning_rate=learning_rate)
        est_cph_tree.fit(X_train, y_train)
        scores_cph_tree[learning_rate] = est_cph_tree.score(X_test, y_test)
    x, y = zip(*scores_cph_tree.items())
    plt.plot(x, y)
    plt.xlabel("learning_rate")
    plt.ylabel("concordance index")
    plt.grid(True)
    plt.title(title)
    plt.show()


def chart_n_estimators(df, clinical_df, title):
    """
    Charts n_estimators (gradient boost hyperparameter) vs c-index
    """
    print("Working on " + title)
    X_train, X_test, y_train, y_test = create_and_split_X_y(df, clinical_df)
    scores_cph_tree = {}
    for i in range(16, 24):
        n_estimators = i * 5
        est_cph_tree = GradientBoostingSurvivalAnalysis(
            learning_rate=1.0, max_depth=1
        )
        est_cph_tree.set_params(n_estimators=n_estimators)
        est_cph_tree.fit(X_train, y_train)
        scores_cph_tree[n_estimators] = est_cph_tree.score(X_test, y_test)
    x, y = zip(*scores_cph_tree.items())
    plt.plot(x, y)
    plt.xlabel("n_estimator")
    plt.ylabel("concordance index")
    plt.grid(True)
    plt.title(title)
    plt.show()


def pipeline_train_grad(clinical_df, exp_df, methy_df, mirna_df):
    """
    DEPRECATED
    """
    print("Training for methy")
    methy_grad = train_grad_boost_for_df(methy_df, clinical_df)
    methy_cols = get_important_features(methy_df, methy_grad)
    print(f"Total of {methy_cols.size} important features: {methy_cols}")
    print("Training for exp")
    exp_grad = train_grad_boost_for_df(exp_df, clinical_df)
    exp_cols = get_important_features(exp_df, exp_grad)
    print(f"Total of {exp_cols.size} important features: {exp_cols}")
    print("Training for mirna")
    mirna_grad = train_grad_boost_for_df(mirna_df, clinical_df)
    mirna_cols = get_important_features(mirna_df, mirna_grad)
    print(f"Total of {mirna_cols.size} important features: {mirna_cols}")


def save_features_for_all_cancers():
    """
    Finds (by training a gradient bost model) and saves to CSV the names of the best features for each cancer
    :return:
    """
    for cancer in ["blca", 'brca', 'hnsc', 'laml', 'lgg', 'luad']:
        methy_df, exp_df, mirna_df, clinical_df = prepare_dfs(f"data/{cancer}")
        merged_df = (mirna_df.pipe(pd.merge, exp_df, left_index=True, right_index=True)
                     .pipe(pd.merge, methy_df, left_index=True, right_index=True))
        print(f"\n### Starting work on cancer {cancer} ###")
        n_estimators = 10000 ##TODO changeme to real number
        print(f"Training for methy")
        monitor = EarlyStoppingMonitor(25, 50)
        methy_model = train_grad_boost_for_df(methy_df, clinical_df, n_estimators=n_estimators, learning_rate=0.05,
                                              subsample=0.5, monitor=monitor)

        print(f"Training for exp")
        monitor = EarlyStoppingMonitor(25, 50)
        exp_model = train_grad_boost_for_df(exp_df, clinical_df, n_estimators=n_estimators, learning_rate=0.05,
                                            subsample=0.5, monitor=monitor)

        print(f"Training for mirna")
        monitor = EarlyStoppingMonitor(25, 50)
        mirna_model = train_grad_boost_for_df(mirna_df, clinical_df, n_estimators=n_estimators, learning_rate=0.05,
                                                  subsample=0.5, monitor=monitor)

        print("Training for merged dataframe")
        monitor = EarlyStoppingMonitor(25, 50)
        merged_model = train_grad_boost_for_df(merged_df, clinical_df, n_estimators=n_estimators, learning_rate=0.05,
                                              subsample=0.5, monitor=monitor)

        feature_path = f"./features/{cancer}"

        methy_cols = get_important_features_as_df(methy_df, methy_model)
        methy_cols.to_csv(f"{feature_path}/methy.csv", index=False)
        exp_cols = get_important_features_as_df(exp_df, exp_model)
        exp_cols.to_csv(f"{feature_path}/exp.csv", index=False)
        mirna_cols = get_important_features_as_df(mirna_df, mirna_model)
        mirna_cols.to_csv(f"{feature_path}/mirna.csv", index=False)
        merged_cols = get_important_features_as_df(merged_df, merged_model)
        merged_cols.to_csv(f"{feature_path}/merged.csv", index=False)
        print(f"finished writing CSV to {feature_path}")


def grid_search_grad(df, clinical_df, params: dict, monitor=None, n_jobs=3):
    res_df = get_event_indicator_and_time_df(clinical_df)
    merged_df = pd.merge(df, res_df, left_index=True, right_index=True)
    X, y = get_x_y(merged_df, attr_labels=['dead_bool', 'event_days_to'], pos_label=True)
    estimator = GradientBoostingSurvivalAnalysis(subsample=0.5)
    gcv = GridSearchCV(estimator,
                       param_grid=params,
                       error_score=0.5,
                       n_jobs=n_jobs)
    print("Starting GridSearch")
    t0 = time.time()
    gcv.fit(X, y, monitor=monitor)
    t1 = time.time()
    print(f"Total GridSearch time: {t1-t0} seconds")
    print(f"Best score: {gcv.best_score_}")
    return gcv

from datetime import datetime
if __name__ == "__main__":
    final_c_index = {}
    for cancer in ['brca', "blca", 'hnsc', 'laml', 'lgg', 'luad']:
        folds_data_path = f"./data/folds/{cancer}"
        print(f"\n##############\n#Starting work on {cancer}#\n##############\n")
        methy_df, exp_df, mirna_df, clinical_df = prepare_dfs(f"data/{cancer}")
        concat_df = (mirna_df.pipe(pd.merge, exp_df, left_index=True, right_index=True)
                     .pipe(pd.merge, methy_df, left_index=True, right_index=True))
        methy_features, exp_features, mirna_features, concat_features = read_and_extract_features(f"./features/{cancer}", threshold=1)
        # methy_partial = sort_and_keep_features(methy_df, clinical_df, methy_features)
        # exp_partial = sort_and_keep_features(exp_df, clinical_df, exp_features)
        # mirna_partial = sort_and_keep_features(mirna_df, clinical_df, mirna_features)
        concat_partial = sort_and_keep_features(concat_df, clinical_df, concat_features)
        res_df = get_event_indicator_and_time_df(clinical_df)
        merged_df = pd.merge(concat_partial, res_df, left_index=True, right_index=True)
        cv_split = get_all_folds(folds_data_path)
        for train_idx, test_idx in cv_split:
            cv_results = []
            merged_train = merged_df.loc[train_idx, :]
            merged_test = merged_df.loc[test_idx, :]
            X_train, y_train = get_x_y(merged_train, attr_labels=['dead_bool', 'event_days_to'], pos_label=True)
            X_test, y_test = get_x_y(merged_test,  attr_labels=['dead_bool', 'event_days_to'], pos_label=True)
            model = GradientBoostingSurvivalAnalysis(n_estimators=10000, learning_rate=0.8,
                                                     max_depth=1, subsample=0.5)
            monitor = EarlyStoppingMonitor(25, 50)
            t0 = time.time()
            model.fit(X_train, y_train, monitor=monitor)
            t1 = time.time()
            print(f"Total training time was {t1 - t0} seconds")
            cindex = model.score(X_test, y_test)
            cv_results.append(cindex)
            print("C-index for fold:", round(cindex, 3))
        mean_cv = sum(cv_results) / len(cv_results)
        final_c_index[cancer] = mean_cv
        print(f"cv results for {cancer}: {mean_cv}")
    print("CV results")
    print(final_c_index.items())
