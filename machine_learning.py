from sklearn.svm import SVR, SVC
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, AdaBoostClassifier, AdaBoostRegressor, \
    GradientBoostingClassifier, GradientBoostingRegressor, ExtraTreesClassifier, ExtraTreesRegressor
import numpy as np
from scipy import stats


def svm_fitting(input_data, target_data, train_indices, test_indices, strategy_dictionary):
    param_set = {'kernel': ['rbf'],
                 'C': stats.expon(scale=0.01),
                 'gamma': stats.expon(scale=0.01)}

    if strategy_dictionary['regression_mode'] == 'regression':
        clf = SVR()
    elif strategy_dictionary['regression_mode'] == 'classification':
        clf = SVC()

    return random_search(clf, param_set, train_indices, test_indices, input_data, target_data)


def random_forest_fitting(input_data, target_data, train_indices, test_indices, strategy_dictionary):
    if strategy_dictionary['regression_mode'] == 'regression':
        clf = RandomForestRegressor(n_jobs=-1)
    elif strategy_dictionary['regression_mode'] == 'classification':
        clf = RandomForestClassifier(n_jobs=-1)

    param_set = {'n_estimators': range(2, 1000),
                 'max_depth': [1, 2, 3, None],
                 'max_features': range(1, 5)}

    return random_search(clf, param_set, train_indices, test_indices, input_data, target_data)


def adaboost_fitting(input_data, target_data, train_indices, test_indices, strategy_dictionary):
    if strategy_dictionary['regression_mode'] == 'regression':
        clf = AdaBoostRegressor()
    elif strategy_dictionary['regression_mode'] == 'classification':
        clf = AdaBoostClassifier()

    param_set = {'learning_rate': [0.1, 0.25, 0.5, 1.0], #stats.expon(scale=1)
                 "n_estimators": range(2, 1000),
                  }

    return random_search(clf, param_set, train_indices, test_indices, input_data, target_data)


def gradient_boosting_fitting(input_data, target_data, train_indices, test_indices, strategy_dictionary):
    if strategy_dictionary['regression_mode'] == 'regression':
        clf = GradientBoostingRegressor()
    elif strategy_dictionary['regression_mode'] == 'classification':
        clf = GradientBoostingClassifier()

    param_set = {'n_estimators': range(2, 1000),
                 'max_depth': [1, 2, 3, None],
                 'learning_rate': [0.1, 0.25, 0.5, 1.0],
                 }

    return random_search(clf, param_set, train_indices, test_indices, input_data, target_data)


def extra_trees_fitting(input_data, target_data, train_indices, test_indices, strategy_dictionary):
    if strategy_dictionary['regression_mode'] == 'regression':
        clf = ExtraTreesRegressor(n_jobs=-1)
    elif strategy_dictionary['regression_mode'] == 'classification':
        clf = ExtraTreesClassifier(n_jobs=-1)

    param_set = {'n_estimators': range(2, 1000),
                 'max_depth': [1, 2, 3, None],
                 }

    return random_search(clf, param_set, train_indices, test_indices, input_data, target_data)


def random_search(clf, param_set, train_indices, test_indices, input_data, target_data):
    random_search_local = RandomizedSearchCV(clf, param_distributions=param_set, cv=5, n_jobs=-1)

    random_search_local.fit(input_data[train_indices], target_data[train_indices])

    error = np.mean(
        np.mean(cross_val_score(random_search_local, input_data[train_indices], target_data[train_indices])))

    training_strategy_score = random_search_local.predict(input_data[train_indices])
    if len(test_indices) != 0:
        fitted_strategy_score = random_search_local.predict(input_data[test_indices])
    else:
        fitted_strategy_score = []

    fitting_dictionary = {
        'training_strategy_score': training_strategy_score,
        'fitted_strategy_score': fitted_strategy_score,
        'model': random_search,
    }
    return fitting_dictionary, error
