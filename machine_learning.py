from sklearn.svm import SVR, SVC
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, AdaBoostClassifier, AdaBoostRegressor, \
    GradientBoostingClassifier, GradientBoostingRegressor, ExtraTreesClassifier, ExtraTreesRegressor
from tensorflow.contrib import learn
from tensorflow.python.estimator.inputs.inputs import numpy_input_fn
import polyaxon as plx
import tensorflow as tf
import numpy as np
from shutil import rmtree
from os.path import exists
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


def tensorflow_fitting(train_indices, test_indices, input_data, target_data):
    classifier = learn.DNNRegressor(
        feature_columns=[tf.contrib.layers.real_valued_column("", dimension=input_data.shape[1])],
        hidden_units=[2048, 1024, 512, 256, 128, 64])

    classifier.fit(input_fn=lambda : input_fn(input_data[train_indices], target_data[train_indices]), steps=2000)

    error = classifier.evaluate(input_fn=lambda: input_fn(input_data[train_indices], target_data[train_indices]), steps=1)
    error = error['loss']

    training_strategy_score = list(classifier.predict(input_data[train_indices]))
    fitted_strategy_score = list(classifier.predict(input_data[test_indices]))

    fitting_dictionary = {
        'training_strategy_score': training_strategy_score,
        'fitted_strategy_score': fitted_strategy_score,
        'error': error,
    }

    return fitting_dictionary, error


def tensorflow_sequence_fitting(
        output_dir, train_indices, test_indices, X, y, strategy_dictionary, train_steps=1000):
    X = X.astype(np.float32)
    y = y.astype(np.float32)

    X = X[:, :, np.newaxis]
    y = y[:, np.newaxis]

    if exists(output_dir):
        rmtree(output_dir)

    config = {
        'name': 'time_series',
        'output_dir': output_dir,
        'eval_every_n_steps': 5,
        'train_steps': train_steps,
        'train_input_data_config': {
            'input_type': plx.configs.InputDataConfig.NUMPY,
            'pipeline_config': {'name': 'train', 'batch_size': 64, 'num_epochs': 1,
                                'shuffle': False},
            'x': {'x': X[train_indices]},
            'y': y[train_indices]
        },
        'eval_input_data_config': {
            'input_type': plx.configs.InputDataConfig.NUMPY,
            'pipeline_config': {'name': 'eval', 'batch_size': 32, 'num_epochs': 1,
                                'shuffle': False},
            'x': {'x': np.array(X[test_indices])},
            'y': y[test_indices]
        },
        'estimator_config': {'output_dir': output_dir},
        'model_config': {
            'module': 'Regressor',
            'loss_config': {'module': 'mean_squared_error'},
            'eval_metrics_config': [{'module': 'streaming_root_mean_squared_error'},
                                    {'module': 'streaming_mean_absolute_error'}],
            'optimizer_config': {'module': 'adagrad', 'learning_rate': strategy_dictionary['learning_rate']},
            'graph_config': {
                'name': 'regressor',
                'features': ['x'],
                'definition': [
                    (plx.layers.LSTM, {'num_units': strategy_dictionary['num_units'],
                                       'num_layers': strategy_dictionary['num_layers']}),
                    (plx.layers.FullyConnected, {'num_units': strategy_dictionary['output_units']}),
                ]
            }
        }
    }
    experiment_config = plx.configs.ExperimentConfig.read_configs(config)
    xp = plx.experiments.create_experiment(experiment_config)
    xp.continuous_train_and_evaluate()

    train_score = [i['results'] for i in xp.estimator.predict(numpy_input_fn({'x': X[train_indices]}, shuffle=False))]
    predicted = [i['results'] for i in xp.estimator.predict(numpy_input_fn({'x': X[test_indices]}, shuffle=False))]

    error = np.sum((train_score - y[train_indices]) ** 2)

    fitting_dictionary = {
        'training_strategy_score': np.concatenate(train_score, axis=0),
        'fitted_strategy_score': np.concatenate(predicted, axis=0),
        'error': error,
    }

    return fitting_dictionary, fitting_dictionary['error']


def input_fn(input, target):
    return tf.constant(input), tf.constant(target)


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
