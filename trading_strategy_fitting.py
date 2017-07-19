import numpy as np
from time import time
from matplotlib import pyplot as plt
from data_input_processing import Data, train_test_indices, generate_training_variables
from strategy_evaluation import post_process_training_results, output_strategy_results, convert_strategy_score_to_profit
from machine_learning import random_forest_fitting, svm_fitting, adaboost_fitting, gradient_boosting_fitting,\
    extra_trees_fitting, tensorflow_fitting, tensorflow_sequence_fitting

SEC_IN_DAY = 86400


def meta_fitting(data_to_predict_local, data_input_2, strategy_dictionary):
    fitting_inputs, fitting_targets = input_processing(data_to_predict_local, data_input_2, strategy_dictionary)
    error = []
    train_indices, test_indices = train_test_indices(fitting_inputs, strategy_dictionary['train_test_ratio'])
    if strategy_dictionary['ml_mode'] == 'svm':
        fitting_dictionary, error = svm_fitting(
            fitting_inputs, fitting_targets, train_indices, test_indices, strategy_dictionary)

    elif strategy_dictionary['ml_mode'] == 'randomforest':
        fitting_dictionary, error = random_forest_fitting(
            fitting_inputs, fitting_targets, train_indices, test_indices, strategy_dictionary)

    elif strategy_dictionary['ml_mode'] == 'adaboost':
        fitting_dictionary, error = adaboost_fitting(
            fitting_inputs, fitting_targets, train_indices, test_indices, strategy_dictionary)

    elif strategy_dictionary['ml_mode'] == 'gradientboosting':
        fitting_dictionary, error = gradient_boosting_fitting(
            fitting_inputs, fitting_targets, train_indices, test_indices, strategy_dictionary)

    elif strategy_dictionary['ml_mode'] == 'extratreesfitting':
        fitting_dictionary, error = extra_trees_fitting(
            fitting_inputs, fitting_targets, train_indices, test_indices, strategy_dictionary)

    fitting_dictionary['train_indices'] = train_indices
    fitting_dictionary['test_indices'] = test_indices
    fitting_dictionary['error'] = error

    return fitting_dictionary


def input_processing(data_to_predict_local, data_input_2, strategy_dictionary):
    fitting_inputs, fitting_targets = generate_training_variables(data_to_predict_local, strategy_dictionary)
    fitting_inputs_2, fitting_targets_2 = generate_training_variables(data_input_2, strategy_dictionary)
    fitting_inputs, fitting_inputs_2 = trim_inputs(fitting_inputs, fitting_inputs_2)
    fitting_inputs = np.hstack((fitting_inputs, fitting_inputs_2))

    return fitting_inputs, fitting_targets


def trim_inputs(fitting_inputs, fitting_inputs_2):
    length_1 = len(fitting_inputs)
    length_2 = len(fitting_inputs_2)

    min_length = np.minimum(length_1, length_2)

    fitting_inputs = fitting_inputs[-min_length:]
    fitting_inputs_2 = fitting_inputs_2[-min_length:]
    return fitting_inputs, fitting_inputs_2


def tic():
    t = time()
    return lambda: (time() - t)


def retrieve_data(ticker, strategy_dictionary):
    end = time() - strategy_dictionary['offset'] * SEC_IN_DAY

    start = end - SEC_IN_DAY * strategy_dictionary['n_days']

    data_local = None
    while data_local is None:
        try:
            data_local = Data(ticker, start, end, strategy_dictionary['candle_size'])
        except:
            pass

    data_local.normalise_data()

    return data_local


def offset_scan_validation(strategy_dictionary, offsets):
    strategy_dictionary['plot_flag'] = False
    strategy_dictionary['ouput_flag'] = True

    for offset in offsets:
        strategy_dictionary['offset'] = offset
        fit_strategy(strategy_dictionary)


def tensorflow_offset_scan_validation(strategy_dictionary, offsets):
    strategy_dictionary['plot_flag'] = False
    strategy_dictionary['ouput_flag'] = True

    for offset in offsets:
        strategy_dictionary['offset'] = offset
        fit_tensorflow(strategy_dictionary)


def fit_strategy(strategy_dictionary):
    toc = tic()

    data_to_predict = retrieve_data(strategy_dictionary['ticker_1'], strategy_dictionary)
    data_2 = retrieve_data(strategy_dictionary['ticker_2'], strategy_dictionary)

    fitting_dictionary = meta_fitting(data_to_predict, data_2, strategy_dictionary)

    fitting_dictionary = post_process_training_results(strategy_dictionary, fitting_dictionary, data_to_predict)

    output_strategy_results(strategy_dictionary, fitting_dictionary, data_to_predict, toc)

    return fitting_dictionary, data_to_predict


def fit_tensorflow(strategy_dictionary):
    toc = tic()

    data_to_predict = retrieve_data(strategy_dictionary['ticker_1'], strategy_dictionary)
    data_input_2 = retrieve_data(strategy_dictionary['ticker_2'], strategy_dictionary)

    fitting_inputs, fitting_targets = input_processing(data_to_predict, data_input_2, strategy_dictionary)
    train_indices, test_indices = train_test_indices(fitting_inputs, strategy_dictionary['train_test_ratio'])

    if strategy_dictionary['sequence_flag']:
        fitting_dictionary, error = tensorflow_sequence_fitting(
            '/home/thomas/test',train_indices, test_indices, fitting_inputs, fitting_targets, strategy_dictionary)

    else:
        fitting_dictionary, error = tensorflow_fitting(train_indices, test_indices, fitting_inputs, fitting_targets)

    fitting_dictionary['train_indices'] = train_indices
    fitting_dictionary['test_indices'] = test_indices

    fitting_dictionary = post_process_training_results(strategy_dictionary, fitting_dictionary, data_to_predict)

    output_strategy_results(strategy_dictionary, fitting_dictionary, data_to_predict, toc)
    return fitting_dictionary, data_to_predict, error

