import numpy as np
from random import randint, choice
from trading_strategy_fitting import fit_strategy, offset_scan_validation, tic
from strategy_evaluation import output_strategy_results


def random_search(strategy_dictionary_local, n_iterations):
    toc = tic()

    counter = 0
    error = -1e5
    while counter < n_iterations:
        counter += 1
        fitting_dictionary, data_to_predict = random_search_iteration(strategy_dictionary_local)
        error_loop = fitting_dictionary['error']

        if error_loop > error and fitting_dictionary['n_trades'] != 0:
            error = error_loop
            strategy_dictionary_local_optimum = strategy_dictionary_local
            fitting_dictionary_optimum = fitting_dictionary

    underlined_output('Best strategy fit')
    output_strategy_results(strategy_dictionary_local_optimum, fitting_dictionary_optimum, data_to_predict, toc)

    return strategy_dictionary


def random_search_iteration(strategy_dictionary_local):
    strategy_dictionary_local = randomise_dictionary_inputs(strategy_dictionary_local)

    data_length = strategy_dictionary_local['n_days'] * 86400 / strategy_dictionary_local['candle_size']

    fitting_dictionary, data_to_predict = fit_strategy(strategy_dictionary_local)

    return fitting_dictionary, data_to_predict


def randomise_dictionary_inputs(strategy_dictionary_local):
    strategy_dictionary_local['ml_mode'] = choice(['adaboost', 'randomforest', 'gradientboosting', 'extratreesfitting']) #'svm'
    strategy_dictionary_local['regression_mode'] = choice(['regression', 'classification'])
    strategy_dictionary_local['preprocessing'] = choice(['PCA', 'FastICA', 'None'])
    return strategy_dictionary_local


def underlined_output(string):
    print string
    print '----------------------'
    print '\n'

if __name__ == '__main__':
    strategy_dictionary = {
        'trading_currencies': ['ETH', 'BTC'],
        'ticker_1': 'BTC_ETH',
        'ticker_2': 'USDT_BTC',
        'candle_size': 300,
        'n_days': 40,
        'offset': 0,
        'bid_ask_spread': 0.001,
        'transaction_fee': 0.0025,
        'train_test_ratio': 0.5,
        'output_flag': True,
        'plot_flag': False,
        'ml_iterations': 100,
        'target_score': 'idealstrategy',
        'windows': [1, 5, 10, 50, 100],
        'web_flag': 'False',
        'filename1': "USDT_BTC.csv",
        'filename2': "BTC_ETH.csv"
    }

    search_iterations = 50

    strategy_dictionary = random_search(strategy_dictionary, search_iterations)

    underlined_output('Offset validation')
    offsets = np.linspace(0, 300, 5)

    offset_scan_validation(strategy_dictionary, offsets)

    print strategy_dictionary
