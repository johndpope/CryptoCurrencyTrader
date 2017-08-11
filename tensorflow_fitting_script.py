import numpy as np
import random
from trading_strategy_fitting import tic, tensorflow_offset_scan_validation, fit_tensorflow,\
    underlined_output
from strategy_evaluation import output_strategy_results


def random_search(strategy_dictionary_local, n_iterations):
    toc = tic()
    counter = 0
    error = 1e10

    while counter < n_iterations:
        counter += 1

        if strategy_dictionary['sequence_flag']:
            strategy_dictionary_local = randomise_sequence_dictionary_inputs(strategy_dictionary_local)
        else:
            strategy_dictionary_local = randomise_dictionary_inputs(strategy_dictionary_local)

        fitting_dictionary, data_to_predict, error_loop, profit_factor = fit_tensorflow(strategy_dictionary_local)

        if error_loop < error:
            error = error_loop
            strategy_dictionary_optimum = strategy_dictionary_local
            fitting_dictionary_optimum = fitting_dictionary

    underlined_output('Best strategy fit')
    output_strategy_results(strategy_dictionary_optimum, fitting_dictionary_optimum, data_to_predict, toc)

    return strategy_dictionary_optimum


def randomise_dictionary_inputs(strategy_dictionary):
    strategy_dictionary['learning_rate'] = 10 ** np.random.uniform(-5, -1)
    strategy_dictionary['keep_prob'] = np.random.uniform(0.2, 0.8)
    return strategy_dictionary


def randomise_sequence_dictionary_inputs(strategy_dictionary):
    strategy_dictionary['learning_rate'] = 10 ** np.random.uniform(-5, -1)
    strategy_dictionary['num_layers'] = random.randint(1, 30)
    strategy_dictionary['num_units'] = random.randint(5, 200)
    return strategy_dictionary


if __name__ == '__main__':
    strategy_dictionary = {
        'trading_currencies': ['ETH', 'BTC'],
        'ticker_1': 'BTC_ETH',
        'ticker_2': 'USDT_BTC',
        'candle_size': 1800,
        'n_days': 40,
        'offset': 0,
        'bid_ask_spread': 0.001,
        'transaction_fee': 0.0025,
        'train_test_ratio': 0.75,
        'output_flag': True,
        'plot_flag': False,
        'target_score': 'idealstrategy',
        'windows': [1, 5, 10, 50, 100],
        'regression_mode': 'regression',
        'preprocessing': 'None',
        'ml_mode': 'tensorflow',
        'sequence_flag': False,
        'output_units': 1,
        'web_flag': False,
        'filename1': "USDT_BTC.csv",
        'filename2': "BTC_ETH.csv"
    }

    search_iterations = 10

    strategy_dictionary = random_search(strategy_dictionary, search_iterations)

    underlined_output('Offset validation')
    offsets = np.linspace(0, 300, 10)

    tensorflow_offset_scan_validation(strategy_dictionary, offsets)

    print strategy_dictionary
    