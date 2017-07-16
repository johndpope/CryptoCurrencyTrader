import threading
import numpy as np
from trading_strategy_fitting import fit_strategy, retrieve_data, input_processing
from poloniex_API import poloniex
from API_settings import API_secret, API_key


def daily_fit(strategy_dictionary_local, counter_local):
    print 'Daily fit.'
    fitting_dictionary, data_to_predict, data_2 = fit_strategy(
        strategy_dictionary_local)

    candle_update(fitting_dictionary, data_to_predict, data_2, strategy_dictionary_local, counter_local)

    threading.Timer(86400.0, daily_fit, args=(strategy_dictionary_local, counter_local)).start()
    
    return 0


def candle_update(fitting_dictionary, data_to_predict, data_2, strategy_dictionary_local, counter_local):
    print '15 minute update.'

    strategy_dictionary_local['n_days'] = 4 * 1800 / 86400.0
    data_1_temp = retrieve_data(strategy_dictionary_local['ticker_1'], strategy_dictionary_local)
    data_2_temp = retrieve_data(strategy_dictionary_local['ticker_2'], strategy_dictionary_local)
    data_to_predict.extend_candle(data_1_temp)
    data_2.extend_candle(data_2_temp)

    fitting_inputs, fitting_targets = input_processing(data_to_predict, data_2, fitting_dictionary['window_1'],
                                                       fitting_dictionary['window_2'], strategy_dictionary_local)

    strategy_score = fitting_dictionary['model'].predict(fitting_inputs)

    poloniex_session = poloniex(API_key, API_secret)

    update_currency_position(poloniex_session, strategy_dictionary_local, fitting_dictionary, strategy_score)
    print strategy_score[-5:]
    print fitting_dictionary['buy_threshold']
    print fitting_dictionary['sell_threshold']
    
    counter_local += 1
    
    if counter >= 48:
        return 0

    threading.Timer(900.0, candle_update, args=(fitting_dictionary, data_to_predict, data_2, strategy_dictionary_local,
                                                counter_local)).start()


def update_currency_position(poloniex_session, strategy_dictionary_local, fitting_dictionary, strategy_score):
    balance = poloniex_session.returnBalances()
    ticker = poloniex_session.returnTicker()

    if balance[strategy_dictionary_local['trading_currencies'][0]] != 0 and strategy_score[-1]\
            < fitting_dictionary['sell_threshold']:
        bid = ticker[strategy_dictionary_local['ticker_1']]['highestBid']
        print bid
        print balance[strategy_dictionary_local['trading_currencies'][0]]
        print poloniex_session.buy(ticker_1_local, bid, balance[strategy_dictionary_local['trading_currencies'][0]])

        candle_update(fitting_dictionary, data_to_predict, data_2, strategy_dictionary_local, fitting_dictionary,
                      counter_local)

    elif balance[strategy_dictionary_local['trading_currencies'][1]] != 0  and strategy_score[-1]\
            > fitting_dictionary['buy_threshold']:
        ask = ticker[strategy_dictionary_local['ticker_1']]['lowestAsk']
        print ask
        print balance[strategy_dictionary_local['trading_currencies'][1]]
        print poloniex_session.sell(ticker_1_local, ask, balance[strategy_dictionary['trading_currencies'][1]])

        candle_update(fitting_dictionary, data_to_predict, data_2, strategy_dictionary_local, fitting_dictionary,
                      counter_local)


if __name__ == '__main__':
    strategy_dictionary = {
        'trading_currencies': ['ETH', 'BTC'],
        'ticker_1': 'BTC_ETH',
        'ticker_2': 'USDT_BTC',
        'offset': 0,
        'n_days': 30,
        'candle_size': 1800,
        'bid_ask_spread': 0.001,
        'transaction_fee': 0.0025,
        'train_test_ratio': 1,
        'regression_mode': 'regression',
        'ml_mode': 'randomforest',
        'output_flag': False,
        'window_range_1': np.linspace(1, 20, 3, dtype=int),
        'window_range_2': np.linspace(200, 1000, 5, dtype=int)
    }

    counter = 0
    daily_fit(strategy_dictionary, counter)
