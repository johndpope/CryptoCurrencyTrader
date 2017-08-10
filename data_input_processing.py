import numpy as np
import pandas as pd
from sklearn.preprocessing import Imputer,minmax_scale
from sklearn.decomposition import PCA, FastICA
from poloniex_API import poloniex
from API_settings import API_secret, API_key


class Data:   
    def __init__(self, currency_pair, start, end, period, web_flag, filename=None):
        self.date = []
        self.price = []
        self.close = []
        self.open = []
        self.high = []
        self.low = []
        self.volume = []
        self.time = []
        self.fractional_close = []
        self.high_low_spread = []
        self.open_close_spread = []
        self.absolute_volatility = []
        self.exponential_moving_average_1 = []
        self.exponential_moving_average_2 = []
        self.exponential_moving_average_3 = []
        self.exponential_moving_average_4 = []
        self.exponential_moving_average_5 = []
        self.exponential_moving_volatility_1 = []
        self.exponential_moving_volatility_2 = []
        self.exponential_moving_volatility_3 = []
        self.exponential_moving_volatility_4 = []
        self.exponential_moving_volatility_5 = []
        self.kalman_signal = []
        self.candle_price_difference = []
        if web_flag:
            self.candle_input_web(currency_pair, start, end, period)
        else:
            self.candle_input_file(filename, start, end, period)

    def candle_input_file(self, filename, start, end, period):
        candle_array = pd.read_csv(filename).as_matrix()

        start_index = (np.abs(candle_array[:, 0] - start)).argmin()
        end_index = (np.abs(candle_array[:, 0] - end)).argmin()

        period_index = period / 300

        self.date = candle_array[start_index:period_index:end_index, 0]
        self.open = candle_array[start_index:period_index:end_index, 3]
        self.close = candle_array[(start_index + period_index - 1):end_index:period_index, 4]
        self.high = np.zeros(len(self.close))
        self.low = np.zeros(len(self.close))

        for i in range(int(np.floor(len(self.high) / period_index))):
            loop_start = i * period_index
            self.high[i] = np.max(candle_array[loop_start:loop_start + period_index, 1])
            self.low[i] = np.min(candle_array[loop_start:loop_start + period_index, 2])


    def candle_input_web(self, currency_pair, start, end, period):
        poloniex_session = poloniex(API_key, API_secret)

        candle_json = poloniex_session.returnChartData(currency_pair, start, end, period)

        candle_length = len(candle_json[u'candleStick'])
        self.date = nan_array_initialise(candle_length)
        self.close = nan_array_initialise(candle_length)
        self.open = nan_array_initialise(candle_length)
        self.high = nan_array_initialise(candle_length)
        self.low = nan_array_initialise(candle_length)

        for loop_counter in range(candle_length):
            self.date[loop_counter] = candle_json[u'candleStick'][loop_counter][u'date']
            self.close[loop_counter] = candle_json[u'candleStick'][loop_counter][u'close']
            self.open[loop_counter] = candle_json[u'candleStick'][loop_counter][u'open']
            self.high[loop_counter] = candle_json[u'candleStick'][loop_counter][u'high']
            self.low[loop_counter] = candle_json[u'candleStick'][loop_counter][u'low']

    def extend_candle(self, new_candle):
        for date in new_candle.date:
            if date in self.date:
                trim_candle(new_candle, np.where(new_candle.date == date))

        self.date = np.concatenate((self.date, new_candle.date))
        self.open = np.concatenate((self.open, new_candle.open))
        self.close = np.concatenate((self.close, new_candle.close))
        self.high = np.concatenate((self.high, new_candle.high))
        self.low = np.concatenate((self.low, new_candle.low))

    def normalise_data(self):
        self.fractional_close = fractional_change(self.close)

    def calculate_high_low_spread(self):
        self.high_low_spread = self.high - self.low

    def calculate_open_close_spread(self):
        self.open_close_spread = self.close - self.open

    def calculate_absolute_volatility(self):
        self.calculate_high_low_spread()
        self.calculate_open_close_spread()
        self.absolute_volatility = np.abs(self.high_low_spread) - np.abs(self.open_close_spread)

    def calculate_indicators(self, strategy_dictionary):
        self.calculate_absolute_volatility()
        self.exponential_moving_average_1 = exponential_moving_average(self.close[:-1],
                                                                       strategy_dictionary['windows'][0])
        self.exponential_moving_average_2 = exponential_moving_average(self.close[:-1],
                                                                       strategy_dictionary['windows'][1])
        self.exponential_moving_average_3 = exponential_moving_average(self.close[:-1],
                                                                       strategy_dictionary['windows'][2])
        self.exponential_moving_average_4 = exponential_moving_average(self.close[:-1],
                                                                       strategy_dictionary['windows'][3])
        self.exponential_moving_average_5 = exponential_moving_average(self.close[:-1],
                                                                       strategy_dictionary['windows'][4])

        self.exponential_moving_volatility_1 = exponential_moving_average(self.absolute_volatility[:-1],
                                                                          strategy_dictionary['windows'][0])
        self.exponential_moving_volatility_2 = exponential_moving_average(self.absolute_volatility[:-1],
                                                                          strategy_dictionary['windows'][1])
        self.exponential_moving_volatility_3 = exponential_moving_average(self.absolute_volatility[:-1],
                                                                          strategy_dictionary['windows'][2])
        self.exponential_moving_volatility_4 = exponential_moving_average(self.absolute_volatility[:-1],
                                                                          strategy_dictionary['windows'][3])
        self.exponential_moving_volatility_5 = exponential_moving_average(self.absolute_volatility[:-1],
                                                                          strategy_dictionary['windows'][4])
        self.kalman_signal = kalman_filter(self.close[:-1])


def kalman_filter(input_price):
    n_iter = len(input_price)
    vector_size = (n_iter,)

    Q = 1E-5

    post_estimate = np.zeros(vector_size)  
    P = np.zeros(vector_size)
    post_estimate_minus = np.zeros(vector_size)
    Pminus = np.zeros(vector_size)
    K = np.zeros(vector_size)

    R = 0.1 ** 2

    post_estimate[0] = input_price[0]
    P[0] = 1.0

    for k in range(1, n_iter):
        post_estimate_minus[k] = post_estimate[k - 1]
        Pminus[k] = P[k - 1] + Q

        K[k] = Pminus[k] / (Pminus[k] + R)
        post_estimate[k] = post_estimate_minus[k] + K[k] * (input_price[k] - post_estimate_minus[k])
        P[k] = (1 - K[k]) * Pminus[k]
        
    return post_estimate


class TradingTargets:
    def __init__(self, normalise_data_obj):
        self.fractional_close = normalise_data_obj.fractional_close
        self.high = normalise_data_obj.high
        self.strategy_score = np.full([len(self.fractional_close)], np.nan)
        self.buy_sell = []

    def ideal_buy_sell(self, bid_ask_spread, transaction_fee):
        effective_fee_factor = effective_fee(bid_ask_spread, transaction_fee)
        fractional_close_length = len(self.fractional_close)

        self.buy_sell = np.zeros(fractional_close_length)

        for index in range(fractional_close_length):
            while_counter = 0
            net_change = 1.0
            while (net_change * effective_fee_factor < 1) & (net_change > effective_fee_factor) \
                    & (index + while_counter < fractional_close_length):
                net_change *= self.fractional_close[index + while_counter]
                while_counter += 1

            if net_change * effective_fee_factor > 1:
                self.buy_sell[index] = 1
            elif net_change < effective_fee_factor:
                self.buy_sell[index] = -1
            elif index + while_counter == fractional_close_length:
                self.buy_sell[index:] = 0

    def ideal_strategy_score(self, strategy_dictionary):
        effective_fee_factor = effective_fee(strategy_dictionary)
        fractional_close_length = len(self.fractional_close)

        self.strategy_score = np.ones(fractional_close_length)

        for index in range(fractional_close_length):
            while_counter = 0
            net_change = 1.0
            down_index = fractional_close_length
            draw_down = 1
            while (net_change * effective_fee_factor < 1) & (index + while_counter < fractional_close_length):
                net_change *= self.fractional_close[index + while_counter]
                while_counter += 1

                if draw_down > net_change:
                    draw_down = net_change
                    down_index = while_counter

            if net_change * effective_fee_factor > 1:
                self.strategy_score[index] = draw_down
            elif index + while_counter == fractional_close_length:
                self.strategy_score[index:] = 1

            while_counter = 0
            net_change = 1.0
            upside = 1
            up_index = fractional_close_length
            while (net_change > effective_fee_factor) & (index + while_counter < fractional_close_length):
                net_change *= self.fractional_close[index + while_counter]
                while_counter += 1

                if upside < net_change:
                    upside = net_change
                    up_index = while_counter

            if (net_change < effective_fee_factor) and up_index < down_index:
                self.strategy_score[index] = upside
            elif index + while_counter == fractional_close_length:
                self.strategy_score[index:] = 1

    def convert_score_to_classification_target(self):
        self.strategy_score[self.strategy_score > 1] = 1
        self.strategy_score[self.strategy_score < 1] = -1


def effective_fee(strategy_dictionary):
    return 1 - strategy_dictionary['transaction_fee'] - strategy_dictionary['bid_ask_spread']


def trim_candle(candle, index):
    np.delete(candle.date, index)
    np.delete(candle.open, index)
    np.delete(candle.close, index)
    np.delete(candle.high, index)
    np.delete(candle.low, index)


def fractional_change(price):
    return price[1:] / price[:-1]


def exponential_moving_average(data, window):
        weights = np.exp(np.linspace(-1., 0., window))
        weights /= weights.sum()
        ema = np.convolve(data, weights, mode='full')[:len(data)]
        ema[:window] = ema[window]
        return ema


def staggered_input(input_vector, offset):
    fractional_price_array = input_vector[offset:]
    for index in range(1, offset):
        fractional_price_array = np.vstack((fractional_price_array, input_vector[offset - index:-index]))

    return fractional_price_array


def calculate_data_length(start, end, period):
    return int((end - start) / period)


def nan_array_initialise(size):
    array = np.empty((size,))
    array[:] = np.NaN
    return array


def generate_training_variables(data_obj, strategy_dictionary):
    trading_targets = TradingTargets(data_obj)
    trading_targets.ideal_strategy_score(strategy_dictionary)

    if strategy_dictionary['regression_mode'] == 'classification':
        trading_targets.convert_score_to_classification_target()

    data_obj.calculate_indicators(strategy_dictionary)

    fitting_inputs = np.vstack((
        #data_obj.exponential_moving_average_1,
        data_obj.exponential_moving_average_2,
        data_obj.exponential_moving_average_3,
        data_obj.exponential_moving_average_4,
        data_obj.exponential_moving_average_5,
        #data_obj.exponential_moving_volatility_1,
        data_obj.exponential_moving_volatility_2,
        data_obj.exponential_moving_volatility_3,
        data_obj.exponential_moving_volatility_4,
        data_obj.exponential_moving_volatility_5,
        data_obj.kalman_signal,
        #data_obj.close[:-1],
        #data_obj.open[:-1],
        #data_obj.high[:-1],
        #data_obj.low[:-1],
        ))

    fitting_inputs = fitting_inputs.T

    fitting_inputs_scaled = minmax_scale(fitting_inputs)

    if strategy_dictionary['preprocessing'] == 'PCA':
        fitting_inputs_scaled = pca_transform(fitting_inputs_scaled)

    if strategy_dictionary['preprocessing'] == 'FastICA':
        fitting_inputs_scaled = fast_ica_transform(fitting_inputs_scaled)

    fitting_targets = trading_targets.strategy_score

    return fitting_inputs_scaled, fitting_targets


def imputer_transform(data):
    imputer = Imputer()
    imputer.fit(data)
    return imputer.transform(data)


def pca_transform(fitting_inputs_scaled):
    pca = PCA()
    pca.fit(fitting_inputs_scaled)

    return pca.transform(fitting_inputs_scaled)


def fast_ica_transform(fitting_inputs_scaled):
    ica = FastICA()
    ica.fit(fitting_inputs_scaled)

    return ica.transform(fitting_inputs_scaled)


def train_test_indices(input_data, train_factor):
    data_length = len(input_data)
    train_indices_local = range(0, int(data_length * train_factor))
    test_indices_local = range(train_indices_local[-1] + 1, data_length)

    return train_indices_local, test_indices_local


def train_test_validation_indices(input_data):
    train_factor = 0.5
    test_factor = 0.25
    data_length = len(input_data)
    train_indices_local = range(0, int(data_length * train_factor))
    test_indices_local = range(train_indices_local[-1] + 1, int(data_length * (train_factor + test_factor)))
    validation_indices_local = range(test_indices_local[-1] + 1, data_length)

    return train_indices_local, test_indices_local, validation_indices_local
