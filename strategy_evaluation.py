import numpy as np
from matplotlib import pyplot as plt


def strategy_profit(currency_position, fractional_price, strategy_dictionary):
    buy_sell_length = len(currency_position)
    portfolio_value = np.ones(buy_sell_length)
    for index in range(1, buy_sell_length):
        if ((currency_position[index - 1] == 0) & (currency_position[index] == 1))\
                | ((currency_position[index - 1] == 1) & (currency_position[index] == 0)):
            portfolio_value[index] = (1  - strategy_dictionary['transaction_fee']
                                      - strategy_dictionary['bid_ask_spread']) * portfolio_value[index - 1]
        elif (currency_position[index - 1] == 1) & (currency_position[index] == 1):
            portfolio_value[index] = portfolio_value[index - 1] * fractional_price[index - 1]
        else:
            portfolio_value[index] = portfolio_value[index - 1]

    return portfolio_value


def convert_to_currency_position(buy_sell):
    buy_sell_length = len(buy_sell)
    
    currency_position = np.zeros(len(buy_sell))
    for index in range(len(currency_position)):
        currency_position[index] = buy_sell[index]
        currency_position[buy_sell == -1] = 0
        
        while_counter = 0
        while (index + while_counter < buy_sell_length) and (buy_sell[index + while_counter] == 0):
            currency_position[index + while_counter] = currency_position[index+ while_counter - 1]
            while_counter += 1
            
    return currency_position


def number_of_trades_from_currency_position(currency_position):
    return np.sum(np.abs(np.diff(currency_position)))


def convert_score_to_buy_sell(strategy_score, buy_threshold, sell_threshold):
    buy_sell = np.zeros(len(strategy_score))

    buy_sell[strategy_score > buy_threshold] = 1
    buy_sell[strategy_score <= sell_threshold] = -1

    return buy_sell


def convert_strategy_score_to_profit(strategy_local, buy_threshold, sell_threshold, fractional_close,
                                     strategy_dictionary):
    fitted_buy_sell = convert_score_to_buy_sell(strategy_local, buy_threshold, sell_threshold)
    fitted_currency_position = convert_to_currency_position(fitted_buy_sell)
    number_of_trades = number_of_trades_from_currency_position(fitted_currency_position)
    return strategy_profit(fitted_currency_position, fractional_close, strategy_dictionary), number_of_trades


def post_process_regression_results(fitting_dictionary, strategy_dictionary, fractional_close):
    profit_optimum = -1e5

    for buy_threshold in np.linspace(min(fitting_dictionary['training_strategy_score']),
                                     max(fitting_dictionary['training_strategy_score']), 20):
        for sell_threshold in np.linspace(min(fitting_dictionary['training_strategy_score']),
                                          max(fitting_dictionary['training_strategy_score']), 20):
            portfolio_value, n_trades = convert_strategy_score_to_profit(
                (fitting_dictionary['training_strategy_score']), buy_threshold, sell_threshold,
                fractional_close[fitting_dictionary['train_indices']], strategy_dictionary)

            profit_fraction = strategy_profit_score(portfolio_value, n_trades)

            if profit_optimum < profit_fraction:
                fitting_dictionary['buy_threshold'] = buy_threshold
                fitting_dictionary['sell_threshold'] = sell_threshold
                fitting_dictionary['number_of_trades'] = n_trades
                profit_optimum = profit_fraction

    fitting_dictionary['portfolio_value'], fitting_dictionary['n_trades'] = convert_strategy_score_to_profit(
        (fitting_dictionary['fitted_strategy_score']), fitting_dictionary['buy_threshold'],
        fitting_dictionary['sell_threshold'], fractional_close[fitting_dictionary['test_indices']],
        strategy_dictionary)

    return fitting_dictionary


def post_process_classification_results(fitting_dictionary, strategy_dictionary, fractional_close):
    fitted_currency_position = convert_to_currency_position(fitting_dictionary['fitted_strategy_score'])
    number_of_trades = number_of_trades_from_currency_position(fitted_currency_position)

    fitting_dictionary['portfolio_value'] = strategy_profit(fitted_currency_position, fractional_close[
        fitting_dictionary['test_indices']], strategy_dictionary)
    fitting_dictionary['n_trades'] = number_of_trades
    return fitting_dictionary


def post_process_training_results(strategy_dictionary, fitting_dictionary, data):
    if strategy_dictionary['regression_mode'] == 'classification':
        return post_process_classification_results(fitting_dictionary, strategy_dictionary, data.fractional_close)

    elif strategy_dictionary['regression_mode'] == 'regression':
        return post_process_regression_results(fitting_dictionary, strategy_dictionary, data.fractional_close)


def strategy_profit_score(strategy_profit_local, number_of_trades):
    profit_fraction = strategy_profit_local[-1] / np.min(strategy_profit_local)
    if number_of_trades == 0:
        profit_fraction = -profit_fraction
    return profit_fraction


def draw_down(strategy_profit_local):
    draw_down_temp = np.diff(strategy_profit_local)
    draw_down_temp[draw_down_temp > 0] = 0
    return np.mean(draw_down_temp)


def output_strategy_results(strategy_dictionary, fitting_dictionary, data_to_predict, toc):
    prediction_data = data_to_predict.close[fitting_dictionary['test_indices']]

    if strategy_dictionary['output_flag']:
        print "Fitting time: ", toc()

        print "Fractional profit compared to buy and hold: ", fitting_dictionary['portfolio_value'][-1]\
                                                                * prediction_data[0]\
                                                                / (fitting_dictionary['portfolio_value'][0]
                                                                * prediction_data[-1]) - 1
        print "Cross validation error: ", fitting_dictionary['error']
        print "Number of days: ", strategy_dictionary['n_days']
        print "Candle time period:", strategy_dictionary['candle_size']
        print "Fitting model: ", strategy_dictionary['ml_mode']
        print "Regression/classification: ", strategy_dictionary['regression_mode']
        print "Number of trades: ", fitting_dictionary['n_trades']
        print "Offset: ", strategy_dictionary['offset']
        print "\n"

    if strategy_dictionary['plot_flag']:
        plt.figure(1)
        plt.plot(prediction_data)

        plt.figure(2)
        plt.plot(fitting_dictionary['portfolio_value'])

        plt.show()
