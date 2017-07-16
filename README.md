# CryptoCurrencyTrader
A machine learning program in python to generate cryptocurrency trading strategies using machine learning.
The script is inspired by both the pytrader project https://github.com/owocki/pytrader, and the auto-sklearn project https://automl.github.io/auto-sklearn/stable/. 

## Input Data
Minor changes were made to the Poloniex API python wrapper which is inluded in the repository https://github.com/s4w3d0ff/python-poloniex. Data is retrieved via the Poloniex API in OHLC (open, high, low, close) candlestick format.

### Technical Indicators - Training Inputs
A series of technical indicators are calculated and provided as inputs to the machine learning optimisation, exponential moving averages and exponential moving volatilities over a series of windows. A kalman filter is also provided as an input.

### Training Targets - Strategy Score
An ideal trading strategy is generated based on past data, if for a given candlestick the trader is holding 
![Alt text](strategyscore.jpg?raw=true "Optional Title")

## Machine Learning Meta-fitting and Hyper Parameter Optimisation
![Alt text](ML_Flowchart.png?raw=true "Optional Title")
