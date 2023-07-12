import pandas as pd
import numpy as np
import pandas_ta as ta

if __name__ == "__main__":

    data = pd.read_parquet('btc_usd_1h.parquet')

    rsi = ta.rsi(close = data['close'], length = 14)

    # Creating the MACD
    macd = ta.macd(close = data['close'])
    macd.columns = ['MACD', 'Histogram', 'Signal']

    # Creating the bollinger bands series
    bbands = ta.bbands(close = data['close'], length = 50)
    cols = ['lower_band', 'middle_band', 'upper_band', 'bandwidth', '%B']
    bbands.columns = cols

    # Adding in some columns for side by side visualization
    bbands['timestamp'] = data['timestamp']
    bbands['close'] = data['close']

    short_sma_period = 12
    medium_sma_period = 24
    long_sma_period = 72

    short_sma = data['close'].rolling(short_sma_period).mean()
    medium_sma = data['close'].rolling(medium_sma_period).mean()
    long_sma = data['close'].rolling(long_sma_period).mean()

    short_sma.name = f'SMA_{short_sma_period}'
    medium_sma.name = f'SMA_{medium_sma_period}'
    long_sma.name = f'SMA_{long_sma_period}'

    log_short_sma = short_sma.apply(np.log)
    log_medium_sma = medium_sma.apply(np.log)
    log_long_sma = long_sma.apply(np.log)
    log_close = data['close'].apply(np.log)

    smas = pd.concat([short_sma, medium_sma, long_sma], axis = 1)
    log_close_to_short_sma = log_close - log_short_sma
    log_close_to_medium_sma = log_close - log_medium_sma
    log_close_to_long_sma = log_close - log_long_sma

    log_close_to_short_sma.name = 'log_close_to_short_sma'
    log_close_to_medium_sma.name = 'log_close_to_medium_sma'
    log_close_to_long_sma.name = 'log_close_to_long_sma'

    fast_ema_period = 12
    slow_ema_period = 26

    ema_26 = data['close'].ewm(halflife = slow_ema_period).mean()
    ema_12 = data['close'].ewm(halflife = fast_ema_period).mean()

    ema_26.name = f'EMA_{slow_ema_period}'
    ema_12.name = f'EMA_{fast_ema_period}'

    emas = pd.concat([ema_12, ema_26], axis = 1)
    mas = pd.concat([emas, smas], axis = 1)
    log_fast_ema = ema_12.apply(np.log)
    log_slow_ema = ema_26.apply(np.log)

    log_macd = log_fast_ema - log_slow_ema
    log_macd.name = 'LogMacd'

    log_fast_ema_slow_sma = log_fast_ema - log_long_sma
    log_fast_ema_med_sma = log_fast_ema - log_medium_sma
    log_fast_ema_fast_sma = log_fast_ema - log_short_sma

    # Creating lagged returns data
    num_lags = 10
    close_to_close_returns = data['close'].pct_change()

    lagged_returns_df = pd.DataFrame()

    for i in range(1, num_lags+1):
        ret = close_to_close_returns.shift(i)
        col = f'Return_{i}_periods_ago'
        lagged_returns_df[col] = ret

    vol_periods = [12, 24, 72]
    vol_features = pd.DataFrame()

    for period in vol_periods:
        vol = close_to_close_returns.rolling(period).std()
        col = f'Rolling_{period}_std'
        vol_features[col] = vol

    # putting all of the features together into one dataframe
    log_fast_ema_slow_sma.name = 'log_fast_ema_slow_sma'
    log_fast_ema_med_sma.name = 'log_fast_ema_med_sma'
    log_fast_ema_fast_sma.name = 'log_fast_ema_fast_sma'
    rsi.name = 'RSI_14'

    features = pd.concat([
        vol_features,
        lagged_returns_df,
        log_fast_ema_slow_sma,
        log_fast_ema_med_sma,
        log_fast_ema_fast_sma,
        log_close_to_short_sma,
        log_close_to_medium_sma,
        log_close_to_long_sma,
        log_macd,
        bbands['%B'],
        rsi
    ], axis = 1)

    features.to_csv("features.csv")


