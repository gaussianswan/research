import datetime
import pandas as pd
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.data import CryptoHistoricalDataClient, CryptoBarsRequest

if __name__ == "__main__":

    data_request = CryptoBarsRequest(
    symbol_or_symbols='BTC/USD',
    start = datetime.datetime(2021, 1, 1),
    end = datetime.datetime.now(),
    timeframe=TimeFrame(1, TimeFrameUnit.Hour)
    )

    client = CryptoHistoricalDataClient()

    # Getting the data and saving it
    data = client.get_crypto_bars(request_params=data_request)
    historical_bars_df = data.df.reset_index()

    historical_bars_df.drop(['symbol'], axis = 1, inplace=True)
    historical_bars_df['timestamp'] = pd.to_datetime(historical_bars_df['timestamp'])
    historical_bars_df.to_parquet("btc_usd_1h.parquet")