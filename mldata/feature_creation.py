import talib as ta
import pandas as pd
from data import OHLCVData

class TechnicalFeatureFactory: 

    def __init__(self, data: OHLCVData) -> None:
        self.data = data 

    def rsi(self, name: str = 'rsi') -> pd.Series: 

        series = ta.RSI(self.data.get_series(column_name = 'Close'))

        if name: 
            series.name = name
            
        return series
    
    def bbands_normalized(self) -> pd.DataFrame: 
        pass 

    def adx(self) -> pd.Series: 
        pass 

    def stoch_rsi(self) -> pd.Series: 
        pass

    def normalized_average_true_range(self) -> pd.Series: 
        pass 

    def price_acceleration(self) -> pd.Series: 
        pass

class WindowedDataFactory: 

    def __init__(self, data: OHLCVData) -> None:
        self.data = data

    def create_windowed_dataframe(self, column: str = 'Close', window: int = 10, prefix: str = 'Prev') -> pd.DataFrame: 

        windowed_data = pd.DataFrame()
        series = self.data.get_series(column_name=column)
        for i in range(1, window+1): 

            prev = series.shift(i)
            windowed_data[f'Prev_{i}'] = prev

        return windowed_data

if __name__ == "__main__": 
    ticker = 'META'

    ohlcv_data = OHLCVData.from_yahoo_finance(ticker=ticker, name = 'META')
    
    feature_factory = TechnicalFeatureFactory() 
    rsi = feature_factory.rsi(data = ohlcv_data)

    print(rsi)
    