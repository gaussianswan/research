import pandas as pd 
import yfinance as yf
import numpy as np

class OHLCVData: 

    def __init__(self, data = pd.DataFrame, name: str = None) -> None:
        self.name = name
        self.data = data

        self.column_names = self.data.columns

    def get_series(self, column_name: str, name: str = None) -> pd.Series: 
        series = self.data[column_name]

        if name: 
            series.name = name

        return series

    @classmethod 
    def from_yahoo_finance(cls, ticker: str, name: str = None): 

        data = yf.download(tickers=ticker, progress=False)
        return cls(data = data, name = name)


if __name__ == "__main__": 

    ticker = 'META'

    data = OHLCVData.from_yahoo_finance(ticker = ticker, name = 'META')
    print(data.data)