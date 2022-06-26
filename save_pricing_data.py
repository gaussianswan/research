import sys
from helperfuncs import load_aws_keys
from coinbase_crypto.pricing_data import PricingDataLoader

def main(): 
    aws_keys = load_aws_keys()

    keys = load_aws_keys()
    data_loader = PricingDataLoader(keys.api_key, keys.secret_key)

    for coin in sys.argv[1:]: 
        try: 
            candles = data_loader.get_candles(coin = coin, frequency = 'D')
        except Exception as e: 
            print(e)

        # Then if things go well, we load it in and save it
        filename = "{}_daily_data.csv".format(coin)
        candles.to_csv("pricing_data/{}".format(filename))

if __name__ == '__main__': 
    main()