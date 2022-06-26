import os 
from dotenv import load_dotenv
from collections import namedtuple

def load_aws_keys(): 
    """Loads my aws keys as a named tuple object
    """
    load_dotenv()
    AWSKeys = namedtuple(
        typename = 'AWSKeys', 
        field_names=['api_key', 'secret_key', 'region'])

    api_key = os.environ['ACCESS_KEYID']
    secret_key = os.environ['SECRET_ACCESS_KEY']
    region = os.environ['REGION']

    my_keys = AWSKeys(api_key, secret_key, region)

    return my_keys

