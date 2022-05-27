from sklearn.preprocessing import MinMaxScaler

# import configparser

PATH_TO_CONFIG = 'secret.test.cfg'
TEST_API_KEY = 'Lvy4Z68yCTmaViokYB0rPieZoniQ2Zm0fccFmCuWt8jg7MkdMgj7Cu0Gsh1QEnIL'
TEST_API_SECRET = 'YAgQBxzCDJLvplbujTre5IblHGzqDJ5naCJAreSiZ85AcMg9ACSEGEji7ljULFoV'


def load_app_config():
    # config = configparser.ConfigParser()
    # config.read_file(open(PATH_TO_CONFIG))
    dictionary = dict()
    dictionary['api_key'] = TEST_API_KEY # config.get('BINANCE', 'API_KEY')
    dictionary['secret_key'] = TEST_API_SECRET # config.get('BINANCE', 'SECRET_KEY')
    return dictionary


def get_scaler():
    return MinMaxScaler(feature_range=(0, 1))
