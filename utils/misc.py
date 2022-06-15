import pandas as pd


def get_last_row(df: pd.DataFrame):
    return df.iloc[-1, :]
