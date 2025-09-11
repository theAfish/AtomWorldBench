import pandas as pd


def load_data(data_file: str) -> pd.DataFrame:
    """
    Load the dataset from a CSV file.
    The CSV is expected to have columns: 'original_cif', 'modified_cif'
    """
    df = pd.read_csv(data_file, sep=',', dtype={'removed_value': str})
    df.fillna(value={'removed_value': 'None'}, inplace=True)
    return df