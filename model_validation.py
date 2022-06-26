import pandas as pd

def create_tsplit_indices(df: pd.DataFrame, start_index: int = 0, num_test_vals: int = 1, num_train_vals: int = 90): 
    """Creates the indices that should be used for training and testing on a rolling basis

    Returns a generator object which gives the training indices and the test indices

    Args:
        df (_type_): Dataframe to base the indices off of
        start_index (int, optional): Starting index to do the time series split. Defaults to 0.
        num_test_vals (int, optional): Number of test values in the split. Defaults to 1.
        num_train_vals (int, optional): Number of training values to have in the split. Defaults to 90.
    """

    # The shape of the dataframe is (row, column)
    assert not df.empty, "Dataframe can't be empty"
    data_shape = df.shape
    assert num_train_vals + num_test_vals < data_shape[0], "There are not enough data points to give you the right splits"
    
    total_num_splits = data_shape[0] - num_train_vals - num_test_vals
    for row in range(start_index, total_num_splits):
        ending_train_row = row + num_train_vals
        ending_test_row = ending_train_row + num_test_vals
        yield list(range(row, ending_train_row)), list(range(ending_train_row, ending_test_row))
         
