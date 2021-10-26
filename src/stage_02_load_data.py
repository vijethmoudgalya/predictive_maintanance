import pandas as pd
from stage_01_get_data import read_params,get_data
from utils.preprcocessing import get_highly_correlated_cols,drop_highly_corelated_cols,dropUnnecessaryColumns
import argparse
import os
import numpy as np


def data_load(config_path):
    '''
                        Method Name: data_load
                            Description: This method loads the data from the file and convert into a pandas dataframe
                            Output: Returns a Dataframes, which is our data for training
                            On Failure: Raise Exception .
        '''
    try:

        config = read_params(config_path)
        data = get_data(config_path)
        columns = config['load_data']['columns']
        highly_corelated_cols = get_highly_correlated_cols(data)
        data_processed = drop_highly_corelated_cols(data, highly_corelated_cols)
        data = dropUnnecessaryColumns(data_processed,columns)
        raw_data_path = config['load_data']['raw_dataset_csv']
        data.to_csv(raw_data_path,sep = ',',index = False)

    
    except Exception as e:
        raise e







if __name__ == "__main__":
    args = argparse.ArgumentParser()
    default_config_path = os.path.join("params.yaml")
    args.add_argument("--config", default=default_config_path)
    parsed_args = args.parse_args()
    #data = data_getter()
    data_load(config_path=parsed_args.config)