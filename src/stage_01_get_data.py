
## read the parameters
# processed
# return the dataframe
import os
import yaml
import pandas as pd
import argparse
def read_params(config_path):
    with open(config_path) as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config

def get_data(config_path):
    config = read_params(config_path)
    #config['base']['project']
    data_path = config['data_source']["s3_source"]
    data_df = pd.read_csv(data_path, sep=',', encoding='utf-8')
    #print(data_df.head())
    #extra comments
    return data_df


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    default_config_path = os.path.join("params.yaml")
    args.add_argument("--config", default=default_config_path)
    parsed_args = args.parse_args()
    get_data(config_path=parsed_args.config)
    #args.add_argument("--datasource", default=None)