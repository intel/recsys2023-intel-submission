import timeit
import sklearn.metrics
import numpy as np
import torch
import os
import pandas as pd
import os
from tqdm import tqdm
from category_encoders.count import CountEncoder as SKLCountEncoder
from category_encoders import *
import pickle

class Indexer:
    def __init__(self, feature_name, partition_key, timestamp = 'f_1', counter = None):
        self.partition_key = partition_key
        self.timestamp = timestamp
        self.feature_name = feature_name
        if isinstance(counter, type(None)):
            self.counter = {}
        else:
            self.counter = counter
        
    @classmethod
    def load_model(cls, filename):
        with open(filename, 'rb') as file:
            data = pickle.load(file)
        return cls(data['feature_name'], data['partition_key'], data['timestamp'], data['indexer'])
        
    def save(self, filename):
        data = {}
        data['partition_key'] = self.partition_key
        data['timestamp'] = self.timestamp
        data['feature_name'] = self.feature_name
        data['indexer'] = self.counter
        
        with open(filename, 'wb') as file:
            pickle.dump(data, file)
        

    def fit_transform(self, df):
        new_features = []
        partition_key = self.partition_key
        feature_name = self.feature_name
        timestamp = self.timestamp
        day_range = df[timestamp].unique()
        day_range.sort()
        time_col = df[timestamp].to_list()
        feature_col = df[feature_name].to_list()
        partition_col = df[partition_key].to_list()        
        for time_value, feature_value, partition_value in zip(time_col, feature_col, partition_col):
            #print(time_value, feature_value, partition_value)
            if feature_value not in self.counter:
                self.counter[feature_value] = {}
            if partition_value not in self.counter[feature_value]:
                self.counter[feature_value][partition_value] = dict((k, 0) for k in day_range)       
            self.counter[feature_value][partition_value][time_value] += 1
            new_features.append(self.counter[feature_value][partition_value][time_value])
        return pd.Series(new_features, index = df.index)
    
    def transform(self, df):
        new_features = []
        partition_key = self.partition_key
        feature_name = self.feature_name
        timestamp = self.timestamp
        time_col = df[timestamp].to_list()
        feature_col = df[feature_name].to_list()
        partition_col = df[partition_key].to_list()        
        for time_value, feature_value, partition_value in zip(time_col, feature_col, partition_col):
            #print(time_value, feature_value, partition_value)
            if feature_value not in self.counter:
                self.counter[feature_value] = {}
            if partition_value not in self.counter[feature_value]:
                self.counter[feature_value][partition_value] = {}
            if time_value not in self.counter[feature_value][partition_value]:
                self.counter[feature_value][partition_value][time_value] = 0
            self.counter[feature_value][partition_value][time_value] += 1
            new_features.append(self.counter[feature_value][partition_value][time_value])
        return pd.Series(new_features, index = df.index)
    
class NewValueEncoder:
    def __init__(self, feature_name, encoder = None):
        self.feature_name = feature_name
        if isinstance(encoder, type(None)):
            self.encoder = None
        else:
            self.encoder = encoder
            
    @classmethod
    def load_model(cls, filename):
        with open(filename, 'rb') as file:
            data = pickle.load(file)
        return cls(data['feature_name'], data['encoder'])
        
    def save(self, filename):
        data = {}
        data['feature_name'] = self.feature_name
        data['encoder'] = self.encoder
        
        with open(filename, 'wb') as file:
            pickle.dump(data, file)

    def fit_transform(self, df):
        feature_name = self.feature_name
        encoder = pd.DataFrame({f'{feature_name}_first_day':df.groupby([feature_name])['f_1'].min()})
        self.encoder = encoder
        df = df.merge(encoder, on = feature_name, how = 'left')
        df[f'{feature_name}_fdflag'] = (df['f_1'] == df[f'{feature_name}_first_day'])
        return df
        
    def transform(self, df):
        encoder_df = self.encoder
        feature_name = self.feature_name
        
        existing_values = encoder_df.index.to_list()
        df[f'{feature_name}_fdflag'] = ~df[feature_name].isin(existing_values)
        return df

class CountEncoder:
    def __init__(self, feature_name = None, handle_unknown = None, encoder = None):
        self.feature_name = feature_name
        if isinstance(encoder, type(None)):
            self.encoder = SKLCountEncoder(cols=[feature_name], handle_unknown=handle_unknown)
        else:
            self.encoder = encoder
        
    @classmethod
    def load_model(cls, filename):
        with open(filename, 'rb') as file:
            data = pickle.load(file)
        return cls(encoder = data['encoder'])
        
    def save(self, filename):
        data = {}
        data['encoder'] = self.encoder
        with open(filename, 'wb') as file:
            pickle.dump(data, file)
        

    def fit_transform(self, df):
        return self.encoder.fit_transform(df)
    
    def transform(self, df):
        return self.encoder.transform(df)

class GroupLabelEncoder:
    def __init__(self, feature_name, f):
        self.feature_name = feature_name
        self.f = f

    def fit_transform(self, df):
        encoder = self.generate_categorify_encoder(df, self.feature_name, self.f) 
        self.encoder = encoder
        df = df.merge(self.encoder, on = self.f, how = 'left')
        return df
        
    def transform(self, df):
        encoder_df = self.encoder
        feature_name = self.feature_name
        f = self.f

        # convert df to dict
        key_feats = [i for i in encoder_df.columns if i != feature_name]
        encoder_df['key'] = encoder_df[key_feats].apply(lambda x: str(list(x)), axis=1)
        encoder_dict = dict(zip(encoder_df['key'].to_list(), encoder_df[feature_name].to_list()))
        max_id = max(list(encoder_dict.values())) + 1

        # get sub df
        sub_df = df[f]
        sub_df['key'] = sub_df[key_feats].apply(lambda x: str(list(x)), axis=1)

        cate_id_list = []
        for key_id in sub_df['key'].to_list():
            if key_id not in encoder_dict:
                encoder_dict[key_id] = max_id
                max_id += 1
            cate_id_list.append(encoder_dict[key_id])
        df[feature_name] = pd.Series(cate_id_list)
        return df

    def generate_categorify_encoder(self, train_df, feature_name, grouped_features):
        k = [i for i in train_df.columns if i not in grouped_features]
        if len(k) == 0:
            raise NotImplementedError("df contains all the grouped keys, not support yet")
        k = k[0]

        df = train_df
        encoder = df.groupby(by = grouped_features, as_index = False)[k].count().drop(k, axis = 1)
        encoder[feature_name] = encoder.index
        return encoder

class Timer:
    level = 0
    viewer = None
    def __init__(self, name):
        self.name = name
        if Timer.viewer:
            Timer.viewer.display(f"{name} started ...")
        else:
            print(f"{name} started ...")

    def __enter__(self):
        self.start = timeit.default_timer()
        Timer.level += 1

    def __exit__(self, *a, **kw):
        Timer.level -= 1
        if Timer.viewer:
            Timer.viewer.display(
                f'{"  " * Timer.level}{self.name} took {timeit.default_timer() - self.start} sec')
        else:
            print(
                f'{"  " * Timer.level}{self.name} took {timeit.default_timer() - self.start} sec')

def fix_na(df):
    df = df.fillna(0)
    for col in df.select_dtypes([float]):
        v = col
        if np.array_equal(df[v], df[v].astype(int)):
            df[v] = df[v].astype(int, copy = False)
    return df
   
def load_csv_to_pandasdf(dataset):
    if not isinstance(dataset, str):
        raise NotImplementedError("Only support pandas Dataframe as input")
    if not os.path.exists(dataset):
        raise FileNotFoundError(f"{dataset} is not exists")
    if os.path.isdir(dataset):
        input_files = sorted(os.listdir(dataset))
        df = pd.read_csv(dataset + "/" + input_files[0], sep = '\t')
        for file in tqdm(input_files[1:]):
            part = pd.read_csv(dataset + "/" + file, sep = '\t')    
            df = pd.concat([df, part],axis=0)
    else:
        df = pd.read_csv(dataset, sep = '\t')
    df = fix_na(df)
    return df

def H_np(y, p):
    e = np.finfo(float).eps
    return -y * np.log(p + e) - (1 - y) * np.log(1 - p + e)
    
def nce_score(y_true, y_pred, verbose = False):
    avg_logloss_y_p = np.mean(sklearn.metrics.log_loss(y_true, y_pred))
    #avg_log_reci_p = np.mean(np.log(1/y_pred))
    ctr = y_true.sum() / y_true.shape[0]
    if not verbose:
        logloss_ctr = H_np(ctr, ctr)
        return avg_logloss_y_p / logloss_ctr

def get_combined_df(file_list, save_path=None, weights = []):
    model_num = len(file_list)
    if len(weights) == 0:
        weights = [1/model_num] * model_num
    df = pd.read_csv(file_list[0], sep='\t')\
        .rename(columns={"row_id": "RowId"})
    df['is_installed'] = df['is_installed'] * weights[0]
    df_seq = df[['RowId']]
    for i in range(1, model_num):
        df_temp = pd.read_csv(file_list[i], sep='\t')\
            .rename(columns={"row_id": "RowId"})
        df_temp = df_seq.merge(df_temp, on='RowId', how='left').reset_index(drop=True)
        df['is_installed'] += df_temp['is_installed'] * weights[i]
    df['is_clicked'] = 0.0
    if save_path:
        df.round({'is_clicked': 1, 'is_installed': 5})\
            .to_csv(save_path, sep='\t', header=True, index=False)
    return df