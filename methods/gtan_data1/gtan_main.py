import os
import numpy as np
import dgl
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch

def load_gtan_data(dataset: str, test_size: float):
    prefix = os.path.join(os.path.dirname(__file__), "..", "..", "data/")
    feat_data, labels, train_idx, test_idx, g, cat_features = None, None, None, None, None, None  # Ensure all variables are initialized
    
    if dataset == "DATA1":
        cat_features = ["cc_num", "merchant", "category", "city"]
        data_path = prefix + "DATA1_full.csv"
        
        # Check if file exists
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Dataset file {data_path} not found.")
        
        df = pd.read_csv(data_path)
        df = df.loc[:, ~df.columns.str.contains('Unnamed')]
        
        # Check if required columns are in the dataset
        required_columns = {"cc_num", "merchant", "category", "city", "is_fraud", "Time"}
        if not required_columns.issubset(df.columns):
            missing_cols = required_columns - set(df.columns)
            raise ValueError(f"Dataset is missing columns: {missing_cols}")
        
        data = df[df["is_fraud"] <= 2].reset_index(drop=True)

        # Build the graph structure based on defined pairs
        out, alls, allt = [], [], []
        pair = ["cc_num", "merchant", "category", "city"]
        for column in pair:
            src, tgt = [], []
            edge_per_trans = 3
            for c_id, c_df in data.groupby(column):
                c_df = c_df.sort_values(by="Time")
                df_len = len(c_df)
                sorted_idxs = c_df.index
                src.extend([sorted_idxs[i] for i in range(df_len)
                            for j in range(edge_per_trans) if i + j < df_len])
                tgt.extend([sorted_idxs[i+j] for i in range(df_len)
                            for j in range(edge_per_trans) if i + j < df_len])
            alls.extend(src)
            allt.extend(tgt)
        alls = np.array(alls)
        allt = np.array(allt)
        g = dgl.graph((alls, allt))

        cal_list = ["cc_num", "merchant", "category", "city"]
        for col in cal_list:
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col].apply(str).values)
        
        feat_data = data.drop("is_fraud", axis=1)
        labels = data["is_fraud"]

        index = list(range(len(labels)))
        g.ndata['label'] = torch.from_numpy(labels.to_numpy()).to(torch.long)
        g.ndata['feat'] = torch.from_numpy(feat_data.to_numpy()).to(torch.float32)

        # Split dataset
        train_idx, test_idx, _, _ = train_test_split(
            index, labels, stratify=labels, test_size=test_size, random_state=2, shuffle=True)
    else:
        raise ValueError(f"Dataset {dataset} not supported")
    
    return feat_data, labels, train_idx, test_idx, g, cat_features
