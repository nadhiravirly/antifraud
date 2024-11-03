from collections import defaultdict
import pandas as pd
import numpy as np
import torch
import dgl
import random
import os
import pickle
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tqdm import tqdm

DATADIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data/")

def featmap_gen(tmp_df=None):
    """
    Handle DATA1 dataset and do some feature engineering.
    """
    time_span = [2, 3, 5, 15, 20, 50, 100, 150, 200, 300, 864, 2590, 5100, 10000, 24000]
    time_name = [str(i) for i in time_span]
    time_list = tmp_df['Time']
    post_fe = []
    for trans_idx, trans_feat in tqdm(tmp_df.iterrows()):
        new_df = pd.Series(trans_feat)
        temp_time = new_df.Time
        temp_amt = new_df.amt
        for length, tname in zip(time_span, time_name):
            lowbound = (time_list >= temp_time - length)
            upbound = (time_list <= temp_time)
            correct_data = tmp_df[lowbound & upbound]
            new_df[f'trans_at_avg_{tname}'] = correct_data['amt'].mean()
            new_df[f'trans_at_totl_{tname}'] = correct_data['amt'].sum()
            new_df[f'trans_at_std_{tname}'] = correct_data['amt'].std()
            new_df[f'trans_at_bias_{tname}'] = temp_amt - correct_data['amt'].mean()
            new_df[f'trans_at_num_{tname}'] = len(correct_data)
            new_df[f'trans_location_num_{tname}'] = len(correct_data.city.unique())
            new_df[f'trans_category_num_{tname}'] = len(correct_data.category.unique())
            new_df[f'trans_merchant_num_{tname}'] = len(correct_data.merchant.unique())
        post_fe.append(new_df)
    return pd.DataFrame(post_fe)

def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def process_graph_data(data_file, graph_file_name):
    """
    Process DATA1 dataset to create graph structure and neighbor risk-aware features.
    """
    data = pd.read_csv(data_file)
    data = featmap_gen(data.reset_index(drop=True))
    data.replace(np.nan, 0, inplace=True)
    data.to_csv(data_file.replace('.csv', '_full.csv'), index=None)
    data = pd.read_csv(data_file.replace('.csv', '_full.csv'))

    data = data.reset_index(drop=True)
    out = []
    alls = []
    allt = []
    # Mapping columns to reference structure
    pair = ["cc_num", "merchant", "city", "category"]  # Map as Source, Target, Location, Type
    for column in pair:
        src, tgt = [], []
        edge_per_trans = 3
        for c_id, c_df in tqdm(data.groupby(column), desc=column):
            c_df = c_df.sort_values(by="Time")
            df_len = len(c_df)
            sorted_idxs = c_df.index
            src.extend([sorted_idxs[i] for i in range(df_len) for j in range(edge_per_trans) if i + j < df_len])
            tgt.extend([sorted_idxs[i+j] for i in range(df_len) for j in range(edge_per_trans) if i + j < df_len])
        alls.extend(src)
        allt.extend(tgt)
    alls = np.array(alls)
    allt = np.array(allt)
    g = dgl.graph((alls, allt))

    cal_list = ["cc_num", "merchant", "city", "category"]  # Ensure consistent labeling
    for col in cal_list:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col].apply(str).values)
    feat_data = data.drop("is_fraud", axis=1)
    feat_data = feat_data.apply(pd.to_numeric, errors='coerce').fillna(0)
    labels = data["is_fraud"]
    g.ndata['label'] = torch.from_numpy(labels.to_numpy()).to(torch.long)
    g.ndata['feat'] = torch.from_numpy(feat_data.to_numpy()).to(torch.float32)
    dgl.data.utils.save_graphs(DATADIR + graph_file_name, [g])

    generate_neighbor_features(graph_file_name)

def generate_neighbor_features(graph_file_name):
    """
    Generate risk-aware neighbor features for the graph.
    """
    graph = dgl.load_graphs(DATADIR + graph_file_name)[0][0]
    graph: dgl.DGLGraph
    print(f"Graph info: {graph}")

    degree_feat = graph.in_degrees().unsqueeze_(1).float()
    risk_feat = count_risk_neighs(graph).unsqueeze_(1).float()

    origin_feat_name = ['degree', 'riskstat']
    edge_feat = torch.cat([degree_feat, risk_feat], dim=1)
    
    features_neigh, feat_names = feat_map(graph, edge_feat)
    features_neigh = torch.cat((edge_feat, features_neigh), dim=1).numpy()
    feat_names = origin_feat_name + feat_names
    features_neigh[np.isnan(features_neigh)] = 0.

    output_path = DATADIR + graph_file_name.replace('graph-', '') + "_neigh_feat.csv"
    features_neigh = pd.DataFrame(features_neigh, columns=feat_names)
    scaler = StandardScaler()
    features_neigh = pd.DataFrame(scaler.fit_transform(features_neigh), columns=features_neigh.columns)

    features_neigh.to_csv(output_path, index=False)

def count_risk_neighs(graph, risk_label=1):
    ret = []
    for center_idx in graph.nodes():
        neigh_idxs = graph.successors(center_idx)
        neigh_labels = graph.ndata['label'][neigh_idxs]
        risk_neigh_num = (neigh_labels == risk_label).sum()
        ret.append(risk_neigh_num)
    return torch.Tensor(ret)

def feat_map(graph, edge_feat):
    tensor_list = []
    feat_names = []
    for idx in tqdm(range(graph.num_nodes())):
        neighs_1_of_center = k_neighs(graph, idx, 1, "in")
        neighs_2_of_center = k_neighs(graph, idx, 2, "in")

        tensor = torch.FloatTensor([
            edge_feat[neighs_1_of_center, 0].sum().item(),
            edge_feat[neighs_2_of_center, 0].sum().item(),
            edge_feat[neighs_1_of_center, 1].sum().item(),
            edge_feat[neighs_2_of_center, 1].sum().item(),
        ])
        tensor_list.append(tensor)

    feat_names = ["1hop_degree", "2hop_degree", "1hop_riskstat", "2hop_riskstat"]
    tensor_list = torch.stack(tensor_list)
    return tensor_list, feat_names

if __name__ == "__main__":
    set_seed(42)
    data_file = os.path.join(DATADIR, 'DATA1.csv')
    process_graph_data(data_file, "graph-DATA1.bin")
