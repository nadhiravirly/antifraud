from collections import defaultdict
import pandas as pd
import numpy as np
import torch
import dgl
import os
import random
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tqdm import tqdm

DATADIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data/")

def featmap_gen(tmp_df=None):
    """
    Feature engineering on DATA1 dataset.
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

def sparse_to_adjlist(sp_matrix, filename):
    """
    Convert sparse matrix to adjacency list and save to file.
    """
    homo_adj = sp_matrix + sp.eye(sp_matrix.shape[0])
    adj_lists = defaultdict(set)
    edges = homo_adj.nonzero()
    for index, node in enumerate(edges[0]):
        adj_lists[node].add(edges[1][index])
        adj_lists[edges[1][index]].add(node)
    with open(filename, 'wb') as file:
        pickle.dump(adj_lists, file)

def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def MinMaxScaling(data):
    mind, maxd = data.min(), data.max()
    return (data - mind) / (maxd - mind)

def k_neighs(graph: dgl.DGLGraph, center_idx: int, k: int, where: str, choose_risk: bool = False, risk_label: int = 1) -> torch.Tensor:
    """Return indices of k-hop neighbors with optional risk filtering."""
    if k == 1:
        neigh_idxs = graph.predecessors(center_idx) if where == "in" else graph.successors(center_idx)
    elif k == 2:
        subg = dgl.khop_in_subgraph(graph, center_idx, 2, store_ids=True)[0] if where == "in" else dgl.khop_out_subgraph(graph, center_idx, 2, store_ids=True)[0]
        neigh_idxs = subg.ndata[dgl.NID][subg.ndata[dgl.NID] != center_idx]
        neigh1s = graph.predecessors(center_idx) if where == "in" else graph.successors(center_idx)
        neigh_idxs = neigh_idxs[~torch.isin(neigh_idxs, neigh1s)]

    return neigh_idxs[graph.ndata['label'][neigh_idxs] == risk_label] if choose_risk else neigh_idxs

def count_risk_neighs(graph: dgl.DGLGraph, risk_label: int = 1) -> torch.Tensor:
    """
    Count the number of risk neighbors for each node.
    """
    ret = []
    for center_idx in graph.nodes():
        neigh_idxs = graph.successors(center_idx)
        ret.append((graph.ndata['label'][neigh_idxs] == risk_label).sum())
    return torch.Tensor(ret)

def feat_map(graph: dgl.DGLGraph, edge_feat: torch.Tensor):
    """Generate neighbor-based features for nodes."""
    tensor_list = []
    feat_names = ["1hop_degree", "2hop_degree", "1hop_riskstat", "2hop_riskstat"]
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
    return torch.stack(tensor_list), feat_names

def process_graph_data(data, output_graph_file):
    """Process DATA1 to create the graph and features."""
    data = featmap_gen(data.reset_index(drop=True))
    data.replace(np.nan, 0, inplace=True)
    data.to_csv(os.path.join(DATADIR, 'DATA1_full.csv'), index=None)

    data = data.reset_index(drop=True)
    alls, allt = [], []
    pair = ["cc_num", "merchant", "category", "city"]
    for column in pair:
        src, tgt = [], []
        edge_per_trans = 3
        for c_id, c_df in tqdm(data.groupby(column), desc=column):
            c_df = c_df.sort_values(by="Time")
            sorted_idxs = c_df.index
            src.extend([sorted_idxs[i] for i in range(len(c_df)) for j in range(edge_per_trans) if i + j < len(c_df)])
            tgt.extend([sorted_idxs[i + j] for i in range(len(c_df)) for j in range(edge_per_trans) if i + j < len(c_df)])
        alls.extend(src)
        allt.extend(tgt)

    g = dgl.graph((np.array(alls), np.array(allt)))
    for col in ["cc_num", "merchant", "category", "city"]:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col].apply(str).values)

    feat_data = data.drop("is_fraud", axis=1).apply(pd.to_numeric, errors='coerce').fillna(0)
    labels = data["is_fraud"]
    g.ndata['label'] = torch.from_numpy(labels.to_numpy()).to(torch.long)
    g.ndata['feat'] = torch.from_numpy(feat_data.to_numpy()).to(torch.float32)
    dgl.data.utils.save_graphs(DATADIR + output_graph_file, [g])

    generate_neighbor_features(g)

def generate_neighbor_features(graph):
    """Generate neighbor-based features for each node."""
    degree_feat = graph.in_degrees().unsqueeze_(1).float()
    risk_feat = count_risk_neighs(graph).unsqueeze_(1).float()

    edge_feat = torch.cat([degree_feat, risk_feat], dim=1)
    features_neigh, feat_names = feat_map(graph, edge_feat)
    features_neigh = torch.cat((edge_feat, features_neigh), dim=1).numpy()
    features_neigh = pd.DataFrame(features_neigh, columns=['degree', 'riskstat'] + feat_names)
    scaler = StandardScaler()
    features_neigh = pd.DataFrame(scaler.fit_transform(features_neigh), columns=features_neigh.columns)
    features_neigh.to_csv(os.path.join(DATADIR, "DATA1_neigh_feat.csv"), index=False)

if __name__ == "__main__":
    data = pd.read_csv(os.path.join(DATADIR, 'DATA1.csv'))
    process_graph_data(data, "graph-DATA1.bin")
