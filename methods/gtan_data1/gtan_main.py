import numpy as np
import dgl
import torch
import os
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold, train_test_split
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder
from dgl.dataloading import MultiLayerFullNeighborSampler, DataLoader
from torch.optim.lr_scheduler import MultiStepLR
from .gtan_model import GraphAttnModel
from . import early_stopper, load_lpa_subtensor

def gtan_main(feat_df, graph, train_idx, test_idx, labels, args, cat_features):
    device = args['device']
    graph = graph.to(device)
    oof_predictions = torch.zeros((len(feat_df), 2), dtype=torch.float32, device=device)
    test_predictions = torch.zeros((len(feat_df), 2), dtype=torch.float32, device=device)
    kfold = StratifiedKFold(n_splits=args['n_fold'], shuffle=True, random_state=args['seed'])

    y_target = labels.iloc[train_idx].values
    num_feat = torch.tensor(feat_df.values, dtype=torch.float32).to(device)
    cat_feat = {col: torch.tensor(feat_df[col].values, dtype=torch.long).to(device) for col in cat_features}
    labels = torch.tensor(labels.values, dtype=torch.long).to(device)
    loss_fn = nn.CrossEntropyLoss().to(device)

    for fold, (trn_idx, val_idx) in enumerate(kfold.split(feat_df.iloc[train_idx], y_target)):
        print(f'Training fold {fold + 1}')
        trn_ind = torch.tensor(np.array(train_idx)[trn_idx], dtype=torch.long).to(device)
        val_ind = torch.tensor(np.array(train_idx)[val_idx], dtype=torch.long).to(device)

        train_sampler = MultiLayerFullNeighborSampler(args['n_layers'])
        train_dataloader = DataLoader(graph, trn_ind, train_sampler, device=device, batch_size=args['batch_size'], shuffle=True)
        val_sampler = MultiLayerFullNeighborSampler(args['n_layers'])
        val_dataloader = DataLoader(graph, val_ind, val_sampler, device=device, batch_size=args['batch_size'], shuffle=False)

        model = GraphAttnModel(
            in_feats=feat_df.shape[1],
            hidden_dim=args['hid_dim'] // 4,
            n_classes=2,
            heads=[4] * args['n_layers'],
            activation=nn.PReLU(),
            n_layers=args['n_layers'],
            drop=args['dropout'],
            device=device,
            gated=args['gated'],
            ref_df=feat_df.iloc[train_idx],
            cat_features=cat_feat
        ).to(device)

        lr = args['lr'] * np.sqrt(args['batch_size'] / 1024)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=args['wd'])
        lr_scheduler = MultiStepLR(optimizer, milestones=[4000, 12000], gamma=0.3)

        earlystopper = early_stopper(patience=args['early_stopping'], verbose=True)

        for epoch in range(args['max_epochs']):
            train_loss_list = []
            model.train()
            for step, (input_nodes, seeds, blocks) in enumerate(train_dataloader):
                batch_inputs, batch_work_inputs, batch_labels, lpa_labels = load_lpa_subtensor(num_feat, cat_feat, labels, seeds, input_nodes, device)
                blocks = [block.to(device) for block in blocks]
                train_batch_logits = model(blocks, batch_inputs, lpa_labels, batch_work_inputs)

                mask = batch_labels != 2
                train_batch_logits = train_batch_logits[mask]
                batch_labels = batch_labels[mask]
                train_loss = loss_fn(train_batch_logits, batch_labels)
                
                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()
                lr_scheduler.step()
                train_loss_list.append(train_loss.item())

            val_loss_list, val_acc_list, val_all_list = 0, 0, 0
            model.eval()
            with torch.no_grad():
                for step, (input_nodes, seeds, blocks) in enumerate(val_dataloader):
                    batch_inputs, batch_work_inputs, batch_labels, lpa_labels = load_lpa_subtensor(num_feat, cat_feat, labels, seeds, input_nodes, device)
                    blocks = [block.to(device) for block in blocks]
                    val_batch_logits = model(blocks, batch_inputs, lpa_labels, batch_work_inputs)
                    oof_predictions[seeds] = val_batch_logits

                    mask = batch_labels != 2
                    val_batch_logits = val_batch_logits[mask]
                    batch_labels = batch_labels[mask]

                    val_loss_list += loss_fn(val_batch_logits, batch_labels).item() * batch_labels.size(0)
                    val_acc_list += (torch.argmax(val_batch_logits, dim=1) == batch_labels).sum().item()
                    val_all_list += batch_labels.size(0)

            val_accuracy = val_acc_list / val_all_list
            earlystopper.earlystop(val_loss_list / val_all_list, model)
            if earlystopper.is_earlystop:
                print("Early Stopping!")
                break

    # Evaluation on test data
    print("Training complete. Evaluating on test data...")
    test_sampler = MultiLayerFullNeighborSampler(args['n_layers'])
    test_dataloader = DataLoader(graph, torch.tensor(test_idx).to(device), test_sampler, device=device, batch_size=args['batch_size'])

    best_model = earlystopper.best_model.to(device)
    best_model.eval()
    with torch.no_grad():
        for input_nodes, seeds, blocks in test_dataloader:
            batch_inputs, batch_work_inputs, batch_labels, lpa_labels = load_lpa_subtensor(num_feat, cat_feat, labels, seeds, input_nodes, device)
            blocks = [block.to(device) for block in blocks]
            test_batch_logits = best_model(blocks, batch_inputs, lpa_labels, batch_work_inputs)
            test_predictions[seeds] = test_batch_logits

    print("Evaluation metrics on test set:")
    # Menghitung metrik AUC, F1, AP berdasarkan test_predictions
    # ...

def load_gtan_data(dataset: str, test_size: float):
    prefix = "data/"
    if dataset == "DATA1":
        cat_features = ["cc_num", "merchant", "category", "city"]
        df = pd.read_csv(prefix + "DATA1_full.csv")
        df = df.loc[:, ~df.columns.str.contains('Unnamed')]
        data = df[df["is_fraud"] <= 2].reset_index(drop=True)

        # Build graph structure
        alls, allt = [], []
        for column in ["cc_num", "merchant", "category", "city"]:
            src, tgt = [], []
            edge_per_trans = 3
            for c_id, c_df in data.groupby(column):
                c_df = c_df.sort_values(by="Time")
                sorted_idxs = c_df.index
                src.extend([sorted_idxs[i] for i in range(len(c_df) - edge_per_trans) for j in range(edge_per_trans)])
                tgt.extend([sorted_idxs[i + j] for i in range(len(c_df) - edge_per_trans) for j in range(edge_per_trans)])
            alls.extend(src)
            allt.extend(tgt)

        g = dgl.graph((alls, allt))
        for col in cat_features:
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col].astype(str))

        feat_data = data.drop(columns=["is_fraud"])
        labels = data["is_fraud"]

        train_idx, test_idx = train_test_split(
            range(len(labels)), stratify=labels, test_size=test_size, random_state=42
        )

        g.ndata['label'] = torch.tensor(labels.values, dtype=torch.long)
        g.ndata['feat'] = torch.tensor(feat_data.values, dtype=torch.float32)
        return feat_data, labels, train_idx, test_idx, g, cat_features
    else:
        raise ValueError(f"Dataset {dataset} not supported")
