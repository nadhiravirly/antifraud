import copy
import torch

def load_lpa_subtensor(node_feat, work_node_feat, labels, seeds, input_nodes, device):
    """
    Load a subset of tensor data for LPA processing, adjusted for DATA1.
    
    Args:
        node_feat (torch.Tensor): The main node features.
        work_node_feat (dict): Dictionary of additional features for nodes.
        labels (torch.Tensor): Labels tensor.
        seeds (torch.Tensor): Seed nodes for batch processing.
        input_nodes (torch.Tensor): Input nodes for batch.
        device (str): Device to load the tensors onto (e.g., 'cpu' or 'cuda').
    
    Returns:
        tuple: batch_inputs, batch_work_inputs, batch_labels, propagate_labels.
    """
    # Load main features for input nodes and transfer to device
    batch_inputs = node_feat[input_nodes].to(device)
    
    # Process categorical features
    batch_work_inputs = {
        i: work_node_feat[i][input_nodes].to(device) 
        for i in work_node_feat if i not in {"is_fraud"}
    }
    
    # Extract batch labels and propagate labels
    batch_labels = labels[seeds].to(device)
    train_labels = copy.deepcopy(labels)
    propagate_labels = train_labels[input_nodes]
    propagate_labels[:seeds.shape[0]] = 2  # Use 2 for unknown/fill values in LPA
    
    return batch_inputs, batch_work_inputs, batch_labels, propagate_labels.to(device)
