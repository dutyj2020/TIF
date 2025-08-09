import torch
import numpy as np
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.utils import to_undirected
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class GNNModel(torch.nn.Module):
    def __init__(self, input_dim=192, hidden_dim=64, output_dim=32, num_classes=2):
        super(GNNModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_classes = num_classes
        self.input_linear = torch.nn.Linear(input_dim, hidden_dim)
        self.conv1 = GCNConv(hidden_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.fc1 = torch.nn.Linear(hidden_dim, output_dim)
        self.fc2 = torch.nn.Linear(output_dim, num_classes)
    def forward(self, data):
        x, edge_index = data.x.to(device), data.edge_index.to(device)
        edge_index = to_undirected(edge_index)
        x = self.input_linear(x)
        x = F.relu(x)
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = global_mean_pool(x, data.batch.to(device))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
state_dict = torch.load('gnn_model.pt', map_location=device)
if 'input_linear.weight' in state_dict:
    del state_dict['input_linear.weight']
if 'input_linear.bias' in state_dict:
    del state_dict['input_linear.bias']
model = GNNModel().to(device)
model.load_state_dict(state_dict, strict=False)
model.eval()
print("Model loaded with dynamic input layer.")
all_x_pooled = torch.load('last_x_pooled.pt')
print(f"Loaded {len(all_x_pooled)} batches of pooled data from last_x_pooled.pt")
class PooledGraphDataset(torch.utils.data.Dataset):
    def __init__(self, pooled_data):
        self.pooled_data = pooled_data
    def __len__(self):
        return len(self.pooled_data)
    def __getitem__(self, idx):
        x_pooled, adj_pooled = self.pooled_data[idx]
        if adj_pooled.dim() == 3:
            batch_size, num_nodes, _ = adj_pooled.size()
            expected_size = batch_size * num_nodes
            actual_size = adj_pooled.numel()
            if actual_size != expected_size * num_nodes:
                raise ValueError(f"Mismatch: adj_pooled has {actual_size} elements, "
                                 f"but expected {expected_size * num_nodes} elements after reshape.")
            x_pooled = x_pooled.view(expected_size, -1)
            adj_pooled = adj_pooled.view(expected_size, num_nodes)
        if adj_pooled.dim() != 2:
            raise ValueError(f"Adjacency matrix adj_pooled at index {idx} is not 2D, but {adj_pooled.dim()}D.")
        if x_pooled.size(0) != adj_pooled.size(0):
            raise ValueError(
                f"Node feature size {x_pooled.size(0)} does not match adjacency matrix size {adj_pooled.size(0)} at index {idx}")
        edge_index = adj_pooled.nonzero(as_tuple=False).t()
        valid_mask = edge_index < x_pooled.size(0)
        valid_edge_index = edge_index[:, valid_mask.all(dim=0)]
        if valid_edge_index.size(1) == 0:
            raise ValueError(f"All edges out of range for node feature size {x_pooled.size(0)} at index {idx}")
        return Data(x=x_pooled, edge_index=valid_edge_index)
def predict_with_new_data(model, data_loader):
    model.eval()
    all_preds = []
    all_confidences = []
    with torch.no_grad():
        for i, data in enumerate(data_loader):
            try:
                data = data.to(device)
                out = model(data)
                pred = out.argmax(dim=1).cpu().numpy()
                confidence = torch.exp(out.max(dim=1).values).cpu().numpy()
                all_preds.extend(pred)
                all_confidences.extend(confidence)
            except Exception as e:
                print(f"Error in batch {i}: {e}")
                break
    return all_preds, all_confidences
all_preds = []
all_confidences = []
for epoch_data in all_x_pooled:
    new_dataset = PooledGraphDataset(epoch_data)
    new_data_loader = DataLoader(new_dataset, batch_size=32, shuffle=False)
    preds, confidences = predict_with_new_data(model, new_data_loader)
    all_preds.extend(preds)
    all_confidences.extend(confidences)
avg_confidence = np.mean(all_confidences)
print(f"Overall Average Confidence: {avg_confidence:.4f}")