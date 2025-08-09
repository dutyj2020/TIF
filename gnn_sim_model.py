import torch
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from torch_geometric.datasets import TUDataset
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.utils import to_undirected
dataset = TUDataset(root='data/DD', name='DD')
dataset = dataset.shuffle()
train_dataset = dataset[:900]
test_dataset = dataset[900:]
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
class GNNModel(torch.nn.Module):
    def __init__(self):
        super(GNNModel, self).__init__()
        self.input_linear = torch.nn.Linear(dataset.num_node_features, 64)
        self.conv1 = GCNConv(64, 64)
        self.conv2 = GCNConv(64, 64)
        self.fc1 = torch.nn.Linear(64, 32)
        self.fc2 = torch.nn.Linear(32, dataset.num_classes)
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        edge_index = to_undirected(edge_index)
        x = self.input_linear(x)
        x = F.relu(x)
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = global_mean_pool(x, batch)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
model = GNNModel()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()
def train():
    model.train()
    for data in train_loader:
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
def test(loader):
    model.eval()
    correct = 0
    total = 0
    all_confidences = []
    with torch.no_grad():
        for data in loader:
            out = model(data)
            pred = out.argmax(dim=1)
            correct += (pred == data.y).sum().item()
            total += data.num_graphs
            confidences = torch.exp(out.max(dim=1).values).tolist()
            all_confidences.extend(confidences)
    accuracy = correct / total
    avg_confidence = sum(all_confidences) / len(all_confidences)
    return accuracy, avg_confidence
for epoch in range(1, 6):
    train()
    train_acc, train_confidence = test(train_loader)
    test_acc, test_confidence = test(test_loader)
    print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Train Confidence: {train_confidence:.4f}, '
          f'Test Acc: {test_acc:.4f}, Test Confidence: {test_confidence:.4f}')
torch.save(model.state_dict(), 'gnn_model.pt')
print("Model saved as gnn_model.pt")
model = GNNModel()
model.load_state_dict(torch.load('gnn_model.pt'))
model.eval()
print("Model loaded from gnn_model.pt")