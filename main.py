import argparse
import torch
import numpy as np
from dataloader import MoleculeDataset
from torch_geometric.loader import DataLoader
from model import GNN
from tqdm import tqdm
import os
import glob
from graphics_utils import save_plot
from utils import calculate_metrics

parser = argparse.ArgumentParser(description = "Graph Neural Networks for HIV Inhibitor molecule Identification.")
parser.add_argument('-lr', '--learning_rate', default = 1e-2)
parser.add_argument('-ep', '--epoch', default = 30)
parser.add_argument('-m', '--mode', default="full") # mode: train, test, full
args = parser.parse_args()

lr = args.learning_rate
total_epoch = int(args.epoch)
MODE = args.mode.lower()

train_dataset = MoleculeDataset(mode="train")
test_dataset = MoleculeDataset(mode="test")
model_edge_dim = train_dataset[0].edge_attr.shape[1]
gnn_model = GNN(feature_size=train_dataset[0].x.shape[1], edge_dim=model_edge_dim)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

weight = torch.tensor([1.3], dtype=torch.float32)
loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=weight)
optimizer = torch.optim.SGD(gnn_model.parameters(), lr=lr, momentum=0.8, weight_decay=0.0001)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)

def infer():
    print("Running Inference...")
    weight_filename = glob.glob("./weights/*pt")[0]
    gnn_model.load_state_dict(torch.load(weight_filename))
    gnn_model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc= "Inferencing..."):
            pred = gnn_model(batch.x.float(), 
                            batch.edge_attr.float(),
                            batch.edge_index, 
                            batch.batch)
            all_preds.append(np.rint(torch.sigmoid(pred).cpu().detach().numpy()))
            all_labels.append(batch.y.cpu().detach().numpy())
        all_preds = np.concatenate(all_preds).ravel()
        all_labels = np.concatenate(all_labels).ravel()
    calculate_metrics(all_preds, all_labels)


def test():
    gnn_model.eval()
    running_loss = 0.0
    step = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc= "Testing..."):
            pred = gnn_model(batch.x.float(), 
                            batch.edge_attr.float(),
                            batch.edge_index, 
                            batch.batch) 
            loss = loss_fn(torch.squeeze(pred), batch.y.float())
            # Update tracking
            running_loss += loss.item()
            step += 1
            all_preds.append(np.rint(torch.sigmoid(pred).cpu().detach().numpy()))
            all_labels.append(batch.y.cpu().detach().numpy())
        all_preds = np.concatenate(all_preds).ravel()
        all_labels = np.concatenate(all_labels).ravel()
        avg_running_loss = running_loss/step
        return avg_running_loss, all_preds, all_labels


def train():
    max_test_loss = 100000
    train_avg_running_loss, test_avg_running_loss  = [], []
    all_preds, all_labels = [], []
    for epoch in tqdm(range(1, total_epoch+1), desc= "Training Epoch"):
        running_loss = 0.0
        step = 0
        gnn_model.train()
        for batch in train_loader:
            # Reset gradients
            optimizer.zero_grad() 
            # Passing the node features and the connection info
            pred = gnn_model(batch.x.float(), 
                                    batch.edge_attr.float(),
                                    batch.edge_index, 
                                    batch.batch) 
            # Calculating the loss and gradients
            loss = loss_fn(torch.squeeze(pred), batch.y.float())
            loss.backward()  
            optimizer.step()  
            # Update tracking
            running_loss += loss.item()
            step += 1
        train_avg_running_loss.append(running_loss/step)
        scheduler.step()
        current_avg_test_loss, all_preds, all_labels = test()
        if current_avg_test_loss <= max_test_loss:
            max_test_loss = current_avg_test_loss
            os.system("rm ./weights/*pt")
            torch.save(gnn_model.state_dict(), "./weights/"+str(epoch)+".pt")
            print("\n >>> Saved weights at epoch: ",epoch)
            calculate_metrics(all_preds, all_labels)
        test_avg_running_loss.append(current_avg_test_loss)
    save_plot(train_avg_running_loss, test_avg_running_loss)


if MODE in ["train","full"]:
    train()
if MODE in ["full", "infer","test"]:
    infer()