import torch
import numpy as np
from rdkit import Chem
from torch_geometric.data import Dataset, Data
import pandas as pd
from tqdm import tqdm
from utils import get_node_features, get_edge_features, get_adjacency_info


class MoleculeDataset(Dataset):

    def __init__(self, root_path="./", mode="train", transform=None, pre_transform=None):
        self.mode = mode
        self.root_path = root_path
        self.filename = self.root_path+"dataset/raw/HIV_"+self.mode+".csv"
        super(MoleculeDataset, self).__init__(root_path, transform, pre_transform)

    @property
    def raw_file_names(self):
        """ If this file exists in raw_dir, the download is not triggered.
            (The download func. is not implemented here)  
        """
        return self.filename

    @property
    def processed_file_names(self):
        """ If these files are found in raw_dir, processing is skipped"""
        self.data = pd.read_csv(self.filename).reset_index()

        return [self.root_path+"dataset/processed/"+self.mode+"/"+str(i)+".pt" for i in list(self.data.index)]

    def download(self):
        pass

    def _get_labels(self, label):
        label = np.asarray([label])
        return torch.tensor(label, dtype=torch.int64)
    
    def len(self):
        return self.data.shape[0]

    def get(self, idx):
        return torch.load(self.root_path+"dataset/processed/"+self.mode+"/"+str(idx)+".pt")

    def process(self):
        self.data = pd.read_csv(self.filename)
        for idx, mol in tqdm(self.data.iterrows(), total=self.data.shape[0]):
            mol_obj = Chem.MolFromSmiles(mol["smiles"])
            node_features = get_node_features(mol_obj)
            edge_features = get_edge_features(mol_obj)
            edge_index = get_adjacency_info(mol_obj)
            label = self._get_labels(mol["HIV_active"])

            data = Data(x=node_features,
                        edge_index=edge_index,
                        edge_attr=edge_features,
                        y=label,
                        smiles=mol["smiles"]
                        )
            torch.save(data, self.root_path+"dataset/processed/"+self.mode+"/"+str(idx)+".pt")