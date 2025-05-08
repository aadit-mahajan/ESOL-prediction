import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import pandas as pd
import numpy as np
from rdkit import Chem

class ESOLDataset(Dataset):
    def __init__(self, dataframe):
        self.df = dataframe

    def __len__(self):
        return len(self.df)
    
    def mol_to_graph(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES string: {smiles}")

        edge_index = []
        node_features = []
        edge_features = []

        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()

            bond_type = bond.GetBondTypeAsDouble()  # 1.0, 2.0, 3.0, 1.5 (aromatic)
            is_conjugated = bond.GetIsConjugated()
            is_in_ring = bond.IsInRing()

            edge_feat = [bond_type, float(is_conjugated), float(is_in_ring)]

            # Add both directions
            edge_index.append((i, j))
            edge_index.append((j, i))
            edge_features.append(edge_feat)
            edge_features.append(edge_feat)

        # one hot encoding of atom types
        for atom in mol.GetAtoms():
            atom_type = atom.GetAtomicNum()
            aromatic = atom.GetIsAromatic()
            formal_charge = atom.GetFormalCharge()
            hybridization = int(atom.GetHybridization())
            degree = atom.GetDegree()
            valence = atom.GetExplicitValence()

            node_features.append([atom_type, aromatic, formal_charge, hybridization, degree, valence])  

        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_features, dtype=torch.float)
        x = torch.tensor(node_features, dtype=torch.float)

        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # fetch graph data
        smiles = row['SMILES']
        graph_data = self.mol_to_graph(smiles)

        # Get cls and mean embeddings
        cls_embedding = row['X_cls']
        mean_embedding = row['X_mean']
        cls_embedding = torch.tensor(cls_embedding, dtype=torch.float)
        mean_embedding = torch.tensor(mean_embedding, dtype=torch.float)

        # Get labels 
        label = torch.tensor(row['label'], dtype=torch.float)

        return graph_data, cls_embedding, mean_embedding, label

def main():
    dataset_path = "./esol.parquet"
    df = pd.read_parquet(dataset_path)

    df['X_cls'] = df['X_cls'].apply(lambda x: np.array(x))
    df['X_mean'] = df['X_mean'].apply(lambda x: np.array(x))

    # Create dataset
    dataset = ESOLDataset(df)
    print("node feature dims", dataset[0][0].x.shape[1])
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # sample one datapoint from dataset and print node feat dims
    for graph_data, cls_embed, mean_embed, labels in dataloader:
        print(f"Graph data: {graph_data}")
        print(f"CLS embedding shape: {cls_embed.shape}")
        print(f"Mean embedding shape: {mean_embed.shape}")
        print(f"Labels shape: {labels.shape}")
        break

if __name__ == "__main__":
    main()