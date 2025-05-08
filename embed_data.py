import pandas as pd
from datasets import load_dataset
from tqdm import tqdm
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer
from rdkit import Chem

def featurize_ChemBERTa(smiles_list, padding=True):
    embeddings_cls = torch.zeros(len(smiles_list), 600)
    embeddings_mean = torch.zeros(len(smiles_list), 600)
    chemberta = AutoModelForMaskedLM.from_pretrained("DeepChem/ChemBERTa-77M-MTR")
    tokenizer = AutoTokenizer.from_pretrained("DeepChem/ChemBERTa-77M-MTR")

    with torch.no_grad():
        for i, smiles in enumerate(tqdm(smiles_list)):
            encoded_input = tokenizer(smiles, return_tensors="pt",padding=padding,truncation=True)
            model_output = chemberta(**encoded_input)
            
            embedding = model_output[0][::,0,::]
            embeddings_cls[i] = embedding
            
            embedding = torch.mean(model_output[0],1)
            embeddings_mean[i] = embedding
            
    return embeddings_cls.numpy(), embeddings_mean.numpy()

def check_smiles(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False
        else:
            return True
    except:
        return False

def main():
    ds = load_dataset("scikit-fingerprints/MoleculeNet_ESOL")
    dataset = pd.DataFrame(ds['train'])

    # Featurize the dataset
    X_cls, X_mean = featurize_ChemBERTa(dataset['SMILES'].tolist(), padding=True)

    # save the dataset as parquet
    dataset['X_cls'] = list(X_cls)
    dataset['X_mean'] = list(X_mean)
    dataset.to_parquet("esol.parquet", index=False)

    # loading the aqsolDB dataset
    aqsoldb = pd.read_csv("curated-solubility-dataset.csv")
    dataset = aqsoldb[['SMILES', 'Solubility']]
    dataset = dataset.rename(columns={"Solubility": "label"})

    # check the smiles data for validity
    dataset['valid'] = dataset['SMILES'].apply(check_smiles)
    dataset = dataset[dataset['valid'] == True]
    print(f"Number of invalid SMILES found: {len(aqsoldb) - len(dataset)}")
    dataset = dataset.drop(columns=['valid'])

    # Featurize the dataset
    X_cls, X_mean = featurize_ChemBERTa(dataset['SMILES'].tolist(), padding=True)
    dataset['X_cls'] = list(X_cls)
    dataset['X_mean'] = list(X_mean)

    # save the dataset as parquet
    dataset.to_parquet("aqsoldb.parquet", index=False)


if __name__ == '__main__':
    main()