import pandas as pd
from datasets import load_dataset
from tqdm import tqdm
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

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

def main():
    ds = load_dataset("scikit-fingerprints/MoleculeNet_ESOL")
    dataset = pd.DataFrame(ds['train'])

    # Featurize the dataset
    X_cls, X_mean = featurize_ChemBERTa(dataset['SMILES'].tolist(), padding=True)

    # save the dataset as parquet
    dataset['X_cls'] = list(X_cls)
    dataset['X_mean'] = list(X_mean)
    dataset.to_parquet("esol.parquet", index=False)

if __name__ == '__main__':
    main()