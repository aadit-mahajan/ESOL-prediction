## ESOL prediction project

This project aims to train a two-track model on the MoleculeNet ESOL dataset to predict the solubilities of molecules from their SMILES data.

It uses a two-track approach to tackle this problem:
1. ChemBERTa embeddings of the SMILES string for each molecule 
2. GNN embeddings for each molecule to capture structural data. 

GNN embeddings are derived from three different models for embedding graph data: 
1. Standard Graph Attention Network to capture basic connectivity importances.
2. Deep GAT network which implements GAT with residual connections. 
3. Graph Isomorphism Network which captures fine-grained structural patterns.
4. MFConv (Molecular Fingerprint Convolution) which has been shown to featurize molecular graphs really well. 

The performance metrics used to assess the prediction power of the model were RMSE, MAE, R², Pearson r, and Spearman ρ.
                
The plots for the performance assessment are added to the plots folder. 

Credits:     
MoleculeNet ESOL dataset:       
https://huggingface.co/datasets/scikit-fingerprints/MoleculeNet_ESOL        

AqSolDB: A curated aqueous solubility dataset:      
Nature Scientific Data - https://doi.org/10.1038/s41597-019-0151-1
link to dataset: https://www.kaggle.com/datasets/sorkun/aqsoldb-a-curated-aqueous-solubility-dataset
