# Cost Sensitive GNN code 

## Repo Structure
The repository is organized as follows:
- `data_process.py`: convert raw node features and adjacency matrix to DGL dataset;
- `main.py`: CSGNN主函数
- `model.py`: 模型：CAREGNN+Cost sensitive;
- `utils.py`: 各种功能组件，包括CAREGNN+Cost sensitive中需要的各种小函数.  


## Running baselines
You can find the baselines in `baselines` directory. For example, you can run Player2Vec using:
```bash
python Player2Vec_main.py 
```

  