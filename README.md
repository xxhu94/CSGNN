# Cost Sensitive GNN code 

## Repo Structure
The repository is organized as follows:
- `1.py`:用于代码片段测试，测试通过后写入程序;  
- `data_process.py`: convert raw node features and adjacency matrix to DGL dataset;
- `main.py`: GAT-COBO的代码，原本打算在这个基础上进行改写代码，后来发现不如重写;
- `test.py`: 整个主函数的重写;
- `model.py`: 模型：CAREGNN+Cost sensitive;
- `utils.py`: 各种功能组件，包括CAREGNN+Cost sensitive中需要的各种小函数.  


## Running baselines
You can find the baselines in `baselines` directory. For example, you can run Player2Vec using:
```bash
python Player2Vec_main.py 
```

  