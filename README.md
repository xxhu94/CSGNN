# CSGNN

A PyTorch implementation for the paper below:  
**Cost Sensitive GNN-based Imbalanced Learning for Mobile Social Network Fraud Detection**

## Running CSGNN
To run the code, you need to have at least Python 3.7 or later versions.  
1.In CSGNN/data directory，run`unzip BUPT.zip` and `unzip Sichuan.zip` to unzip the datasets;  
2.Run `python data_process.py` to generate Sichuan and BUPT dataset in DGL;  
3.-Run `python main.py --config ./config/csgnn_sichuan.yml` to run CSGNN with default settings on Sihcuan dataset;  
-Run `python main.py --config ./config/csgnn_bupt.yml` to run CSGNN with default settings on BUPT dataset.   

## Repo Structure
The repository is organized as follows:
- `data_process.py`: convert raw node features and adjacency matrix to DGL dataset;
- `main.py`:  training and testing CSGNN;
- `model.py`: CSGNN model implementations;
- `utils.py`: utility functions.  

## Citation

```
@article{hu2023cost,
  title={Cost Sensitive GNN-based Imbalanced Learning for Mobile Social Network Fraud Detection},
  author={Hu, Xinxin and Chen, Haotian and Chen, Hongchang and Liu, Shuxin and Li, Xing and Zhang, Shibo and Wang, Yahui and Xue, Xiangyang},
  journal={IEEE Transactions on Computational Social Systems },
  year={2023},
  doi={10.1109/TCSS.2023.3302651}
}
```

  
