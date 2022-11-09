from model import CSGNN,CostMatrix
import torch
import dgl
import yaml
import argparse
import random, os, sys
import numpy as np
import pandas as pd
from dgl.data.utils import load_graphs, load_info
from utils import EarlyStopping,calculate_H,calculate_S
from sklearn.metrics import confusion_matrix,accuracy_score,recall_score,roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report
from imblearn.metrics import geometric_mean_score

# setting Beijing time
import logging
import datetime
def beijing(sec,what):
    beijing_time = datetime.datetime.now() + datetime.timedelta(hours=8)
    return beijing_time.timetuple()
logging.Formatter.converter = beijing
# logging setting
log_name=(datetime.datetime.now() + datetime.timedelta(hours=8)).strftime('%Y-%m-%d')
logging.basicConfig(
    format='%(asctime)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=24,
    # filename=log_name+'.log',
    # filemode='a'
    )

def get_config(config_path="./config/pcgnn_amazon.yml"):
    with open(config_path, "r") as setting:
        config = yaml.load(setting, Loader=yaml.FullLoader)
    return config

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default="/code/CSGNN/config/csgnn_sichuan.yml", help='path to the config file')
    # parser.add_argument('--config', type=str, default="/code/CSGNN/config/csgnn_bupt.yml", help='path to the config file')
    args = vars(parser.parse_args())
    return args

def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    dgl.seed(seed)
    dgl.random.seed(seed)


def gen_mask(g, train_rate, val_rate, IR, IR_set):
    labels = g.ndata['label']
    g.ndata['label'] = labels.long()
    labels = np.array(labels)
    n_nodes = len(labels)
    if IR_set == 0:
        index = list(range(n_nodes))
    # Unbalanced sampling based on IR
    else:
        fraud_index = np.where(labels == 1)[0].tolist()
        benign_index = np.where(labels == 0)[0].tolist()
        if len(np.unique(labels)) == 3:
            Courier_index = np.where(labels == 2)[0].tolist()
        if IR < (len(fraud_index) / len(benign_index)):
            number_sample = int(IR * len(benign_index))
            sampled_fraud_index = random.sample(fraud_index, number_sample)
            sampled_benign_index = benign_index
            if len(np.unique(labels)) == 3:
                sampled_Courier_index = random.sample(Courier_index, number_sample)
        else:
            number_sample = int(len(fraud_index) / IR)
            sampled_benign_index = random.sample(benign_index, number_sample)
            sampled_fraud_index = fraud_index
            if len(np.unique(labels)) == 3:
                sampled_Courier_index = Courier_index
        if len(np.unique(labels)) == 2:
            index = sampled_benign_index + sampled_fraud_index
        else:
            index = sampled_benign_index + sampled_fraud_index + sampled_Courier_index
        labels = labels[index]

    train_idx, val_test_idx, _, y_validate_test = train_test_split(index, labels, stratify=labels,
                                                                   train_size=train_rate, test_size=1 - train_rate,
                                                                   random_state=2, shuffle=True)
    val_idx, test_idx, _, _ = train_test_split(val_test_idx, y_validate_test, train_size=val_rate / (1 - train_rate),
                                               test_size=1 - val_rate / (1 - train_rate),
                                               random_state=2, shuffle=True)
    train_mask = torch.zeros(n_nodes, dtype=torch.bool)
    val_mask = torch.zeros(n_nodes, dtype=torch.bool)
    test_mask = torch.zeros(n_nodes, dtype=torch.bool)
    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True
    g.ndata['train_mask'] = train_mask
    g.ndata['val_mask'] = val_mask
    g.ndata['test_mask'] = test_mask
    return g, train_idx


def load_data(args):
    if args.dataset == 'BUPT':
        dataset, _ = load_graphs("/code/CSGNN/data/BUPT_tele.bin")  # glist will be [g1]
        num_classes = load_info("/code/CSGNN/data/BUPT_tele.pkl")['num_classes']
        # {0: 99861, 1: 8448, 2: 8074}
        graph,_ = gen_mask(dataset[0], args.train_size, args.val_size, args.IR, args.IR_set)
        feat = graph.ndata['feat'].float().to(device)

    # 引入Sichuan数据集
    elif args.dataset == 'Sichuan':
        dataset, _ = load_graphs("/code/CSGNN/data/Sichuan_tele.bin")  # glist will be [g1]
        num_classes = load_info("/code/CSGNN/data/Sichuan_tele.pkl")['num_classes']
        # {0: 4144, 1: 1962}
        graph,_ = gen_mask(dataset[0], args.train_size, args.val_size, args.IR, args.IR_set)
        feat = graph.ndata['feat'].float().to(device)


    else:
        sys.exit("Error dataset name!")

    for e in graph.etypes:
        graph = graph.int().to(device)
        dgl.remove_self_loop(graph, etype=e)
        dgl.add_self_loop(graph, etype=e)

    return graph, num_classes, feat


def main(args):
    # For batch testing partitions
    if args.blank == 1:
        logging.log(24, f"---------------------------")
        sys.exit()

    # load data
    graph, num_classes, feat = load_data(args)

    labels = graph.ndata['label'].long().squeeze().to(device)
    train_mask = graph.ndata['train_mask']
    val_mask = graph.ndata['val_mask']
    test_mask = graph.ndata['test_mask']

    train_idx = torch.nonzero(train_mask, as_tuple=False).squeeze(1).to(device)
    val_idx = torch.nonzero(val_mask, as_tuple=False).squeeze(1).to(device)
    test_idx = torch.nonzero(test_mask, as_tuple=False).squeeze(1).to(device)

    # Reinforcement learning module only for positive training nodes
    rl_idx = torch.nonzero(train_mask.to(device) & labels.bool(), as_tuple=False).squeeze(1).int()

    # creat model
    model = CSGNN(in_dim=feat.shape[-1],
                   num_classes=num_classes,
                   hid_dim=args.hid_dim,
                   num_layers=args.num_layers,
                   activation=torch.tanh,
                   step_size=args.step_size,
                   edges=graph.canonical_etypes)
    model = model.to(device)

    cost=CostMatrix(num_classes,labels[train_mask].cpu(),num_classes)

    # Create training components
    # _, cnt = torch.unique(labels, return_counts=True)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer_GNN = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr_GNN,
                                     weight_decay=args.weight_decay_GNN)
    optimizer_Cost = torch.optim.Adam(filter(lambda p: p.requires_grad, cost.parameters()), lr=args.lr_Cost,
                                      weight_decay=args.weight_decay_Cost)
    if args.early_stop:
        stopper = EarlyStopping(args.patience)

    # train the model
    for epoch in range(args.max_epoch):
        model.train()

        # calculate model logits
        logits_gnn, logits_sim = model(graph, feat)

        # cost matrix logits
        cost_table=cost.cost_table_calc()
        logits_gnn=cost.cost_logits(logits_gnn,labels,cost_table).to(device)

        # GNN loss, Eq.(41)
        loss_GNN = loss_fn(logits_gnn[train_idx], labels[train_idx]) + \
                  args.sim_weight * loss_fn(logits_sim[train_idx], labels[train_idx])


        # calculate validate error,parameter
        # Eq.(14)
        H=calculate_H(labels[train_idx])
        # Eq.(17)
        S=calculate_S(logits_gnn[val_idx],labels[val_idx])
        # Eq.(18)
        R=confusion_matrix(labels[val_mask].cpu().detach().numpy(), logits_gnn.data[val_idx].argmax(dim=1).cpu(),normalize='all')
        # Eq.(19)
        T=torch.mul(torch.mul(args.cost_weight*H,torch.tensor(S)),torch.tensor(R))

        val_err = 1 - accuracy_score(labels[val_mask].cpu().detach().numpy(), logits_gnn.data[val_idx].argmax(dim=1).cpu())
        # cost loss, Eq.(20)
        loss_cost = torch.pow(torch.norm(T-cost.cost_matrix),2)+val_err  # 默认求2范数


        tr_recall = recall_score(labels[train_idx].cpu(), logits_gnn.data[train_idx].argmax(dim=1).cpu(),average='macro')
        # calculate train AUC
        if num_classes == 2:
            # for binary classes
            tr_auc = roc_auc_score(labels[train_idx].cpu(), torch.softmax(logits_gnn.data[train_idx].cpu(),dim=1)[:,1],average='macro')
        else:
            # for multi classes
            tr_auc = roc_auc_score(labels[train_idx].cpu(), torch.softmax(logits_gnn.data[train_idx].cpu(),dim=1),average='macro',multi_class='ovo')

        # validation
        val_loss = loss_fn(logits_gnn[val_idx], labels[val_idx]) + \
                   args.sim_weight * loss_fn(logits_sim[val_idx], labels[val_idx])
        val_recall = recall_score(labels[val_idx].cpu(), logits_gnn.data[val_idx].argmax(dim=1).cpu(),average='macro')

        # calculate validation AUC
        if num_classes == 2:
            val_auc = roc_auc_score(labels[val_idx].cpu(), torch.softmax(logits_gnn.data[val_idx].cpu(), dim=1)[:, 1],
                                    average='macro')
        else:
            val_auc = roc_auc_score(labels[val_idx].cpu(), torch.softmax(logits_gnn.data[val_idx].cpu(), dim=1),
                                    average='macro', multi_class='ovo')

        # GNN parameter backward
        optimizer_GNN.zero_grad()
        loss_GNN.backward()
        optimizer_GNN.step()

        # cost matrix parameter backward
        optimizer_Cost.zero_grad()
        loss_cost.backward()
        optimizer_Cost.step()

        # Print out performance
        print("Epoch {}, Train: Recall: {:.4f} AUC: {:.4f} Loss: {:.4f} | Val: Recall: {:.4f} AUC: {:.4f} Loss: {:.4f} "
              .format(epoch, tr_recall, tr_auc, loss_GNN.item(), val_recall, val_auc, val_loss.item()))

        # Adjust p value with reinforcement learning module
        model.RLModule(graph, epoch, rl_idx)

        if args.early_stop:
            if stopper.step(val_auc, model,epoch):
                break

    # test after all epoch
    model.eval()
    if args.early_stop:
        model.load_state_dict(torch.load('es_checkpoint.pt'))

    # forward
    logits_gnn, logits_sim = model.forward(graph, feat)
    cost_table = cost.cost_table_calc()
    logits_gnn = cost.cost_logits(logits_gnn, labels, cost_table).to(device)
    logp = logits_gnn
    test_h=torch.argmax(logp[test_idx],dim=1)

    if np.isnan(logp[test_mask].cpu().detach().numpy()).any() == True:
        forest_auc=0.
    else:
        if num_classes == 2:
            forest_auc = roc_auc_score(labels[test_mask].cpu().detach().numpy(), torch.softmax(logp[test_mask].cpu(), dim=1)[:, 1].detach().numpy(),
                                     average='macro')
        else:
            forest_auc = roc_auc_score(labels[test_mask].cpu().detach().numpy(), torch.softmax(logp[test_mask].cpu(), dim=1).detach().numpy(),
                                     average='macro', multi_class='ovo')


    test_gmean = geometric_mean_score(labels[test_mask].cpu(), test_h.cpu())
    test_recall = recall_score(labels[test_idx].cpu().detach().numpy(), test_h.cpu().detach().numpy(), average='macro')
    print("Test set results:",
          "\nAuc= {:.4f}".format(forest_auc),
          "G-mean= {:.4f}".format(test_gmean),
          "Recall= {:.4f}".format(test_recall)
          )



if __name__ == '__main__':

    cfg = get_args()
    config = get_config(cfg['config'])
    args = argparse.Namespace(**config)
    print(args)
    if args.gpu >= 0 and torch.cuda.is_available():
        device = 'cuda:{}'.format(args.gpu)
    else:
        device = 'cpu'
    setup_seed(42)
    main(args)
