from model import SENGNN,CostMatrix
import torch
import dgl
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

# 修改日志记录为北京时间
import logging
import datetime
def beijing(sec,what):
    beijing_time = datetime.datetime.now() + datetime.timedelta(hours=8)
    return beijing_time.timetuple()
logging.Formatter.converter = beijing
# logging基础配置
log_name=(datetime.datetime.now() + datetime.timedelta(hours=8)).strftime('%Y-%m-%d')
logging.basicConfig(
    format='%(asctime)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=24,
    filename=log_name+'.log',
    filemode='a'
    )


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
        dataset, _ = load_graphs("/code/AdaGAT-Dgl/data/BUPT_tele.bin")  # glist will be [g1]
        num_classes = load_info("/code/AdaGAT-Dgl/data/BUPT_tele.pkl")['num_classes']
        # {0: 99861, 1: 8448, 2: 8074}
        graph,_ = gen_mask(dataset[0], args.train_size, args.val_size, args.IR, args.IR_set)
        feat = graph.ndata['feat'].float().to(device)

    # 引入Sichuan数据集
    elif args.dataset == 'Sichuan':
        dataset, _ = load_graphs("/code/AdaGAT-Dgl/data/Sichuan_tele.bin")  # glist will be [g1]
        num_classes = load_info("/code/AdaGAT-Dgl/data/Sichuan_tele.pkl")['num_classes']
        # {0: 4144, 1: 1962}
        graph,_ = gen_mask(dataset[0], args.train_size, args.val_size, args.IR, args.IR_set)
        feat = graph.ndata['feat'].float().to(device)

    elif args.dataset == 'Citeseer':
        dataset = dgl.data.CiteseerGraphDataset()
        # graph = dataset[0]
        graph,_ = gen_mask(dataset[0], args.train_size, args.val_size, args.IR, args.IR_set)
        num_classes = dataset.num_classes
        feat = graph.ndata['feat'].float().to(device)

    else:
        # Load dataset
        dataset = dgl.data.FraudDataset(args.dataset, train_size=0.4)
        # dataset = dgl.data.CiteseerGraphDataset()
        graph = dataset[0]
        num_classes = dataset.num_classes
        feat = graph.ndata['feature'].float().to(device)

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
    model = SENGNN(in_dim=feat.shape[-1],
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

        # cross entropy loss
        loss_GNN = loss_fn(logits_gnn[train_idx], labels[train_idx]) + \
                  args.sim_weight * loss_fn(logits_sim[train_idx], labels[train_idx])


        # calculate validate error,parameter
        H=calculate_H(labels[train_idx])

        S=calculate_S(logits_gnn[val_idx],labels[val_idx])

        R=confusion_matrix(labels[val_mask].cpu().detach().numpy(), logits_gnn.data[val_idx].argmax(dim=1).cpu(),normalize='all')

        T=torch.mul(torch.mul(args.cost_weight*H,torch.tensor(S)),torch.tensor(R))

        val_err = 1 - accuracy_score(labels[val_mask].cpu().detach().numpy(), logits_gnn.data[val_idx].argmax(dim=1).cpu())

        loss_cost = torch.pow(torch.norm(T-cost.cost_matrix),2)+val_err  # 默认求2范数


        tr_recall = recall_score(labels[train_idx].cpu(), logits_gnn.data[train_idx].argmax(dim=1).cpu(),average='macro')
        # 计算训练AUC
        if num_classes == 2:
            # 二类别使用
            tr_auc = roc_auc_score(labels[train_idx].cpu(), torch.softmax(logits_gnn.data[train_idx].cpu(),dim=1)[:,1],average='macro')
        else:
            # 多分类使用
            tr_auc = roc_auc_score(labels[train_idx].cpu(), torch.softmax(logits_gnn.data[train_idx].cpu(),dim=1),average='macro',multi_class='ovo')

        # validation
        val_loss = loss_fn(logits_gnn[val_idx], labels[val_idx]) + \
                   args.sim_weight * loss_fn(logits_sim[val_idx], labels[val_idx])
        val_recall = recall_score(labels[val_idx].cpu(), logits_gnn.data[val_idx].argmax(dim=1).cpu(),average='macro')

        # 计算验证AUC
        # 之所以使用logits_gnn.data是因为logits_gnn是一个tensor数据类型，里面有.data方法，可以取得tensor的数据值
        if num_classes == 2:
            val_auc = roc_auc_score(labels[val_idx].cpu(), torch.softmax(logits_gnn.data[val_idx].cpu(), dim=1)[:, 1],
                                    average='macro')
        else:
            # 多分类专用
            val_auc = roc_auc_score(labels[val_idx].cpu(), torch.softmax(logits_gnn.data[val_idx].cpu(), dim=1),
                                    average='macro', multi_class='ovo')

        # 临时test
        tes_recall = recall_score(labels[test_idx].cpu(), logits_gnn.data[test_idx].argmax(dim=1).cpu(),average='macro')
        if num_classes == 2:
            tes_auc = roc_auc_score(labels[test_idx].cpu(), torch.softmax(logits_gnn.data[test_idx].cpu(), dim=1)[:, 1],
                                    average='macro')
        else:
            # 多分类专用
            tes_auc = roc_auc_score(labels[test_idx].cpu(), torch.softmax(logits_gnn.data[test_idx].cpu(), dim=1),
                                    average='macro', multi_class='ovo')


        # model parameter backward
        # GNN梯度计算，和参数优化
        optimizer_GNN.zero_grad()
        loss_GNN.backward()
        optimizer_GNN.step()

        # cost matrix parameter backward
        optimizer_Cost.zero_grad()
        loss_cost.backward()
        optimizer_Cost.step()

        # Print out performance
        print("Epoch {}, Train: Recall: {:.4f} AUC: {:.4f} Loss: {:.4f} | Val: Recall: {:.4f} AUC: {:.4f} Loss: {:.4f} | Tes: Recall: {:.4f} AUC: {:.4f} "
              .format(epoch, tr_recall, tr_auc, loss_GNN.item(), val_recall, val_auc, val_loss.item(), tes_recall, tes_auc))

        # calculate val error

        # if val_err_new > val_err:
        #     lr_Cost =lr_Cost*0.01

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
    # 增加测试部分=============================================================== #

    # model.eval()
    logp = logits_gnn
    test_h=torch.argmax(logp[test_idx],dim=1)

    # 计算Auc,区分2类别和其他类别
    if np.isnan(logp[test_mask].cpu().detach().numpy()).any() == True:
        forest_auc=0.
    else:
        if num_classes == 2:
            forest_auc = roc_auc_score(labels[test_mask].cpu().detach().numpy(), torch.softmax(logp[test_mask].cpu(), dim=1)[:, 1].detach().numpy(),
                                     average='macro')
        else:
            # 寻找ndarray中nan值位置：np.argwhere(np.isnan(x))
            forest_auc = roc_auc_score(labels[test_mask].cpu().detach().numpy(), torch.softmax(logp[test_mask].cpu(), dim=1).detach().numpy(),
                                     average='macro', multi_class='ovo')

    # 调用sklearn中的metrics的classification_report一次性获得precision、recall、f1-score、support（类别出现次数）
    target_names=['{}'.format(i) for i in range(num_classes)]
    report = classification_report(labels[test_idx].cpu().detach().numpy(), test_h.cpu().detach().numpy(), target_names=target_names, digits=4)
    test_gmean = geometric_mean_score(labels[test_mask].cpu(), test_h.cpu())
    print("Test set results:",
          "\nAuc= {:.4f}".format(forest_auc),
          "G-mean= {:.4f}".format(test_gmean),
          "\nReport=\n{}".format(report))

    test_acc = torch.sum(test_h == labels[test_idx]) * 1.0 / len(labels[test_idx])
    test_f1 = f1_score(labels[test_mask].cpu(), test_h.cpu(), average='macro')

    # logging.log(23, f"Test G-Mean: {test_gmean * 100:.2f}% ")
    # logging.log(25, f"---CARE-GNN------------------------------\n dataset: {args.dataset},  train_size:{args.train_size} ")
    # logging.log(25,f"\nReport=:\n {report}")
    # logging.log(23, f"macro f1=:{test_f1:.4f}, acc=: {test_acc:.4f}")

    # For batch testing
    test_recall = recall_score(labels[test_mask].cpu().detach().numpy(), test_h.cpu().detach().numpy(), average='macro')
    logging.log(24, f"AUC:{forest_auc:.4f},F1:{test_f1:.4f},Recall:{test_recall:.4f},G-mean:{test_gmean:.4f}")


if __name__ == '__main__':
    # hyper parameters
    parser = argparse.ArgumentParser(description='GCN-based Anti-Spam Model')
    parser.add_argument("--dataset", type=str, default="Sichuan",
                        help="DGL dataset for this model (Sichuan,BUPT,Citeseer,yelp, or amazon)")
    parser.add_argument("--gpu", type=int, default=0, help="GPU index,0:GPU,1:CPU. Default: -1, using CPU.")
    parser.add_argument("--hid_dim", type=int, default=64, help="Hidden layer dimension")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of layers")
    parser.add_argument("--max_epoch", type=int, default=60, help="The max number of epochs. Default: 30")
    parser.add_argument('--patience', type=int, default=200, help='patience in early stopping')
    parser.add_argument('--lr_GNN', type=float, default=0.02, help='Initial learning rate for GNN.')
    parser.add_argument('--lr_Cost', type=float, default=0.01, help='Initial learning rate for Cost matrix.')
    parser.add_argument('--weight_decay_GNN', type=float, default=1e-3,
                        help='Weight decay for optimizer(all layers) GNN.')
    parser.add_argument('--weight_decay_Cost', type=float, default=1e-3,
                        help='Weight decay for optimizer(all layers) Cost matrix.')
    parser.add_argument("--step_size", type=float, default=0.02, help="RL action step size (lambda 2). Default: 0.02")
    parser.add_argument("--sim_weight", type=float, default=2, help="Similarity loss weight (lambda 1). Default: 2")
    parser.add_argument('--early-stop', action='store_true', default=False, help="indicates whether to use early stop")
    parser.add_argument("--cost_weight", type=float, default=1000, help="Cost matrix weight . Default: 2")
    parser.add_argument('--train_size', type=float, default=0.2, help='train size.')
    parser.add_argument('--val_size', type=float, default=0.2, help='val size.')
    parser.add_argument('--IR', type=float, default=0.1, help='imbalanced ratio.')
    parser.add_argument('--IR_set', type=int, default=0, help='whether to set imbalanced ratio,1 for set ,0 for not.')
    parser.add_argument('--blank', type=int, default=0, help='use during find best hyperparameter.')

    args = parser.parse_args()
    print(args)
    if args.gpu >= 0 and torch.cuda.is_available():
        device = 'cuda:{}'.format(args.gpu)
    else:
        device = 'cpu'
    setup_seed(42)
    main(args)
