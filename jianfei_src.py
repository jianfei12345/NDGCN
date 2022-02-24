import copy
import math
from scipy.spatial.distance import pdist, squareform
import numpy as np
import networkx as nx
import torch
from torch_geometric.data import Data
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv
from sklearn.model_selection import KFold
import torchmetrics
import plotly_express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import pandas as pd
import time
import warnings
import argparse
from torch_geometric.utils import from_networkx, negative_sampling, to_networkx
from node2vec import Node2Vec
from scipy import sparse
from networkx.algorithms.bipartite.matrix import from_biadjacency_matrix
from networkx.algorithms import bipartite
import venn
import os
import matplotlib.pyplot as plt





device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
warnings.filterwarnings('ignore')  #ignore userwarning



#Node2Vec embedding. input: nx graph, output: a ndarray
def embed_Node2Vec(nx_graph, dimension):
    node2vec = Node2Vec(nx_graph, dimensions=dimension, walk_length=30, num_walks=200, workers=4)
    model = node2vec.fit(window=10, min_count=1, batch_words=4)
    embedding = model.wv.vectors
    return embedding








# feature concatenate of input1 and input2 in link prediction, label:adj.
#input1, input2, label: numpy array
def feature_concatenate_edge(input1, input2, label):
    row1 = input1.shape[0]
    row2 = input2.shape[0]
    col1 = input1.shape[1]
    col2 = input2.shape[1]
    row_num = row1 * row2
    col_num = col1 + col2
    conc_feature = np.zeros((row_num, col_num))
    for i in range(row1):
        for j in range(row2):
            index = i * row2 + j
            conc_feature[index, :] = np.concatenate((input1[i, :], input2[j, :]))
    label_conca = label.flatten()
    return conc_feature, label_conca







#不同算法不同模型可能参数不同，别的模型可能需要另外弄参数
#InterAB链路预测参数
def parse_args_interAB():
    parser = argparse.ArgumentParser("Hyperparameters setting")
    parser.add_argument('--k_fold', type=int,default=10) #参数名前面需要加横线
    parser.add_argument("--epoch",type=int,default=200)
    parser.add_argument("--topk",type=int,default=15) #The number of neighbours in constructing the network by similarity matrix based on the top k similar neighbours
    parser.add_argument('--mode',default='client')
    parser.add_argument('--port',default=51015)
    parser.add_argument("--Nlayer3", type=int, default=10)
    parser.add_argument("--Dlayer3", type=int, default=10)
    parser.add_argument("--k_times", type=int, default=100)
    args = parser.parse_args()
    return args




#density calculation of bipartite
def density_calculation(Dataset):
    label = Dataset['label']
    A = sparse.csr_matrix(label)
    g = from_biadjacency_matrix(A)
    density = bipartite.density(g, list(range(max(label.shape))))
    return density




def summary_Datasets(Datasets):
    row_name = list(Datasets.keys())
    col_name = ['No.row', 'No.col', 'No.interactions', 'Density(%)']
    a = pd.DataFrame(index=row_name, columns=col_name)

    for keys in Datasets.keys():
        a.loc[keys, 'No.row'] = len(Datasets[keys]['input1_name'])
        a.loc[keys, 'No.col'] = len(Datasets[keys]['input2_name'])
        a.loc[keys, 'No.interactions'] = Datasets[keys]['label'].sum()
        a.loc[keys, 'Density(%)'] = density_calculation(Datasets[keys]) * 100

    return a




def venn_plot_row(Dataset):
    # 根据输入参数使用不同函数
    def test_function(input_key):
        function_map = {
            '2': venn.venn2,
            '3': venn.venn3,
            '4': venn.venn4,
            '5': venn.venn5,
            '6': venn.venn6,
        }
        return function_map[input_key](labels=labels, names=name)

    label = []
    name = list(Dataset.keys())
    for keys in Dataset.keys():
        label.append(list(Dataset[keys]['input1_name'].str.lower()))
    labels = venn.get_labels(label)

    fig, ax = test_function(input_key=str(len(Dataset)))
    return fig





def venn_plot_col(Dataset):
    # 根据输入参数使用不同函数
    def test_function(input_key):
        function_map = {
            '2': venn.venn2,
            '3': venn.venn3,
            '4': venn.venn4,
            '5': venn.venn5,
            '6': venn.venn6,
        }
        return function_map[input_key](labels=labels, names=name)

    label = []
    name = list(Dataset.keys())
    for keys in Dataset.keys():
        label.append(list(Dataset[keys]['input2_name'].str.lower()))
    labels = venn.get_labels(label)

    fig, ax = test_function(input_key=str(len(Dataset)))
    return fig




def venn_plot_interaction(Dataset):

    def interaction_to_list(subDataset):
        loc_1 = np.where(subDataset['label'] == 1)
        loc_1 = pd.DataFrame(list(loc_1), index=['row', 'col']).T
        row_mapping = {i: subDataset['input1_name'][i] for i in range(len(subDataset['input1_name']))}
        col_mapping = {i: subDataset['input2_name'][i] for i in range(len(subDataset['input2_name']))}
        loc_1.row = loc_1['row'].map(row_mapping)
        loc_1.col = loc_1['col'].map(col_mapping)
        a = loc_1.row.str.cat(loc_1.col).str.lower().tolist()
        return a

    def test_function(input_key):
        function_map = {
            '2': venn.venn2,
            '3': venn.venn3,
            '4': venn.venn4,
            '5': venn.venn5,
            '6': venn.venn6,
        }
        return function_map[input_key](labels=labels, names=name)

    label = []
    name = list(Dataset.keys())
    for keys in Dataset.keys():
        label.append(interaction_to_list(Dataset[keys]))
    labels = venn.get_labels(label)
    fig, ax = test_function(input_key=str(len(Dataset)))
    return fig



def venn_plot(Dataset):
    os.makedirs(os.path.join('visualization_summary', 'Dataset_summary'))
    fig_row = venn_plot_row(Dataset)
    fig_row.savefig(os.path.join('visualization_summary', 'Dataset_summary', 'row.pdf'))
    plt.close()
    fig_col = venn_plot_col(Dataset)
    fig_col.savefig(os.path.join('visualization_summary', 'Dataset_summary', 'col.pdf'))
    plt.close()
    fig_interaction = venn_plot_interaction(Dataset)
    fig_interaction.savefig(os.path.join('visualization_summary', 'Dataset_summary', 'interaction.pdf'))
    plt.close()





#random perturbation by setting p elements to 0
#input: numpy array, output: np array
def random_perturbation_zero_matrix(matrix, p):
    indices = np.random.choice(np.arange(matrix.size), replace=False, size=int(matrix.size * p))
    matrix[np.unravel_index(indices, matrix.shape)] = 0
    return matrix





# random perturbation by add gaussian noise to p elements in the matrix
def random_perturbation_noise_matrix(matrix, p):
    q = 1-p
    indices = np.random.choice(np.arange(matrix.size), replace=False, size=int(matrix.size * q))
    noise = np.random.normal(0, 1, matrix.shape)
    noise[np.unravel_index(indices, matrix.shape)] = 0
    matrix = matrix + noise
    return matrix



#edge perturbation by randomly adding or deleting edges in an undirected and unweighted networkx graph
def add_and_remove_edges(G, p_new_connection, p_remove_connection):
    '''
    for each node,
      add a new connection to random other node, with prob p_new_connection,
      remove a connection, with prob p_remove_connection

    operates on G in-place
    '''
    new_edges = []
    rem_edges = []

    for node in G.nodes():
        # find the other nodes this one is connected to
        connected = [to for (fr, to) in G.edges(node)]
        # and find the remainder of nodes, which are candidates for new edges
        unconnected = [n for n in G.nodes() if not n in connected]

        # probabilistically add a random edge
        if len(unconnected): # only try if new edge is possible
            if random.random() < p_new_connection:
                new = random.choice(unconnected)
                G.add_edge(node, new)
                print("\tnew edge:\t {} -- {}".format(node, new))
                new_edges.append( (node, new) )
                # book-keeping, in case both add and remove done in same cycle
                unconnected.remove(new)
                connected.append(new)

        # probabilistically remove a random edge
        if len(connected): # only try if an edge exists to remove
            if random.random() < p_remove_connection:
                remove = random.choice(connected)
                G.remove_edge(node, remove)
                print( "\tedge removed:\t {} -- {}".format(node, remove))
                rem_edges.append( (node, remove) )
                # book-keeping, in case lists are important later?
                connected.remove(remove)
                unconnected.append(remove)
    return rem_edges, new_edges





# evaluation for  binary classification
def model_evaluation(pred, target):
    evaluation = {}

    #define evaluation metrics
    metric_acc = torchmetrics.Accuracy().to(device) #输入为预测标签或概率矩阵
    metric_roc = torchmetrics.ROC(pos_label=1).to(device)  #设置target正标签,然后以不同阈值分别对pred分类正负标签，得到不同迅销矩阵，计算tpr,fpr
    metric_auroc = torchmetrics.AUROC(pos_label=1).to(device) #二分类:pos_label=1,正标签预测概率。多分类:num_class,概率矩阵
    metric_pr = torchmetrics.BinnedPrecisionRecallCurve(num_classes=1,thresholds=500).to(device) #paired PR values with common threshold(binned)
    metric_AUC = torchmetrics.AUC(reorder=True).to(device)
    metric_F1 = torchmetrics.F1(ignore_index=0).to(device)
    metric_pre = torchmetrics.Precision(ignore_index=0).to(device)
    metric_Recall = torchmetrics.Recall(ignore_index=0).to(device) #precision和recall计算的是正标签，没有设置正标签，所以每类都会计算一下
    metric_ConfusionMatrix = torchmetrics.ConfusionMatrix(num_classes=2).to(device) #列名预测值，行名标签值
    metric_averageprecision = torchmetrics.BinnedAveragePrecision(num_classes=1, thresholds=500).to(device)
    metrics_MCC = torchmetrics.MatthewsCorrcoef(num_classes=2).to(device)

    # model evaluation
    acc = metric_acc(pred, target)
    auroc = metric_auroc(pred[:,1], target)
    precision = metric_pre(pred, target)
    F1 = metric_F1(pred, target)
    Recall = metric_Recall(pred, target)
    roc = metric_roc(pred[:,1], target)
    pr = metric_pr(pred[:,1], target)
    auprc = metric_averageprecision(pred[: ,1], target)
    confusionMatrix = metric_ConfusionMatrix(pred, target)
    MCC = metrics_MCC(pred[:,1], target)

    evaluation['accuracy'] = acc
    evaluation['auroc'] = auroc
    evaluation['precision'] = precision
    evaluation['F1'] = F1
    evaluation['Recall'] = Recall
    evaluation['roc'] = roc
    evaluation['pr'] = pr
    evaluation['auprc'] = auprc
    evaluation['confusionMatrix'] = confusionMatrix
    evaluation['MCC'] = MCC

    return evaluation






# evaluation for multilabel classification
def model_evaluation_multilabel(pred, target):
    evaluation = {}

    #define evaluation metrics
    metric_acc = torchmetrics.Accuracy().to(device) #输入为预测标签或概率矩阵
  #  metric_roc = torchmetrics.ROC(pos_label=1).to(device)  #设置target正标签,然后以不同阈值分别对pred分类正负标签，得到不同迅销矩阵，计算tpr,fpr
  #  metric_auroc = torchmetrics.AUROC(pos_label=1).to(device) #二分类:pos_label=1,正标签预测概率。多分类:num_class,概率矩阵
  #  metric_pr = torchmetrics.BinnedPrecisionRecallCurve(num_classes=1,thresholds=5000).to(device) #paired PR values with common threshold(binned)
    metric_AUC = torchmetrics.AUC(reorder=True).to(device)
    metric_F1 = torchmetrics.F1(ignore_index=0).to(device)
    metric_pre = torchmetrics.Precision(ignore_index=0).to(device)
    metric_Recall = torchmetrics.Recall(ignore_index=0).to(device) #precision和recall计算的是正标签，没有设置正标签，所以每类都会计算一下
  #  metric_ConfusionMatrix = torchmetrics.ConfusionMatrix(num_classes=2).to(device) #列名预测值，行名标签值
  #  metric_averageprecision = torchmetrics.BinnedAveragePrecision(num_classes=1, thresholds=5000).to(device)

    # model evaluation
    acc = metric_acc(pred, target)
  #  auroc = metric_auroc(pred[:,1], target)
    precision = metric_pre(pred, target)
    F1 = metric_F1(pred, target)
    Recall = metric_Recall(pred, target)
   # roc = metric_roc(pred[:,1], target)
   # pr = metric_pr(pred[:,1], target)
  #  auprc = metric_averageprecision(pred[: ,1], target)
  #  confusionMatrix = metric_ConfusionMatrix(pred, target)
    evaluation['accuracy'] = acc
  #  evaluation['auroc'] = auroc
    evaluation['precision'] = precision
    evaluation['F1'] = F1
    evaluation['Recall'] = Recall
 #   evaluation['roc'] = roc
 #   evaluation['pr'] = pr
  #  evaluation['auprc'] = auprc
  #  evaluation['confusionMatrix'] = confusionMatrix

    return evaluation






#summary of kfold result, average accuracy, precision, F1, recall. for ROC and PRC, calculate the entire prediction probability and label to  draw the ROC and PRC curves as well as the AUROC and AUPRC
def k_fold_summary(result_kfold):
    result_summary = pd.DataFrame(columns=['accuracy', 'auroc', 'precision', 'F1', 'recall', 'auprc', 'MCC'])
    acc = np.zeros(len(result_kfold))
    auroc = np.zeros(len(result_kfold))
    pre = np.zeros(len(result_kfold))
    F1 = np.zeros(len(result_kfold))
    recall = np.zeros(len(result_kfold))
    auprc = np.zeros(len(result_kfold))
    MCC = np.zeros(len(result_kfold))
    for i in range(len(result_kfold)):
        acc[i] = result_kfold['Fold' + str(i)]['accuracy'].cpu().numpy()
        auroc[i] = result_kfold['Fold' + str(i)]['auroc'].cpu().numpy()
        pre[i] = result_kfold['Fold' + str(i)]['precision'].cpu().numpy()
        F1[i] = result_kfold['Fold' + str(i)]['F1'].cpu().numpy()
        recall[i] = result_kfold['Fold' + str(i)]['Recall'].cpu().numpy()
        auprc[i] = result_kfold['Fold' + str(i)]['auprc'].cpu().numpy()
        MCC[i] = result_kfold['Fold' + str(i)]['MCC'].cpu().numpy()

    result_summary['accuracy'] = acc
    result_summary['auroc'] = auroc
    result_summary['precision'] = pre
    result_summary['F1'] = F1
    result_summary['recall'] = recall
    result_summary['auprc'] = auprc
    result_summary['MCC'] = MCC

    return result_summary.round(3)





#summary of local loocv result, average accuracy, precision, F1, recall. for ROC and PRC, calculate the entire prediction probability and label to  draw the ROC and PRC curves as well as the AUROC and AUPRC
def local_cv_summary(result_kfold, keys):
    result_summary = pd.DataFrame(columns=['accuracy', 'auroc', 'precision', 'F1', 'recall', 'auprc', 'MCC'], index=keys)
    acc = np.zeros(len(result_kfold))
    auroc = np.zeros(len(result_kfold))
    pre = np.zeros(len(result_kfold))
    F1 = np.zeros(len(result_kfold))
    recall = np.zeros(len(result_kfold))
    auprc = np.zeros(len(result_kfold))
    MCC = np.zeros(len(result_kfold))
    for i in range(len(keys)):
        acc[i] = result_kfold[str(keys[i])]['accuracy'].cpu().numpy()
        auroc[i] = result_kfold[str(keys[i])]['auroc'].cpu().numpy()
        pre[i] = result_kfold[str(keys[i])]['precision'].cpu().numpy()
        F1[i] = result_kfold[str(keys[i])]['F1'].cpu().numpy()
        recall[i] = result_kfold[str(keys[i])]['Recall'].cpu().numpy()
        auprc[i] = result_kfold[str(keys[i])]['auprc'].cpu().numpy()
        MCC[i] = result_kfold[str(keys[i])]['MCC'].cpu().numpy()

    result_summary['accuracy'] = acc
    result_summary['auroc'] = auroc
    result_summary['precision'] = pre
    result_summary['F1'] = F1
    result_summary['recall'] = recall
    result_summary['auprc'] = auprc
    result_summary['MCC'] = MCC

    return result_summary.round(3)






def reset_weights(m):
  '''
    Try resetting model weights to avoid
    weight leakage.
  '''
  for layer in m.children():
   if hasattr(layer, 'reset_parameters'):
   # print(f'Reset trainable parameters of layer = {layer}')
    layer.reset_parameters()




def gaussian_kernal_similarity(rl, n_array, metric_distance):
    xita = rl / (sum(sum(n_array ** 2)) / n_array.shape[0])
    z = math.e ** ((-xita) * (squareform(pdist(n_array, metric=metric_distance)) ** 2))
    return z

#input: np array. construction of torchgeometric  Data by thresholded similarity matrix of top k neighbours
def sim_to_torchgeometric_Data(sim, k):
    kn = np.sort(sim.T)[:, -(k+1)]
    sim[sim < kn] = 0
    sim_net = nx.DiGraph(sim)
    sim_net.remove_edges_from(nx.selfloop_edges(sim_net))  # delete self loop
    sim_net_edgelist = nx.to_pandas_edgelist(sim_net)
    edge_index = torch.from_numpy(sim_net_edgelist.values[:, 0:2].T)
    edge_index = edge_index.long()
    edge_weight = torch.from_numpy(sim_net_edgelist.values[:, 2].T)
    edge_weight = edge_weight.float()
    sim_data = Data(x=torch.eye(sim.shape[0]), edge_index=edge_index, edge_weight=edge_weight)
    return sim_data


#k fold split for adjacent matrix
def k_fold_split(k_fold, data):
    splited_data = {}
    kfold = KFold(n_splits=k_fold, shuffle=True)
    for fold, (train_id, test_id) in enumerate(kfold.split(torch.zeros(data.numel(), 1))):
        fold_data = {}
        train_mask = torch.zeros(data.numel(), 1)
        train_mask[train_id] = 1
        train_mask = (train_mask > 0)
        train_mask = torch.reshape(train_mask, data.shape)
        test_mask = ~train_mask
        fold_data['train_mask'] = train_mask
        fold_data['test_mask'] = test_mask
        splited_data['Fold' + str(fold)] = fold_data
    return splited_data



#data: a one dim  label  or array-like matrix. to splite rows into different train,  test  set
def k_fold_split_onedim(k_fold, data):
    splited_data = {}
    kfold = KFold(n_splits=k_fold, shuffle=True)
    for fold, (train_id, test_id) in enumerate(kfold.split(data)):
        fold_data = {}
        train_mask = torch.zeros_like(data)
        train_mask[train_id] = 1
        train_mask = (train_mask > 0)
        test_mask = ~train_mask
        fold_data['train_mask'] = train_mask
        fold_data['test_mask'] = test_mask
        splited_data['Fold' + str(fold)] = fold_data
    return splited_data




#negative sampling for imbalanced classes,label:adjacent matrix, 1:positive,0:negative. sampling  from negative to  make them 1:1, return a masked matrix covering the balanced classes 1  and 0
def negSampling_balanced(label):
    #label: a two-dimensional tensor with element 0 and 1. 0:negative and 1:positive
    loc_0 = (label == 0).nonzero()[np.random.choice((label == 0).nonzero().shape[0], size=(label == 1).sum().item(), replace=False), :]
    loc_1 = (label == 1).nonzero()
    loc = torch.cat((loc_0, loc_1), dim=0)
    loc = tuple(np.transpose(loc.cpu().numpy()))
    qq = np.zeros_like(label.cpu())
    qq[loc] = 1
    qq_masked = torch.from_numpy(qq)
    qq_masked = (qq_masked > 0)  # 等样本负抽样，然后和正样本合并成一个数据集，masked,对masked数据进行k折划分

    return qq_masked




#negative sampling for imbalanced classes,label:adjacent matrix, 1:positive,0:negative. sampling  from negative to  make them 1:1, return a masked matrix covering the balanced classes 1  and 0
def negSampling_balanced_predefined_size(label, size):
    #label: a two-dimensional tensor with element 0 and 1. 0:negative and 1:positive
    loc_0 = (label == 0).nonzero()[np.random.choice((label == 0).nonzero().shape[0], size=size, replace=False), :]
    loc_1 = (label == 1).nonzero()
    loc = torch.cat((loc_0, loc_1), dim=0)
    loc = tuple(np.transpose(loc.cpu().numpy()))
    qq = np.zeros_like(label.cpu())
    qq[loc] = 1
    qq_masked = torch.from_numpy(qq)
    qq_masked = (qq_masked > 0)  # 等样本负抽样，然后和正样本合并成一个数据集，masked,对masked数据进行k折划分

    return qq_masked



#kfold splition for masked adajacent matrix
def k_fold_split_masked_adj(k_fold, adj_masked):
    #k_fold: number for kfold splition
    #adj_masked: a masked two dimensional bool tensor with True and False, True is the selected balanced classes used for further splition
    splited_data = {}
    kfold = KFold(n_splits=k_fold, shuffle=True)
    for fold, (train_id, test_id) in enumerate(kfold.split(torch.zeros(adj_masked.sum(), 1))):
        fold_data = {}
        loc_masked = (adj_masked == 1).nonzero()
        loc_train = loc_masked[train_id]
        loc_train = tuple(np.transpose(loc_train.numpy()))
        ww = np.zeros_like(adj_masked.cpu())
        ww[loc_train] = 1
        train_mask = torch.from_numpy(ww)
        train_mask = (train_mask>0)

        loc_test = loc_masked[test_id]
        loc_test = tuple(np.transpose(loc_test.cpu().numpy()))
        ww = np.zeros_like(adj_masked.cpu())
        ww[loc_test] = 1
        test_mask = torch.from_numpy(ww)
        test_mask = (test_mask>0)

        fold_data['train_mask'] = train_mask
        fold_data['test_mask'] = test_mask
        splited_data['Fold' + str(fold)] = fold_data
    return splited_data





#visualization
def vis_result(result_all):
    # ROC curve
    fig_roc = px.line(x=result_all['roc'][0].cpu(), y=result_all['roc'][1].cpu())
    fig_roc.add_shape(type='line', line=dict(dash='dash'), x0=0, x1=1, y0=0, y1=1)
    fig_roc.update_layout(title='ROC  curve', xaxis_title='FPR', yaxis_title='TPR')
    fig_roc.add_annotation(x=0.6, y=0.3, text='AUROC = ' + str(round(result_all['auroc'].item(), 3)), showarrow=False,
                           font=dict(color='#FF0000', size=20))
    #fig_roc.show(renderer='browser')
    # PRC curve
    fig_prc = px.line(x=result_all['pr'][1].cpu(), y=result_all['pr'][0].cpu())
    fig_prc.add_shape(type='line', line=dict(dash='dash'), x0=1, x1=0, y0=0, y1=1)
    fig_prc.update_layout(title='PRC curve', xaxis_title='Recall', yaxis_title='Precision')
    fig_prc.add_annotation(x=0.4, y=0.3, text='AUPRC = ' + str(round(result_all['auprc'].item(), 3)), showarrow=False,
                           font=dict(color='#FF0000', size=20))
    #fig_prc.show(renderer='browser')
    # subplots
    fig = make_subplots(rows=1, cols=2, subplot_titles=('ROC curve', 'PRC curve'))
    trace_roc = go.Scatter(x=result_all['roc'][0].cpu(), y=result_all['roc'][1].cpu())
    trace_prc = go.Scatter(x=result_all['pr'][1].cpu(), y=result_all['pr'][0].cpu())
    fig.add_trace(trace_roc, row=1, col=1)
    fig.add_trace(trace_prc, row=1, col=2)
    fig.update_xaxes(title_text='FPR', row=1, col=1)
    fig.update_yaxes(title_text='TPR', row=1, col=1)
    fig.update_xaxes(title_text='Recall', row=1, col=2)
    fig.update_yaxes(title_text='Precision', row=1, col=2)
    fig.add_shape(row=1, col=1, type='line', line=dict(dash='dash'), x0=0, x1=1, y0=0, y1=1)
    fig.add_shape(row=1, col=2, type='line', line=dict(dash='dash'), x0=1, x1=0, y0=0, y1=1)
    fig.add_annotation(row=1, col=1, x=0.6, y=0.3, text='AUROC = ' + str(round(result_all['auroc'].item(), 3)),
                       showarrow=False, font=dict(color='#FF0000', size=20))
    fig.add_annotation(row=1, col=2, x=0.4, y=0.3, text='AUPRC = ' + str(round(result_all['auprc'].item(), 3)),
                       showarrow=False, font=dict(color='#FF0000', size=20))
    fig.update_layout(showlegend=False)
    #fig.show(renderer='browser')
    return fig




#visualization of loss curve for each epoch in the training process
def vis_loss_auroc(loss_record, auroc_record):
    fig = make_subplots(rows=1, cols=2, subplot_titles=('Loss curves', 'AUROC curves'))
    fig.update_xaxes(title_text='epoch', row=1, col=1)
    fig.update_yaxes(title_text='loss', row=1, col=1)
    fig.update_xaxes(title_text='epoch', row=1, col=2)
    fig.update_yaxes(title_text='AUROC', row=1, col=2)
    for i in range(loss_record.shape[1]):
        fig.add_trace(go.Scatter(y=loss_record.iloc[:, i],
                                      x=np.arange(0, loss_record.shape[0]),
                                      name='Fold' + str(i), mode='lines', legendgroup=1), row=1, col=1)
        fig.add_trace(go.Scatter(y=auroc_record.iloc[:, i],
                                 x=np.arange(0, loss_record.shape[0]),
                                 name='Fold' + str(i), mode='lines', legendgroup=2), row=1, col=2)

    #fig.show(renderer='browser')
    return fig




#ROC and PRC curves visualization for kfold cv
def vis_result_kfold(result_kfold):
    # ROC 曲线
    fig_roc = go.Figure()
    fig_roc.add_shape(type='line', line=dict(dash='dash'), x0=0, x1=1, y0=0, y1=1)
    for i in range(len(result_kfold)):
        fig_roc.add_trace(go.Scatter(x=result_kfold['Fold' + str(i)]['roc'][0].cpu().numpy(),
                                     y=result_kfold['Fold' + str(i)]['roc'][1].cpu().numpy(),
                                     name='Fold' + str(i) + ': AUROC =' + str(
                                         round(result_kfold['Fold' + str(i)]['auroc'].item(), 3)), mode='lines'))
    fig_roc.update_layout(legend_x=0.85, legend_y=0, title='ROC curves', xaxis_title='FPR', yaxis_title='TPR')
    #fig_roc.show(renderer='browser')
    # PRC曲线
    fig_prc = go.Figure()
    fig_prc.add_shape(type='line', line=dict(dash='dash'), x0=1, x1=0, y0=0, y1=1)
    for i in range(len(result_kfold)):
        fig_prc.add_trace(go.Scatter(x=result_kfold['Fold' + str(i)]['pr'][1].cpu().numpy(),
                                     y=result_kfold['Fold' + str(i)]['pr'][0].cpu().numpy(),
                                     name='Fold' + str(i) + ': AUPRC =' + str(
                                         round(result_kfold['Fold' + str(i)]['auprc'].item(), 3)), mode='lines'))
    fig_prc.update_layout(legend_x=0.15, legend_y=0, title='PRC curves', xaxis_title='Recall', yaxis_title='Precision')
    #fig_prc.show(renderer='browser')
    # subplot
    fig = make_subplots(rows=1, cols=2, subplot_titles=('ROC curves', 'PRC curves'))
    fig.add_shape(row=1, col=1, type='line', line=dict(dash='dash'), x0=0, x1=1, y0=0, y1=1)
    fig.add_shape(row=1, col=2, type='line', line=dict(dash='dash'), x0=1, x1=0, y0=0, y1=1)
    fig.update_xaxes(title_text='FPR', row=1, col=1)
    fig.update_yaxes(title_text='TPR', row=1, col=1)
    fig.update_xaxes(title_text='Recall', row=1, col=2)
    fig.update_yaxes(title_text='Precision', row=1, col=2)
    for i in range(len(result_kfold)):
        fig.add_trace(go.Scatter(x=result_kfold['Fold' + str(i)]['roc'][0].cpu().numpy(),
                                 y=result_kfold['Fold' + str(i)]['roc'][1].cpu().numpy(),
                                 name='Fold' + str(i) + ': AUROC =' + str(
                                     round(result_kfold['Fold' + str(i)]['auroc'].item(), 3)), mode='lines', legendgroup=1), row=1,
                      col=1)
        fig.add_trace(go.Scatter(x=result_kfold['Fold' + str(i)]['pr'][1].cpu().numpy(),
                                 y=result_kfold['Fold' + str(i)]['pr'][0].cpu().numpy(),
                                 name='Fold' + str(i) + ': AUPRC =' + str(
                                     round(result_kfold['Fold' + str(i)]['auprc'].item(), 3)), mode='lines', legendgroup=2), row=1,
                      col=2)
    #fig.show(renderer='browser')
    return fig






#model construction: link prediction between A and B
class NDGNN(nn.Module):
    def __init__(self, Nlayer1, Nlayer2, Nlayer3, Dlayer1, Dlayer2, Dlayer3):
        super(NDGNN, self).__init__()
        self.Nconv1 = GCNConv(Nlayer1, Nlayer2)
        self.Nconv2 = GCNConv(Nlayer2, Nlayer3)
        self.Dconv1 = GCNConv(Dlayer1, Dlayer2)
        self.Dconv2 = GCNConv(Dlayer2, Dlayer3)


    def encoder_N(self, ncRNA_Data):
        nc_x, nc_edge_index, nc_edge_weight = ncRNA_Data.x, ncRNA_Data.edge_index, ncRNA_Data.edge_weight
        nc_x = self.Nconv1(nc_x, nc_edge_index, nc_edge_weight)
        nc_x = torch.relu(nc_x)
        nc_x = self.Nconv2(nc_x, nc_edge_index, nc_edge_weight)
        return nc_x

    def encoder_D(self, dis_Data):
        dis_x, dis_edge_index, dis_edge_weight = dis_Data.x, dis_Data.edge_index, dis_Data.edge_weight
        dis_x = self.Dconv1(dis_x, dis_edge_index, dis_edge_weight)
        dis_x = torch.relu(dis_x)
        dis_x = self.Dconv2(dis_x, dis_edge_index, dis_edge_weight)
        return dis_x

    def decoder(self, nc_x, dis_x):
        return torch.sigmoid(torch.mm(nc_x, dis_x.t()))

    def forward(self, ncRNA_Data, dis_Data):
        encode_N = self.encoder_N(ncRNA_Data)
        encode_D = self.encoder_D(dis_Data)
        decode = self.decoder(encode_N, encode_D)

        return decode





#cross validation in a splited dataset, return a kfolds result and a total result
def model_cv(splited_data, model_name, label, epoch, input1, input2, label_mask, Nlayer3, Dlayer3):
    result_kfold = {}
    pred_all = torch.zeros_like(label).to(device)
    Nlayer1 = input1.x.size(0)
    Nlayer2 = round(((input1.x.size(0) * Nlayer3)**0.5)/3)
    
    Dlayer1 = input2.x.size(0)
    Dlayer2 = round(((input2.x.size(0) * Dlayer3)**0.5)/3)
    
    metric_auroc = torchmetrics.AUROC(pos_label=1).to(device)
    loss_record = pd.DataFrame(np.zeros((epoch, len(splited_data))),
                               columns=[('Fold' + str(i)) for i in range(len(splited_data))])
    auroc_record = pd.DataFrame(np.zeros((epoch, len(splited_data))),
                                columns=[('Fold' + str(i)) for i in range(len(splited_data))])

    for i in range(len(splited_data)):
        model = model_name(Nlayer1, Nlayer2, Nlayer3, Dlayer1, Dlayer2, Dlayer3).to(device)
        model.apply(reset_weights)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.02)

        # weight for Cross entropy
        r = label[splited_data['Fold' + str(i)]['train_mask']].eq(0).sum() / label[
            splited_data['Fold' + str(i)]['train_mask']].eq(1).sum()
        weight_BCE = label[splited_data['Fold' + str(i)]['train_mask']].float()
        r_1 = torch.ones_like(weight_BCE) * r
        r_0 = torch.ones_like(weight_BCE).float()
        weight_BCE = torch.where(weight_BCE == 1, r_1, r_0)
        criterion = nn.BCELoss(weight=weight_BCE)

        for j in range(epoch):
            outputs = model(input1, input2)
            loss = criterion(outputs[splited_data['Fold' + str(i)]['train_mask']],
                             label[splited_data['Fold' + str(i)]['train_mask']].float())
            pred_epoch = outputs[splited_data['Fold' + str(i)]['test_mask']]
            pred_epoch = torch.cat((1 - torch.unsqueeze(pred_epoch, 1), torch.unsqueeze(pred_epoch, 1)), 1)
            auroc_record.iloc[j, i] = metric_auroc(pred_epoch[:, 1],
                                                   label[splited_data['Fold' + str(i)]['test_mask']]).item()
            loss_record.iloc[j, i] = loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        pred_all = pred_all + torch.mul(outputs, splited_data['Fold' + str(i)]['test_mask'].to(device))

        pred = outputs[splited_data['Fold' + str(i)]['test_mask']]
        pred = torch.cat((1 - torch.unsqueeze(pred, 1), torch.unsqueeze(pred, 1)), 1)

        result_kfold['Fold' + str(i)] = model_evaluation(pred, label[
            splited_data['Fold' + str(i)]['test_mask']])

    pred_all = torch.cat((1 - torch.unsqueeze(pred_all[label_mask], 1), torch.unsqueeze(pred_all[label_mask], 1)), 1)
    result_all = model_evaluation(pred_all, label[label_mask])

    return result_kfold,  result_all, loss_record, auroc_record





# cross validation in a splited dataset, return a total result, without model evaluation for each fold which may cause error due to the absense of positive samples, mainly for  loocv
def model_cv_nokfold_nonegsamp_epoch(splited_data, model_name, label, epoch, input1, input2, label_mask, Nlayer3, Dlayer3):
    pred_all = torch.zeros_like(label).to(device)
    Nlayer1 = input1.x.size(0)
    Nlayer2 = round(((input1.x.size(0) * Nlayer3) ** 0.5) / 3)

    Dlayer1 = input2.x.size(0)
    Dlayer2 = round(((input2.x.size(0) * Dlayer3) ** 0.5) / 3)

    for i in range(len(splited_data)):
        model = model_name(Nlayer1, Nlayer2, Nlayer3, Dlayer1, Dlayer2, Dlayer3).to(device)
        model.apply(reset_weights)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.02)

        # weight for Cross entropy
        r = label[splited_data['Fold' + str(i)]['train_mask']].eq(0).sum() / label[
            splited_data['Fold' + str(i)]['train_mask']].eq(1).sum()
        weight_BCE = label[splited_data['Fold' + str(i)]['train_mask']].float()
        r_1 = torch.ones_like(weight_BCE) * r
        r_0 = torch.ones_like(weight_BCE).float()
        weight_BCE = torch.where(weight_BCE == 1, r_1, r_0)

        criterion = nn.BCELoss(weight=weight_BCE)


        for j in range(epoch):
            outputs = model(input1, input2)
            loss = criterion(outputs[splited_data['Fold' + str(i)]['train_mask']],label[splited_data['Fold' + str(i)]['train_mask']].float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        pred_all = pred_all + torch.mul(outputs, splited_data['Fold' + str(i)]['test_mask'].to(device))

    pred_all = torch.cat((1 - torch.unsqueeze(pred_all[label_mask], 1), torch.unsqueeze(pred_all[label_mask], 1)), 1)
    result_all = model_evaluation(pred_all, label[label_mask])

    return result_all






# cross validation in a splited dataset, return a total result, without model evaluation for each fold which may cause error due to the absense of positive samples, mainly for  loocv
def model_cv_nokfold(splited_data, model_name, label, epoch, input1, input2, label_mask, Nlayer3, Dlayer3):
    pred_all = torch.zeros_like(label).to(device)
    Nlayer1 = input1.x.size(0)
    Nlayer2 = round(((input1.x.size(0) * Nlayer3) ** 0.5) / 3)

    Dlayer1 = input2.x.size(0)
    Dlayer2 = round(((input2.x.size(0) * Dlayer3) ** 0.5) / 3)

    for i in range(len(splited_data)):
        model = model_name(Nlayer1, Nlayer2, Nlayer3, Dlayer1, Dlayer2, Dlayer3).to(device)
        model.apply(reset_weights)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.02)

        """"# weight for Cross entropy计算加权交叉熵权重的时候需要固定数据集，所以加权交叉熵和负抽样不能同时存在
        r = label[splited_data['Fold' + str(i)]['train_mask']].eq(0).sum() / label[
            splited_data['Fold' + str(i)]['train_mask']].eq(1).sum()
        weight_BCE = label[splited_data['Fold' + str(i)]['train_mask']].float()
        r_1 = torch.ones_like(weight_BCE) * r
        r_0 = torch.ones_like(weight_BCE).float()
        weight_BCE = torch.where(weight_BCE == 1, r_1, r_0)
        """

        criterion = nn.BCELoss()


        for j in range(epoch):
            outputs = model(input1, input2)
            #loss = criterion(outputs[splited_data['Fold' + str(i)]['train_mask']],label[splited_data['Fold' + str(i)]['train_mask']].float())
            train_mask_epoch = ((negSampling_balanced_predefined_size(label, splited_data['Fold' + str(i)]['train_mask'].sum().item()).cpu().int() - (label.cpu() & splited_data['Fold' + str(i)]['test_mask'])) == 1)
            loss = criterion(outputs[train_mask_epoch], label[train_mask_epoch].float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        pred_all = pred_all + torch.mul(outputs, splited_data['Fold' + str(i)]['test_mask'].to(device))

    pred_all = torch.cat((1 - torch.unsqueeze(pred_all[label_mask], 1), torch.unsqueeze(pred_all[label_mask], 1)), 1)
    result_all = model_evaluation(pred_all, label[label_mask])

    return result_all


#cross validation in a splited dataset, return a kfolds result and a total result
def model_cv_negSamp_epoch(splited_data, model_name, label, epoch, input1, input2, label_mask, Nlayer3, Dlayer3):
    result_kfold = {}
    pred_all = torch.zeros_like(label).to(device)
    Nlayer1 = input1.x.size(0)
    Nlayer2 = round(((input1.x.size(0) * Nlayer3)**0.5)/3)
    
    Dlayer1 = input2.x.size(0)
    Dlayer2 = round(((input2.x.size(0) * Dlayer3)**0.5)/3)
    
    metric_auroc = torchmetrics.AUROC(pos_label=1).to(device)
    loss_record = pd.DataFrame(np.zeros((epoch, len(splited_data))),
                               columns=[('Fold' + str(i)) for i in range(len(splited_data))])
    auroc_record = pd.DataFrame(np.zeros((epoch, len(splited_data))),
                                columns=[('Fold' + str(i)) for i in range(len(splited_data))])

    for i in range(len(splited_data)):
        model = model_name(Nlayer1, Nlayer2, Nlayer3, Dlayer1, Dlayer2, Dlayer3).to(device)
        model.apply(reset_weights)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.02)

        """"# weight for Cross entropy, for negative sampling per epoch, the length of weights should be changed
        r = label[splited_data['Fold' + str(i)]['train_mask']].eq(0).sum() / label[
            splited_data['Fold' + str(i)]['train_mask']].eq(1).sum()
        weight_BCE = label[splited_data['Fold' + str(i)]['train_mask']].float()
        r_1 = torch.ones_like(weight_BCE) * r
        r_0 = torch.ones_like(weight_BCE).float()
        weight_BCE = torch.where(weight_BCE == 1, r_1, r_0)
        criterion = nn.BCELoss(weight=weight_BCE)
        
        """
        criterion = nn.BCELoss()
        for j in range(epoch):
            outputs = model(input1, input2)
            train_mask_epoch = ((negSampling_balanced_predefined_size(label, splited_data['Fold' + str(i)]['train_mask'].sum().item()).cpu().int() - (label.cpu() & splited_data['Fold' + str(i)]['test_mask'])) ==1)
            loss = criterion(outputs[train_mask_epoch], label[train_mask_epoch].float())
            pred_epoch = outputs[splited_data['Fold' + str(i)]['test_mask']]
            pred_epoch = torch.cat((1 - torch.unsqueeze(pred_epoch, 1), torch.unsqueeze(pred_epoch, 1)), 1)
            auroc_record.iloc[j, i] = metric_auroc(pred_epoch[:, 1],
                                                   label[splited_data['Fold' + str(i)]['test_mask']]).item()
            loss_record.iloc[j, i] = loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        pred_all = pred_all + torch.mul(outputs, splited_data['Fold' + str(i)]['test_mask'].to(device))

        pred = outputs[splited_data['Fold' + str(i)]['test_mask']]
        pred = torch.cat((1 - torch.unsqueeze(pred, 1), torch.unsqueeze(pred, 1)), 1)

        result_kfold['Fold' + str(i)] = model_evaluation(pred, label[
            splited_data['Fold' + str(i)]['test_mask']])

    pred_all = torch.cat((1 - torch.unsqueeze(pred_all[label_mask], 1), torch.unsqueeze(pred_all[label_mask], 1)), 1)
    result_all = model_evaluation(pred_all, label[label_mask])

    result_kfold_kfold = {}
    kfold_summar = k_fold_summary(result_kfold)
    result_kfold_kfold['kfold'] = result_kfold
    result_kfold_kfold['kfold_summary'] = kfold_summar
    result_kfold_kfold['loss_record'] = loss_record
    result_kfold_kfold['auroc_record'] = auroc_record

    result_cv ={}
    result_cv['result_kfold'] = result_kfold_kfold
    result_cv['result_all'] = result_all

    return result_cv







#initialize a  model and  then train, return a trained model
def model_train(model_name, label, epoch, input1, input2, label_mask, Nlayer3, Dlayer3):
        pred_all = torch.zeros_like(label).to(device)
        Nlayer1 = input1.x.size(0)
        Nlayer2 = round(((input1.x.size(0) * Nlayer3) ** 0.5)/3)
        
        Dlayer1 = input2.x.size(0)
        Dlayer2 = round(((input2.x.size(0) * Dlayer3) ** 0.5)/3)
        
        model = model_name(Nlayer1, Nlayer2, Nlayer3, Dlayer1, Dlayer2, Dlayer3).to(device)
        model.apply(reset_weights)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.02)

        # weight for Cross entropy
        r = label[label_mask].eq(0).sum() / label[label_mask].eq(1).sum()
        weight_BCE = label[label_mask].float()
        r_1 = torch.ones_like(weight_BCE) * r
        r_0 = torch.ones_like(weight_BCE).float()
        weight_BCE = torch.where(weight_BCE == 1, r_1, r_0)
        criterion = nn.BCELoss(weight=weight_BCE)

        for j in range(epoch):
            outputs = model(input1, input2)
            loss = criterion(outputs[label_mask],
                             label[label_mask].float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return model





#initialize a  model and  then train, return a trained model
def model_train_negSamp_epoch(model_name, label, epoch, input1, input2, Nlayer3, Dlayer3):
        pred_all = torch.zeros_like(label).to(device)
        Nlayer1 = input1.x.size(0)
        Nlayer2 = round(((input1.x.size(0) * Nlayer3) ** 0.5)/3)

        Dlayer1 = input2.x.size(0)
        Dlayer2 = round(((input2.x.size(0) * Dlayer3) ** 0.5)/3)

        model = model_name(Nlayer1, Nlayer2, Nlayer3, Dlayer1, Dlayer2, Dlayer3).to(device)
        model.apply(reset_weights)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.02)

        # weight for Cross entropy
        """ 
        r = label[label_mask].eq(0).sum() / label[label_mask].eq(1).sum()
        weight_BCE = label[label_mask].float()
        r_1 = torch.ones_like(weight_BCE) * r
        r_0 = torch.ones_like(weight_BCE).float()
        weight_BCE = torch.where(weight_BCE == 1, r_1, r_0)
        criterion = nn.BCELoss(weight=weight_BCE)
        """
        criterion = nn.BCELoss()
        for j in range(epoch):
            outputs = model(input1, input2)
            label_mask_epoch = negSampling_balanced(label)
            loss = criterion(outputs[label_mask_epoch], label[label_mask_epoch].float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return model




def pred_train_ktimes_summary(model_name, label, epoch, Dataset1_circ, Dataset1_dis, Nlayer3, Dlayer3, k_times,Dataset):
    pred_ktimes = np.zeros((label.numel(), k_times))
    for i in range(k_times):
        model_trained = model_train_negSamp_epoch(model_name, label, epoch, Dataset1_circ, Dataset1_dis, Nlayer3,Dlayer3)
        pred_ktimes[:, i] = torch.mul(model_trained(Dataset1_circ, Dataset1_dis),~(label == 1)).cpu().detach().numpy().flatten()

    pred = pd.DataFrame(pred_ktimes.mean(axis=1).reshape(label.size(0), label.size(1)), index=Dataset['input1_name'],columns=Dataset['input2_name'])  # 对所有的0进行预测，包括负抽样和剩下的样本
    return pred


# summary of k times kfold_cv, each time includs negative sampling, kfold spliting, cv.return average and std
def k_times_cv(k, label, k_fold, epoch, Dataset1_circ, Dataset1_dis, model_name, Nlayer3, Dlayer3):
    auroc = np.zeros(k)
    auprc = np.zeros(k)
    for i in range(k):
        label_mask = negSampling_balanced(label)
        splited_data = k_fold_split_masked_adj(k_fold, label_mask)
        # model evaluation by cross validation
        result_kfold, result_all, _, _ = model_cv(splited_data, model_name, label, epoch, Dataset1_circ,
                                                        Dataset1_dis, label_mask, Nlayer3, Dlayer3)
        auroc[i] = result_all['auroc'].item()
        auprc[i] = result_all['auprc'].item()

    return np.mean(auroc), np.std(auroc), np.mean(auprc), np.std(auprc)






# summary of k times kfold_cv, each time includs negative sampling, kfold spliting, cv.return average and std
def k_times_cv_negSamp_epoch(k, label, k_fold, epoch, Dataset1_circ, Dataset1_dis, model_name, Nlayer3, Dlayer3):
    auroc = np.zeros(k)
    auprc = np.zeros(k)
    for i in range(k):
        label_mask = negSampling_balanced(label)
        splited_data = k_fold_split_masked_adj(k_fold, label_mask)
        # model evaluation by cross validation
        result_kfold, result_all, _, _ = model_cv_negSamp_epoch(splited_data, model_name, label, epoch, Dataset1_circ,
                                                        Dataset1_dis, label_mask, Nlayer3, Dlayer3)
        auroc[i] = result_all['auroc'].item()
        auprc[i] = result_all['auprc'].item()

    return np.mean(auroc), np.std(auroc), np.mean(auprc), np.std(auprc)





# one time cv
def one_time_cv(label, k_fold, epoch, Dataset1_circ, Dataset1_dis, model_name, Nlayer3, Dlayer3):
    label_mask = negSampling_balanced(label)
    splited_data = k_fold_split_masked_adj(k_fold, label_mask)
    # model evaluation by cross validation
    result_kfold, result_all, loss_record, auroc_record = model_cv(splited_data, model_name, label, epoch, Dataset1_circ,
                                                    Dataset1_dis, label_mask, Nlayer3, Dlayer3)

    return result_kfold, result_all, loss_record, auroc_record



# one time cv
def one_time_cv_negSamp_epoch(label,label_mask, k_fold, epoch, Dataset1_circ, Dataset1_dis, model_name, Nlayer3, Dlayer3):
    #label_mask = negSampling_balanced(label)
    result_cv = {}
    for i in k_fold:
        splited_data = k_fold_split_masked_adj(i, label_mask)
        # model evaluation by cross validation
        result_cv['Kfold='+str(i)] = model_cv_negSamp_epoch(splited_data, model_name, label, epoch, Dataset1_circ,
                                           Dataset1_dis, label_mask, Nlayer3, Dlayer3)
    return result_cv





#interLP
#input: feature matrix, np array. label: np array.The order of input1 and input2 matchs the row and column of label
#return a dict, result_kfold, result_all, model_trained, pred
def run_model(Dataset, topk, k_fold, epoch, model_name, Nlayer3, Dlayer3):
    time_start = time.time()
    Dataset1_circ = sim_to_torchgeometric_Data(Dataset['input1'], topk).to(device)
    Dataset1_dis = sim_to_torchgeometric_Data(Dataset['input2'], topk).to(device)

    label = torch.from_numpy(Dataset['label']).to(device)

    result_kfold, result_all, loss_record, auroc_record = one_time_cv(label, k_fold, epoch, Dataset1_circ, Dataset1_dis, model_name, Nlayer3, Dlayer3)


    #vis_loss_auroc(loss_record, auroc_record)
    #vis_result(result_all)

    label_mask = negSampling_balanced(label)
    model_trained = model_train(NDGNN, label, epoch, Dataset1_circ, Dataset1_dis, label_mask, Nlayer3, Dlayer3)

    pred = pd.DataFrame(torch.mul(model_trained(Dataset1_circ, Dataset1_dis), ~(label == 1)).cpu().detach().numpy(), index=Dataset['input1_name'], columns=Dataset['input2_name']) #对所有的0进行预测，包括负抽样和剩下的样本

    time_end = time.time()
    print('Time cost:', round(time_end - time_start, 2), 's')

    return dict(result_kfold = result_kfold, result_all = result_all, model_trained = model_trained, pred = pred)




def run_model_negSamp_epoch(Dataset, model_name,parameters_topk, parameters_embedding, topk=15, k_fold=5, epoch=200, Nlayer3=10, Dlayer3=10, pred_k_times=100, localcv_topk=10, localcv_kfold=10, robustness_p = [0.2,0.5,0.8,0.95], casestudy_nrow=5, casestudy_ncol=5, casestudy_topk=5):

    time_start = time.time()
    start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

    Dataset1_circ = sim_to_torchgeometric_Data(Dataset['input1'], topk).to(device)
    Dataset1_dis = sim_to_torchgeometric_Data(Dataset['input2'], topk).to(device)

    label = torch.from_numpy(Dataset['label']).to(device)
    label_mask = negSampling_balanced(label)

    density = density_calculation(Dataset)

    result_cv = one_time_cv_negSamp_epoch(label,label_mask, k_fold, epoch, Dataset1_circ, Dataset1_dis, model_name, Nlayer3, Dlayer3)
    time_end1 = time.time()


    #vis_loss_auroc(loss_record, auroc_record)
    #vis_result(result_all)

    #model_trained = model_train_negSamp_epoch(model_name, label, epoch, Dataset1_circ, Dataset1_dis, Nlayer3, Dlayer3)
    pred = pred_train_ktimes_summary(model_name, label, epoch, Dataset1_circ, Dataset1_dis, Nlayer3, Dlayer3, pred_k_times, Dataset) #k_times: default 100, repeat 100 times and average the results(probability, not label) as pred
    #pred = pd.DataFrame(torch.mul(model_trained(Dataset1_circ, Dataset1_dis), ~(label == 1)).cpu().detach().numpy(), index=Dataset['input1_name'], columns=Dataset['input2_name']) #对所有的0进行预测，包括负抽样和剩下的样本
    time_end2 = time.time()

    #disease, ncRNA-specific数据样本不平衡，要使用加权交叉熵损失函数，这个不需要每次epoch负抽样
    output_localcv = local_cv_row_col_topk(Dataset1_circ, Dataset1_dis, Dataset,model_name, localcv_topk, localcv_kfold)
    time_end3 = time.time()

    #需要负抽样，不需要加权交叉熵。为什么和交叉验证结果相差甚远，因为下面的函数没有每次epoch负抽样，需要每次epoch负抽样。计算加权交叉熵权重的时候需要固定数据集，所以加权交叉熵和负抽样不能同时存在
    output_parameters = parameters_experiment(label, label_mask, Dataset, model_name, parameters_topk, parameters_embedding)
    time_end4 = time.time()

    #模型对缺失值不鲁棒，对高斯噪声很敏感。因为不管缺失值多少，只要不影响矩阵的秩就可以。因为高斯噪声是随机的，所以加高斯噪声后数据不存在规律，不再满足y=F(x)函数,图神经网络好像对攻击很鲁棒，原因：因为构建网络的时候是按照相似性矩阵部分元素（很少一部分）构建的，所以即使大规模置0也可以有很好的效果，2.加高斯噪声的话，相当于满秩，满秩可以拟合任意输出。理论上满秩可以拟合任意输出。但是实际上加噪声会使模型更加复杂，参数难以拟合，影响模型性能。
    output_robustness = robustness_experiment(Dataset, model_name, robustness_p)
    time_end5 = time.time()

    output_case_study = case_study(pred, Dataset['label'], casestudy_nrow, casestudy_ncol, casestudy_topk)
    output_noval_prediction = noval_prediction(pred, 30)

    end_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    time_experiment = pd.DataFrame(columns=['Time consuming(s)','kfold','k_times','topk','embedding','robustness_p','epoch'], index=['CV','Prediction','local_cv','parameters','robustness','total'])
    time_experiment.loc['CV', 'Time consuming(s)'] = round(time_end1 - time_start, 2)
    time_experiment.loc['Prediction', 'Time consuming(s)'] = round(time_end2 - time_end1, 2)
    time_experiment.loc['local_cv', 'Time consuming(s)'] = round(time_end3 - time_end2, 2)
    time_experiment.loc['parameters', 'Time consuming(s)'] = round(time_end4 - time_end3, 2)
    time_experiment.loc['robustness', 'Time consuming(s)'] = round(time_end5 - time_end4, 2)
    time_experiment.loc['total', 'Time consuming(s)'] = round(time_end5 - time_start, 2)
    time_experiment.loc['CV', 'kfold'] = str(k_fold)
    time_experiment.loc['CV', 'epoch'] = str(epoch)
    time_experiment.loc['CV', 'topk'] = str(topk)
    time_experiment.loc['CV', 'embedding'] = str(Nlayer3)
    time_experiment.loc['Prediction', 'k_times'] = str(pred_k_times)
    time_experiment.loc['local_cv', 'kfold'] = str(localcv_kfold)
    time_experiment.loc['parameters', 'topk'] = str(parameters_topk)
    time_experiment.loc['parameters', 'embedding'] = str(parameters_embedding)
    time_experiment.loc['robustness', 'robustness_p'] = str(robustness_p)
    print('model:'+str(model_name),'\n', 'Cross validation:','Kfold:'+str(k_fold), 'topk:'+str(topk), 'epoch:'+str(epoch), 'input1_embedding:'+str(Nlayer3), 'input2_embedding:'+str(Dlayer3),'\n','Prediction:','topk:'+str(topk), 'epoch:'+str(epoch), 'input1_embedding:'+str(Nlayer3), 'input2_embedding:'+str(Dlayer3),'k_times:'+str(pred_k_times), '\n', 'local_cv:','kfold:5','epoch:200','\n','parameters experiment:','topk:'+str(parameters_topk),'embedding:'+str(parameters_embedding),'\n','robustness experiment:','p'+str(robustness_p),'\n', start_time, '\t', str(k_fold)+'fold-CV:', round(time_end1 - time_start, 2), 's', '\t', str(pred_k_times)+'times-pred:', round(time_end2 - time_end1, 2), 's', '\t','local_cv:', round(time_end3 - time_end2, 2), 's', '\t','parameters_experiment:', round(time_end4 - time_end3, 2), 's', '\t','robustness_experiment:', round(time_end5 - time_end4, 2), 's', '\t','total_time:', round(time_end5 - time_start, 2), 's', '\t', end_time)

    return dict( result_cv = result_cv, result_pred = pred, result_localcv = output_localcv, output_parameters = output_parameters, output_robustness = output_robustness, result_case_study=output_case_study, result_noval_prediction = output_noval_prediction, result_time = time_experiment)





def InterAB_Run(Dataset, model,epoch=200,Nlayer3=10, Dlayer3=10, pred_k_times=100, localcv_topk=10,localcv_kfold=20,parameters_topk=[5, 10, 15, 20, 25, 30],parameters_embedding=[2, 6, 10, 15, 20],robustness_p=[0.05,0.1,0.2, 0.5, 0.8, 0.9, 0.95], casestudy_nrow=5, casestudy_ncol=5, casestudy_topk=5):
    #Dataset: 字典，多个数据集
    #model: AB链路预测模型
    #k_fold: list,5,10不能改变，因为后面可视化用的5，10
    #epoch:迭代次数
    #Nlayer3,Dlayer3: 编码维度
    #pred_k_times:对K次预测结果求平均值
    #localcv_topk:对行和列和最大的前k个对象进行local cv
    #localcv_kfold:k折交叉验证for local cv
    #parameters_topk:list，参数的取值范围
    #parameters_embedding: list，参数的取值范围
    #casestudy_nrow:行和最大的前n个元素进行case study
    #casestudy_ncol:列和最大的前N个元素进行 case study
    #casestudy_topk:对每个case取概率最大的前k个对象
    #robustness_p:缺失值或高斯噪声的比例
    output = {}
    for keys in Dataset.keys():
        output[keys] = run_model_negSamp_epoch(Dataset[keys], model_name=model,
                                                           parameters_topk=parameters_topk,
                                                           parameters_embedding=parameters_embedding, k_fold=[5,10],
                                                           epoch=epoch, Nlayer3=Nlayer3, Dlayer3=Dlayer3, pred_k_times=pred_k_times, localcv_topk=localcv_topk,
                                                           localcv_kfold=localcv_kfold,
                                                           robustness_p=robustness_p,casestudy_nrow=casestudy_nrow, casestudy_ncol=casestudy_ncol, casestudy_topk=casestudy_topk)
    vis_all(output)
    venn_plot(Dataset)
    output['summary_Dataset'] = summary_Datasets(Dataset)
    return output




def experiment_cv(Dataset, model_name, topk=15, k_fold=5, epoch=200, Nlayer3=10, Dlayer3=10):

    Dataset1_circ = sim_to_torchgeometric_Data(Dataset['input1'], topk).to(device)
    Dataset1_dis = sim_to_torchgeometric_Data(Dataset['input2'], topk).to(device)

    label = torch.from_numpy(Dataset['label']).to(device)
    label_mask = negSampling_balanced(label)

    splited_data = k_fold_split_masked_adj(k_fold, label_mask)

    result_cv = model_cv_nokfold(splited_data, model_name, label, epoch, Dataset1_circ,Dataset1_dis, label_mask, Nlayer3, Dlayer3)
    return result_cv





#local LOOCV for disease-specific or RNA-specific cross validation
def local_LOOCV_col(Dataset1_circ, Dataset1_dis, col_id,k_fold, model_name, Dataset, epoch, Nlayer3, Dlayer3):

    label_mask = np.zeros_like(Dataset['label'])
    label_mask = torch.from_numpy(label_mask)
    label_mask[:, col_id] = 1
    label_mask = (label_mask > 0)
    splited_data = k_fold_split_masked_adj(k_fold, label_mask)
    #Dataset1_circ = sim_to_torchgeometric_Data(Dataset['input1'], 15).to(device)
    #Dataset1_dis = sim_to_torchgeometric_Data(Dataset['input2'], 15).to(device)
    label = torch.from_numpy(Dataset['label']).to(device)
    result_all = model_cv_nokfold_nonegsamp_epoch(splited_data, model_name, label, epoch, Dataset1_circ, Dataset1_dis, label_mask, Nlayer3, Dlayer3)

    return result_all




#local loocv for topk columns, return a dataframe
def local_cv_col_topk(Dataset1_circ, Dataset1_dis, Dataset, model_name, topk, k_fold):
    # top5最大值索引
    index_topk = np.argsort(Dataset['label'].sum(axis=0))[-topk:]
    # top5最大值
    topk_value = np.sort(Dataset['label'].sum(axis=0))[-topk:]
    keys = list(Dataset['input2_name'][index_topk])

    output_local_cv = {}
    for i in range(topk):
        output_local_cv[str(keys[i])] = local_LOOCV_col(Dataset1_circ, Dataset1_dis, index_topk[i], k_fold, model_name, Dataset, 200, 10, 10)

    output = local_cv_summary(output_local_cv, keys)
    output.insert(0, 'No.interaction', topk_value)
    return output





#local LOOCV for disease-specific or RNA-specific cross validation
def local_LOOCV_row(Dataset1_circ, Dataset1_dis, row_id, k_fold, model_name, Dataset, epoch, Nlayer3, Dlayer3):
    label_mask = np.zeros_like(Dataset['label'])
    label_mask = torch.from_numpy(label_mask)
    label_mask[row_id, :] = 1
    label_mask = (label_mask > 0)
    splited_data = k_fold_split_masked_adj(k_fold, label_mask)
    #Dataset1_circ = sim_to_torchgeometric_Data(Dataset['input1'], 15).to(device)
    #Dataset1_dis = sim_to_torchgeometric_Data(Dataset['input2'], 15).to(device)
    label = torch.from_numpy(Dataset['label']).to(device)
    result_all= model_cv_nokfold_nonegsamp_epoch(splited_data, model_name, label, epoch, Dataset1_circ, Dataset1_dis, label_mask, Nlayer3, Dlayer3)
    return result_all




#local loocv for topk columns, return a dataframe, localcv样本不平衡，需要用加权交叉熵损失函数
def local_cv_row_topk(Dataset1_circ, Dataset1_dis, Dataset, model_name, topk, k_fold):
    # top5最大值索引
    index_topk = np.argsort(Dataset['label'].sum(axis=1))[-topk:]
    # top5最大值
    topk_value = np.sort(Dataset['label'].sum(axis=1))[-topk:]
    keys = list(Dataset['input1_name'][index_topk])

    output_local_cv = {}
    for i in range(topk):
        output_local_cv[str(keys[i])] = local_LOOCV_row(Dataset1_circ, Dataset1_dis, index_topk[i], k_fold, model_name, Dataset, 200, 10, 10)

    output = local_cv_summary(output_local_cv, keys)
    output.insert(0, 'No.interaction', topk_value)
    return output


def local_cv_row_col_topk(Dataset1_circ, Dataset1_dis, Dataset,model_name, topk, kfold):
    output_row = local_cv_row_topk(Dataset1_circ, Dataset1_dis, Dataset,model_name, topk, kfold)
    output_col = local_cv_col_topk(Dataset1_circ, Dataset1_dis, Dataset,model_name, topk, kfold)

    output = {}
    output['row'] = output_row
    output['col'] = output_col
    return output






def parameters_effect(label, label_mask, k_fold, model_name, Dataset, epoch,topk, Nlayer3, Dlayer3):

    splited_data = k_fold_split_masked_adj(k_fold, label_mask)
    Dataset1_circ = sim_to_torchgeometric_Data(Dataset['input1'], topk).to(device)
    Dataset1_dis = sim_to_torchgeometric_Data(Dataset['input2'], topk).to(device)
    result_all= model_cv_nokfold(splited_data, model_name, label, epoch, Dataset1_circ, Dataset1_dis, label_mask, Nlayer3, Dlayer3)
    return result_all



def parameters_effect_topk(label, label_mask, Dataset, model_name,parameters_topk):
    output_parameters_topk = {}
    for i in parameters_topk:
        output_parameters_topk[str(i)] = parameters_effect(label, label_mask, 5, model_name, Dataset, epoch=200, topk=i, Nlayer3=10,Dlayer3=10)

    output = local_cv_summary(output_parameters_topk, parameters_topk)
    return output

def parameters_effect_embedding_size(label, label_mask, Dataset, model_name,parameters_embedding):
    output_parameters_embedding = {}
    for i in parameters_embedding:
        output_parameters_embedding[str(i)] = parameters_effect(label, label_mask, 5,model_name,Dataset,200,15,Nlayer3=i,Dlayer3=i)

    output = local_cv_summary(output_parameters_embedding, parameters_embedding)
    return output

def parameters_experiment(label, label_mask, Dataset, model_name, parameters_topk, parameters_embedding):
    output = {}
    output['topk'] = parameters_effect_topk(label, label_mask, Dataset, model_name, parameters_topk)
    output['embedding'] = parameters_effect_embedding_size(label, label_mask, Dataset, model_name, parameters_embedding)

    return output



def robustness_experiment_zero(Dataset, model_name, p):
    output_row = {}
    output_col = {}
    output_both = {}
    output = {}
    #p = list(np.round(np.arange(0,1,0.1),2))
    #row perturbation
    for i in p:
        Dataset_temp = copy.deepcopy(Dataset)
        Dataset_temp['input1'] = random_perturbation_zero_matrix(Dataset_temp['input1'], i)
        output_row[str(i)] = experiment_cv(Dataset_temp, model_name)
    #col perturbation
    for i in p:
        Dataset_temp = copy.deepcopy(Dataset)
        Dataset_temp['input2'] = random_perturbation_zero_matrix(Dataset_temp['input2'], i)
        output_col[str(i)] = experiment_cv(Dataset_temp, model_name)
    #both row and col perturbation
    for i in p:
        Dataset_temp = copy.deepcopy(Dataset)
        Dataset_temp['input1'] = random_perturbation_zero_matrix(Dataset_temp['input1'], i)
        Dataset_temp['input2'] = random_perturbation_zero_matrix(Dataset_temp['input2'], i)
        output_both[str(i)] = experiment_cv(Dataset_temp, model_name)

    robustness_row = local_cv_summary(output_row, p)
    robustness_col = local_cv_summary(output_col, p)
    robustness_both = local_cv_summary(output_both, p)
    output['row'] = robustness_row
    output['col'] = robustness_col
    output['both'] = robustness_both
    return output




def robustness_experiment_noise(Dataset, model_name, p):
    output_row = {}
    output_col = {}
    output_both = {}
    output = {}
    #p = list(np.round(np.arange(0,1,0.1),2))
    for i in p:
        Dataset_temp = copy.deepcopy(Dataset)
        Dataset_temp['input1'] = random_perturbation_noise_matrix(Dataset_temp['input1'], i)
        output_row[str(i)] = experiment_cv(Dataset_temp, model_name)

    for i in p:
        Dataset_temp = copy.deepcopy(Dataset)
        Dataset_temp['input2'] = random_perturbation_noise_matrix(Dataset_temp['input2'], i)
        output_col[str(i)] = experiment_cv(Dataset_temp, model_name)

    for i in p:
        Dataset_temp = copy.deepcopy(Dataset)
        Dataset_temp['input1'] = random_perturbation_noise_matrix(Dataset_temp['input1'], i)
        Dataset_temp['input2'] = random_perturbation_noise_matrix(Dataset_temp['input2'], i)
        output_both[str(i)] = experiment_cv(Dataset_temp, model_name)

    robustness_row = local_cv_summary(output_row, p)
    robustness_col = local_cv_summary(output_col, p)
    robustness_both = local_cv_summary(output_both, p)
    output['row'] = robustness_row
    output['col'] = robustness_col
    output['both'] = robustness_both
    return output


def robustness_experiment(Dataset, model_name, p):
    output = {}
    output_zero = robustness_experiment_zero(Dataset, model_name, p)
    output_noise = robustness_experiment_noise(Dataset, model_name, p)
    output['zero'] = output_zero
    output['noise'] = output_noise
    return output


def noval_prediction(pred, topk):
    a = pd.DataFrame(columns=pred.columns)
    for i in range(pred.shape[1]):
        a.iloc[:, i] = pred.iloc[:, i].nlargest(pred.shape[0], keep='all').index

    output = a.iloc[0:topk, :]
    return output




def case_study_col(pred,label, ncol, topk):
    q = pd.DataFrame(columns=['col', 'score'])
    index_ncol = np.argsort(label.sum(axis=0))[-ncol:]
    for i in list(index_ncol):
        a = pd.DataFrame(pred.iloc[:,i].nlargest(topk, keep='all'))
        a['col'] = a.columns[0]
        a.columns = ['score', 'col']
        a = a[['col', 'score']]
        q = pd.concat([q,a], axis=0)
    return q



def case_study_row(pred,label, nrow, topk):
    q = pd.DataFrame(columns=['row', 'score'])
    index_nrow = np.argsort(label.sum(axis=1))[-nrow:]
    for i in list(index_nrow):
        a = pd.DataFrame(pred.iloc[i, :].nlargest(topk))
        a['row'] = a.columns[0]
        a.columns = ['score', 'row']
        a = a[['row', 'score']]
        q = pd.concat([q,a], axis=0)
    return q



def case_study(pred, label, nrow, ncol, topk):
    result_col = case_study_col(pred, label, ncol, topk)
    result_row = case_study_row(pred, label, nrow, topk)
    result = dict(row = result_row, col = result_col)
    return result





def vis_all(outputs):
    #创建visualization目录并切换到该目录
    os.makedirs('visualization_summary')
    os.chdir('visualization_summary')
    for key in outputs.keys():
        os.makedirs(os.path.join(key, 'CV'))
        os.makedirs(os.path.join(key, 'CV', '5_fold'))
        fig = vis_loss_auroc(outputs[key]['result_cv']['Kfold=5']['result_kfold']['loss_record'],
                                         outputs[key]['result_cv']['Kfold=5']['result_kfold']['auroc_record'])
        fig.write_image(os.path.join(key, 'CV', '5_fold', 'loss.pdf'), width=700, height=380)
        fig = vis_result_kfold(outputs[key]['result_cv']['Kfold=5']['result_kfold']['kfold'])
        fig.write_image(os.path.join(key, 'CV', '5_fold', 'kfold.pdf'), width=780, height=380)
        fig = vis_result(outputs[key]['result_cv']['Kfold=5']['result_all'])
        fig.write_image(os.path.join(key, 'CV', '5_fold', 'all.pdf'), width=700, height=380)
        os.makedirs(os.path.join(key, 'CV', '10_fold'))
        fig = vis_loss_auroc(outputs[key]['result_cv']['Kfold=10']['result_kfold']['loss_record'],
                                         outputs[key]['result_cv']['Kfold=10']['result_kfold']['auroc_record'])
        fig.write_image(os.path.join(key, 'CV', '10_fold', 'loss.pdf'), width=700, height=380)
        fig = vis_result_kfold(outputs[key]['result_cv']['Kfold=10']['result_kfold']['kfold'])
        fig.write_image(os.path.join(key, 'CV', '10_fold', 'kfold.pdf'), width=780, height=380)
        fig = vis_result(outputs[key]['result_cv']['Kfold=10']['result_all'])
        fig.write_image(os.path.join(key, 'CV', '10_fold', 'all.pdf'), width=700, height=380)

        os.makedirs(os.path.join(key, 'parameters'))
        temp = outputs[key]['output_parameters']['topk']
        temp['topk'] = temp.index
        temp = pd.melt(temp, id_vars=['topk'], value_vars=['auroc', 'auprc', 'MCC', 'accuracy'])
        temp['topk'] = temp['topk'].astype(str)
        fig = px.histogram(temp, x='topk', y='value', color='variable', barmode='group',
                           labels={'topk': 'topk', 'value': 'value'})
        fig.write_image(os.path.join(key, 'parameters', 'topk.pdf'), height=300, width=550)
        temp = outputs[key]['output_parameters']['embedding']
        temp['embedding'] = temp.index
        temp = pd.melt(temp, id_vars=['embedding'], value_vars=['auroc', 'auprc', 'MCC', 'accuracy'])
        temp['embedding'] = temp['embedding'].astype(str)
        fig = px.histogram(temp, x='embedding', y='value', color='variable', barmode='group',
                           labels={'embedding': 'embedding', 'value': 'value'})
        fig.write_image(os.path.join(key, 'parameters', 'embedding.pdf'), height=300, width=550)

        os.makedirs(os.path.join(key, 'robustness'))
        temp = outputs[key]['output_robustness']['zero']['both']
        temp['p'] = temp.index
        temp = pd.melt(temp, id_vars=['p'], value_vars=['auroc', 'auprc', 'MCC', 'accuracy'])
        temp['p'] = temp['p'].astype(str)
        fig = px.histogram(temp, x='p', y='value', color='variable', barmode='group',
                           labels={'p': 'p', 'value': 'value'})
        fig.write_image(os.path.join(key, 'robustness', 'zero.pdf'), height=300, width=550)
        temp = outputs[key]['output_robustness']['noise']['both']
        temp['p'] = temp.index
        temp = pd.melt(temp, id_vars=['p'], value_vars=['auroc', 'auprc', 'MCC', 'accuracy'])
        temp['p'] = temp['p'].astype(str)
        fig = px.histogram(temp, x='p', y='value', color='variable', barmode='group',
                           labels={'p': 'p', 'value': 'value'})
        fig.write_image(os.path.join(key, 'robustness', 'noise.pdf'), height=300, width=550)
    #返回到远工作目录
    os.chdir(os.path.abspath(os.path.join(os.getcwd(), "..")))

















#####for intra network link prediction#####################
#IntraLP链路预测参数
def parse_args_intraLP():
    parser = argparse.ArgumentParser("Hyperparameters setting")
    parser.add_argument('--k_fold', type=int,default=5) #参数名前面需要加横线
    parser.add_argument("--epoch",type=int,default=200)
    parser.add_argument("--k_times", type=int, default=100)
    parser.add_argument('--mode',default='client')
    parser.add_argument('--port',default=51015)
    args = parser.parse_args()
    return args


# model for intra network link prediction
class Intra_LP(nn.Module):
    def __init__(self, intraLP_layer1, intraLP_layer2, intraLP_layer3):
        super(Intra_LP, self).__init__()
        self.GCN1 = GCNConv(intraLP_layer1, intraLP_layer2)
        self.GCN2 = GCNConv(intraLP_layer2, intraLP_layer3)

    def encoder(self, Data):
        Data_x, Data_edge_index = Data.x, Data.edge_index
        Data_x = self.GCN1(Data_x, Data_edge_index)
        Data_x = torch.relu(Data_x)
        Data_x = self.GCN2(Data_x, Data_edge_index)
        return Data_x

    def decoder(self, Data_x):
        output = torch.sigmoid(torch.mm(Data_x, Data_x.t()))
        return output


    def forward(self, Data):
        encode = self.encoder(Data)
        decode = self.decoder(encode)
        return decode



#nx graph to pygData and adjacent matrix
def nx_pygData(G):
    G_Data = from_networkx(G)
    G_Data.x = torch.eye(len(G.nodes))
    adj = torch.from_numpy(np.array(nx.adj_matrix(G).todense()))

    return G_Data, adj


#1:1 negative sampling and kfold splition
def negSamp_kfold_split(G_Data, k_fold):
    neg_edge_index = negative_sampling(G_Data.edge_index, force_undirected=True)

    adj = torch.from_numpy(np.array(nx.adj_matrix(to_networkx(G_Data)).todense()))
    adj_mask = np.zeros_like(adj)
    loc = torch.cat((neg_edge_index, G_Data.edge_index), dim=1)
    loc = tuple(loc.numpy())
    adj_mask[loc] = 1
    adj_mask = torch.from_numpy(adj_mask)
    adj_mask = (adj_mask > 0)
    splited_data = k_fold_split_masked_adj(k_fold, adj_mask)

    return splited_data, adj_mask




def IntraLP_one_cv(splited_data, model_name, epoch, input, label, label_mask):
    layer1 = input.x.size(0)
    layer2 = round(((layer1 * 10) ** 0.5)/3)
    layer3 = 10
    result_kfold = {}
    input = input.to(device)
    label = label.to(device)
    pred_all = torch.zeros_like(label).to(device)
    metric_auroc = torchmetrics.AUROC(pos_label=1).to(device)
    loss_record = pd.DataFrame(np.zeros((epoch, len(splited_data))), columns=[('Fold' + str(i)) for i in range(len(splited_data))])
    auroc_record = pd.DataFrame(np.zeros((epoch, len(splited_data))),columns=[('Fold' + str(i)) for i in range(len(splited_data))])

    for i in range(len(splited_data)):
        model = model_name(layer1, layer2, layer3).to(device)
        model.apply(reset_weights)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


        # weight for Cross entropy
        """
        r = label[splited_data['Fold' + str(i)]['train_mask']].eq(0).sum() / label[
            splited_data['Fold' + str(i)]['train_mask']].eq(1).sum()
        weight_BCE = label[splited_data['Fold' + str(i)]['train_mask']].float()
        r_1 = torch.ones_like(weight_BCE) * r
        r_0 = torch.ones_like(weight_BCE).float()
        weight_BCE = torch.where(weight_BCE == 1, r_1, r_0)
        """
        criterion = nn.BCELoss()


        for j in range(epoch):
            outputs = model(input)
            train_mask_epoch = ((negSampling_balanced_predefined_size(label, splited_data['Fold' + str(i)]['train_mask'].sum().item()).cpu().int() - (label.cpu() & splited_data['Fold' + str(i)]['test_mask'])) == 1)
            loss = criterion(outputs[train_mask_epoch], label[train_mask_epoch].float())
            pred_epoch = outputs[splited_data['Fold' + str(i)]['test_mask']]
            pred_epoch = torch.cat((1 - torch.unsqueeze(pred_epoch, 1), torch.unsqueeze(pred_epoch, 1)), 1)
            auroc_record.iloc[j, i] = metric_auroc(pred_epoch[:,1], label[splited_data['Fold' + str(i)]['test_mask']]).item()
            loss_record.iloc[j, i] = loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        pred_all = pred_all + torch.mul(outputs, splited_data['Fold' + str(i)]['test_mask'].to(device))

        pred = outputs[splited_data['Fold' + str(i)]['test_mask']]
        pred = torch.cat((1 - torch.unsqueeze(pred, 1), torch.unsqueeze(pred, 1)), 1)

        result_kfold['Fold' + str(i)] = model_evaluation(pred, label[splited_data['Fold' + str(i)]['test_mask']])

    pred_all = torch.cat((1 - torch.unsqueeze(pred_all[label_mask], 1), torch.unsqueeze(pred_all[label_mask], 1)),1)
    result_all = model_evaluation(pred_all, label[label_mask])

    return result_kfold, result_all, loss_record, auroc_record




#train a model epoch times
def IntraLP_train(model_name, epoch, input, label):
    layer1 = input.x.size(0)
    layer2 = round(((layer1 * 10) ** 0.5) / 3)
    layer3 = 10
    input = input.to(device)
    label = label.to(device)

    model = model_name(layer1, layer2, layer3).to(device)
    model.apply(reset_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # weight for Cross entropy
    """
    r = label[label_mask].eq(0).sum() / label[label_mask].eq(1).sum()
    weight_BCE = label[label_mask].float()
    r_1 = torch.ones_like(weight_BCE) * r
    r_0 = torch.ones_like(weight_BCE).float()
    weight_BCE = torch.where(weight_BCE == 1, r_1, r_0)
    criterion = nn.BCELoss(weight=weight_BCE)
    """

    criterion = nn.BCELoss()



    for j in range(epoch):
        outputs = model(input)
        label_mask_epoch = negSampling_balanced(label)
        loss = criterion(outputs[label_mask_epoch], label[label_mask_epoch].float())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return model





#label prediction based on k times sub-prediction
def pred_train_ktimes_summary_intraLP(model_name, label, epoch, k_times, G_Data):
    pred_ktimes = np.zeros((label.numel(), k_times))
    for i in range(k_times):
        model_trained = IntraLP_train(model_name, epoch, G_Data, label)
        pred_ktimes[:, i] = torch.mul(model_trained(G_Data.to(device)), ~(label.to(device) == 1)).cpu().detach().numpy().flatten()

    pred = pred_ktimes.mean(axis=1).reshape(label.size(0), label.size(1))
    return pred







def run_IntraIP(G, model_name, epoch, k_fold, k_times):
    time_start = time.time()

    G_Data, label = nx_pygData(G)
    splited_data, label_mask = negSamp_kfold_split(G_Data, k_fold)
    result_kfold, result_all, loss_record, auroc_record = IntraLP_one_cv(splited_data, model_name, epoch, G_Data, label, label_mask)
    #vis_loss_auroc(loss_record, auroc_record)
    #vis_result(result_all)
    model_trained = IntraLP_train(model_name, epoch, G_Data, label)
    pred = pred_train_ktimes_summary_intraLP(model_name, label, epoch, k_times, G_Data)
    #pred = torch.mul(model_trained(G_Data.to(device)), ~(label.to(device) == 1)).cpu().detach().numpy()

    time_end = time.time()
    print('Time cost:', round(time_end - time_start, 2), 's')

    return dict(result_kfold = result_kfold, result_all = result_all, model_trained = model_trained, pred = pred, loss_record = loss_record, auroc_record = auroc_record)













#Node classification
#model construction of node classification
class NC_1(torch.nn.Module):

    def __init__(self, features, hidden, classes):
        super(NC_1, self).__init__()
        self.conv1 = GCNConv(features, hidden)  # shape（输入的节点特征维度 * 中间隐藏层的维度）
        self.conv2 = GCNConv(hidden, classes)  # shaape（中间隐藏层的维度 * 节点类别）

    def forward(self, data):
        # 加载节点特征和邻接关系
        x, edge_index = data.x, data.edge_index
        # 传入卷积层
        x = self.conv1(x, edge_index)
        x = F.relu(x)  # 激活函数
        x = F.dropout(x, training=self.training)  # dropout层，防止过拟合
        x = self.conv2(x, edge_index)  # 第二层卷积层
        # 将经过两层卷积得到的特征输入log_softmax函数得到概率分布
        return F.log_softmax(x, dim=1)






def one_cv_NC(splited_data, data, model_name, epoch):
    NC_layer1 = data.x.shape[1]
    NC_layer3 = data.y.max().item() + 1
    NC_layer2 = round(((NC_layer1 * NC_layer3)**0.5)/3)

    data = data.to(device)
    result_kfold = {}
    pred_all = torch.zeros(data.x.shape[0], data.y.max().item() + 1).to(device)
    metric_acc = torchmetrics.Accuracy().to(device)
    loss_record = pd.DataFrame(np.zeros((epoch, len(splited_data))),
                               columns=[('Fold' + str(i)) for i in range(len(splited_data))])
    acc_record = pd.DataFrame(np.zeros((epoch, len(splited_data))),
                                columns=[('Fold' + str(i)) for i in range(len(splited_data))])

    for i in range(len(splited_data)):
        model = model_name(NC_layer1, NC_layer2, NC_layer3).to(device)
        model.apply(reset_weights)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        for j in range(epoch):
            optimizer.zero_grad()  # 梯度设为零
            out = model(data)  # 模型输出
            loss = F.nll_loss(out[splited_data['Fold' + str(i)]['train_mask']],
                              data.y[splited_data['Fold' + str(i)]['train_mask']])  # 计算损失
            pred_epoch = out[splited_data['Fold' + str(i)]['test_mask']]
            pred_epoch = torch.cat((1 - torch.unsqueeze(pred_epoch, 1), torch.unsqueeze(pred_epoch, 1)), 1)
            acc_record.iloc[j, i] = metric_acc(pred_epoch[:, 1],
                                                   data.y[splited_data['Fold' + str(i)]['test_mask']]).item()
            loss_record.iloc[j, i] = loss.item()

            loss.backward()  # 反向传播计算梯度
            optimizer.step()  # 一步优化

        pred = out[splited_data['Fold' + str(i)]['test_mask']]
        pred_all[splited_data['Fold' + str(i)]['test_mask']] = pred
        result_kfold['Fold' + str(i)] = model_evaluation_multilabel(pred, data.y[splited_data['Fold' + str(i)]['test_mask']])
    result_all = model_evaluation_multilabel(pred_all, data.y)  #计算result_all可以直接用k_fold_aummary函数
    return dict(result_kfold = result_kfold, result_all = result_all, acc_record = acc_record, loss_record = loss_record)


def run_NC(data, k_fold, model_name, epoch):
    time_start = time.time()

    splited_data = k_fold_split_onedim(k_fold, data.y)
    output = one_cv_NC(splited_data, data, model_name, epoch)
    #vis_loss_auroc(output['loss_record'], output['acc_record']) #accuracy

    time_end = time.time()
    print('Time cost:', round(time_end - time_start, 2), 's')

    return output






