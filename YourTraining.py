import mnist
import numpy as np
import pickle
from autograd.utils import PermIterator, buildgraph
from util import setseed

from copy import deepcopy
from typing import List
from autograd.BaseGraph import Graph
from autograd.BaseNode import *

'''
Define Graph
'''
lr = 1e-3   # 学习率
wd1 = 1e-5  # L1正则化
wd2 = 1e-3  # L2正则化
batchsize = 128

def buildGraph(Y):
    """
    建图
    @param Y: n 样本的label
    @return: Graph类的实例, 建好的图
    """
    # TODO: YOUR CODE HERE
    nodes = [StdScaler(mnist.mean_X, mnist.std_X), Linear(mnist.num_feat, mnist.num_class), LogSoftmax(), NLLLoss(Y)]
    nodes = [
        StdScaler(mnist.mean_X, mnist.std_X), 
        Transform(0.3, 0.3, 0.3),
        Linear(mnist.num_feat, 1024), 
        BatchNorm(1),
        Dropout(),
        relu(),
        Linear(1024, 512),
        BatchNorm(1),
        relu(),
        Linear(512, 256),
        BatchNorm(1),
        relu(),
        Linear(256, mnist.num_class),
        LogSoftmax(), 
        NLLLoss(Y),
    ]
    graph=Graph(nodes)
    return graph


'''
Model parameters
'''
import sys
from IPython import embed
setseed(0)

save_path = "model/myModel.npy"

X = mnist.trn_X
Y = mnist.trn_Y

if __name__ == '__main__':
    graph = buildGraph(Y)
    # 训练
    best_train_acc = 0
    dataloader = PermIterator(X.shape[0], batchsize)
    for i in range(1, 60+1):
        hatys = []
        ys = []
        losss = []
        graph.train()
        for perm in dataloader:
            tX = X[perm]
            tY = Y[perm]
            graph[-1].y = tY
            graph.flush()
            pred, loss = graph.forward(tX)[-2:]
            hatys.append(np.argmax(pred, axis=1))
            ys.append(tY)
            graph.backward()
            graph.optimstep(lr, wd1, wd2)
            losss.append(loss)
        loss = np.average(losss)
        acc = np.average(np.concatenate(hatys)==np.concatenate(ys))
        print(f"epoch {i} loss {loss:.3e} acc {acc:.4f}")
        if acc > best_train_acc:
            best_train_acc = acc
            with open(save_path, "wb") as f:
                pickle.dump(graph, f)

    # 测试
    with open(save_path, "rb") as f:
        graph = pickle.load(f)
    graph.eval()
    graph.flush()
    pred = graph.forward(mnist.val_X, removelossnode=1)[-1]
    haty = np.argmax(pred, axis=1)
    print("valid acc", np.average(haty==mnist.val_Y))