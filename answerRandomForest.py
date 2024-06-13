from numpy.random import rand
import mnist
from answerTree import *
import numpy as np

# 超参数
# TODO: You can change the hyperparameters here
num_tree = 35      # 树的数量
ratio_data = 0.5   # 采样的数据比例
ratio_feat = 0.6 # 采样的特征比例
hyperparams = {
    "depth":5, 
    "purity_bound":1e-2,
    "gainfunc": gain
    } # 每颗树的超参数


def buildtrees(X, Y):
    """
    构建随机森林
    @param X: n*d, 每行是一个输入样本。 n: 样本数量， d: 样本的维度
    @param Y: n, 样本的label
    @return: List of DecisionTrees, 随机森林
    """
    # TODO: YOUR CODE HERE
    # 提示：整体流程包括样本扰动、属性扰动和预测输出
    n, d = X.shape
    ret = []
    for i in range(num_tree):
        data_idx = np.random.choice( n, size=int(n * ratio_data), replace=False )
        feat_idx = np.random.choice( d, size=int(d * ratio_feat), replace=False )

        tree = buildTree(X[data_idx], Y[data_idx], feat_idx.tolist(), 
                         hyperparams['depth'], hyperparams['purity_bound'], hyperparams['gainfunc'], '')
        ret.append(tree)
    return ret

def infertrees(trees, X):
    """
    随机森林预测
    @param trees: 随机森林
    @param X: n*d, 每行是一个输入样本。 n: 样本数量， d: 样本的维度
    @return: n, 预测的label
    """
    pred = [inferTree(tree, X)  for tree in trees]
    pred = list(filter(lambda x: not np.isnan(x), pred))
    upred, ucnt = np.unique(pred, return_counts=True)
    return upred[np.argmax(ucnt)]
