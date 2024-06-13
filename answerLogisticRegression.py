import numpy as np

# 超参数
# TODO: You can change the hyperparameters here
lr = 10  # 学习率
wd = 1e-2  # l2正则化项系数
EPS = 1e-6


def predict(X, weight, bias):
    """
    使用输入的weight和bias，预测样本X是否为数字0。
    @param X: (n, d) 每行是一个输入样本。n: 样本数量, d: 样本的维度
    @param weight: (d,)
    @param bias: (1,)
    @return: (n,) 线性模型的输出，即wx+b
    """
    # TODO: YOUR CODE HERE
    wx = X @ weight
    return ( wx + bias )

def sigmoid(x):
    return 1 / (np.exp(-x) + 1)


def step(X, weight, bias, Y):
    """
    单步训练, 进行一次forward、backward和参数更新
    @param X: (n, d) 每行是一个训练样本。 n: 样本数量， d: 样本的维度
    @param weight: (d,)
    @param bias: (1,)
    @param Y: (n,) 样本的label, 1表示为数字0, -1表示不为数字0
    @return:
        haty: (n,) 模型的输出, 为正表示数字为0, 为负表示数字不为0
        loss: (1,) 由交叉熵损失函数计算得到
        weight: (d,) 更新后的weight参数
        bias: (1,) 更新后的bias参数
    """
    # TODO: YOUR CODE HERE
    n = X.shape[0]
    # forward
    haty = predict(X, weight, bias)
    sig  = sigmoid(Y * haty)
    loss = np.sum( np.log( sig + EPS ) ) / n
    # backward
    tmp = Y * (sig - 1)
    db = np.mean(tmp)
    dw = X.T @ tmp / n

    weight -= lr * (dw + 2 * wd * weight)
    bias -= lr * db

    return haty, loss, weight, bias
