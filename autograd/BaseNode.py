from typing import List
import math
import numpy as np
import numpy as np
import scipy.ndimage
import scipy.special
from .Init import * 

def shape(X):
    if isinstance(X, np.ndarray):
        ret = "ndarray"
        if np.any(np.isposinf(X)):
            ret += "_posinf"
        if np.any(np.isneginf(X)):
            ret += "_neginf"
        if np.any(np.isnan(X)):
            ret += "_nan"
        return f" {X.shape} "
    if isinstance(X, int):
        return "int"
    if isinstance(X, float):
        ret = "float"
        if np.any(np.isposinf(X)):
            ret += "_posinf"
        if np.any(np.isneginf(X)):
            ret += "_neginf"
        if np.any(np.isnan(X)):
            ret += "_nan"
        return ret
    else:
        raise NotImplementedError(f"unsupported type {type(X)}")

class Node(object):
    def __init__(self, name, *params):
        self.grad = [] # 节点的梯度，self.grad[i]对应self.params[i]在反向传播时的梯度
        self.cache = [] # 节点保存的临时数据
        self.name = name # 节点的名字
        self.params = list(params) # 用于Linear节点中存储weight和bias参数使用

    def num_params(self):
        return len(self.params)

    def cal(self, X):
        '''
        计算函数值。请在其子类中完成具体实现。
        '''
        pass

    def backcal(self, grad):
        '''
        计算梯度。请在其子类中完成具体实现。
        '''
        pass

    def flush(self):
        '''
        初始化或刷新节点内部数据，包括梯度和缓存
        '''
        self.grad = []
        self.cache = []

    def forward(self, X, debug=False):
        '''
        正向传播。输入X，输出正向传播的计算结果。
        '''
        if debug:
            print(self.name, shape(X))
        ret = self.cal(X)
        if debug:
            print(shape(ret))
        return ret

    def backward(self, grad, debug=False):
        '''
        反向传播。输入grad（该grad为反向传播到该节点的梯度），输出反向传播到下一层的梯度。
        '''
        if debug:
            print(self.name, shape(grad))
        ret = self.backcal(grad)
        if debug:
            print(shape(ret))
        return ret
    
    def eval(self):
        pass

    def train(self):
        pass


class relu(Node):
    # input X: (*)，即可能是任意维度
    # output relu(X): (*)
    def __init__(self):
        super().__init__("relu")

    def cal(self, X):
        self.cache.append(X)
        return np.clip(X, 0, None)

    def backcal(self, grad):
        return np.multiply(grad, self.cache[-1] > 0) 

class sigmoid(Node):
    # input X: (*)，即可能是任意维度
    # output sigmoid(X): (*)
    def __init__(self):
        super().__init__("sigmoid")

    def cal(self, X):
        # TODO: YOUR CODE HERE
        self.cache.append(X)
        return 1/(1 + np.exp(-X))  

    def backcal(self, grad):
        # TODO: YOUR CODE HERE
        sig = 1 / ( 1 + np.exp(self.cache[-1]) )
        return np.multiply(grad, sig * (1 - sig))
    
class tanh(Node):
    # input X: (*)，即可能是任意维度
    # output tanh(X): (*)
    def __init__(self):
        super().__init__("tanh")

    def cal(self, X):
        ret = np.tanh(X)
        self.cache.append(ret)
        return ret


    def backcal(self, grad):
        return np.multiply(grad, np.multiply(1+self.cache[-1], 1-self.cache[-1]))
    

class Linear(Node):
    # input X: (*,d1)
    # param weight: (d1, d2)
    # param bias: (d2)
    # output Linear(X): (*, d2)
    def __init__(self, indim, outdim):
        """
        初始化
        @param indim: 输入维度
        @param outdim: 输出维度
        """
        weight = kaiming_uniform(indim, outdim)
        bias = zeros(outdim)
        super().__init__("linear", weight, bias)

    def cal(self, X):
        # TODO: YOUR CODE HERE
        self.cache.append(X)
        return X @ self.params[0] + self.params[1]

    def backcal(self, grad):
        '''
        需要保存weight和bias的梯度，可以参考Node类和BatchNorm类
        '''
        # TODO: YOUR CODE HERE
        X = self.cache[-1]
        partial_y = grad.reshape(-1, grad.shape[-1])
        xx        = X.reshape(-1, X.shape[-1])
        self.grad.append(xx.T @ partial_y)
        self.grad.append( np.sum(partial_y, axis=0) )

        return grad @ self.params[0].T 


class StdScaler(Node):
    '''
    input shape (*)
    output (*)
    '''
    EPS = 1e-3
    def __init__(self, mean, std):
        super().__init__("StdScaler")
        self.mean = mean
        self.std = std

    def cal(self, X):
        X = X.copy()
        X -= self.mean
        X /= (self.std + self.EPS)
        return X

    def backcal(self, grad):
        return grad/ (self.std + self.EPS)
    


class BatchNorm(Node):
    '''
    input shape (*)
    output (*)
    '''
    EPS = 1e-8
    def __init__(self, indim, momentum: float = 0.9):
        super().__init__("batchnorm", ones((indim)), zeros(indim))
        self.momentum = momentum
        self.mean = None
        self.std = None
        self.updatemean = True
        self.indim = indim

    def cal(self, X):
        if self.updatemean:
            tmean, tstd = np.mean(X, axis=0, keepdims=True), np.std(X, axis=0, keepdims=True)
            if self.mean is None or self.std is None:
                self.mean = tmean
                self.std = tstd
            else:
                self.mean *= self.momentum
                self.mean += (1-self.momentum) * tmean
                self.std *= self.momentum
                self.std += (1-self.momentum) * tstd
        X = X.copy()
        X -= self.mean
        X /= (self.std + self.EPS)
        self.cache.append(X.copy())
        X *= self.params[0]
        X += self.params[1]
        return X

    def backcal(self, grad):
        X = self.cache[-1]
        self.grad.append(np.multiply(X, grad).reshape(-1, self.indim).sum(axis=0))
        self.grad.append(grad.reshape(-1, self.indim).sum(axis=0))
        return (grad*self.params[0])/ (self.std + self.EPS)
    
    def eval(self):
        self.updatemean = False

    def train(self):
        self.updatemean = True


class Dropout(Node):
    '''
    input shape (*)
    output (*)
    '''
    def __init__(self, p: float = 0.1):
        super().__init__("dropout")
        assert 0<=p<=1, "p 是dropout 概率，必须在[0, 1]中"
        self.p = p
        self.dropout = True

    def cal(self, X):
        if self.dropout:
            X = X.copy()
            mask = np.random.rand(*X.shape) < self.p
            np.putmask(X, mask, 0)
            self.cache.append(mask)
        else:
            X = X*(1/(1-self.p))
        return X
    
    def backcal(self, grad):
        if self.dropout:
            grad = grad.copy()
            np.putmask(grad, self.cache[-1], 0)
            return grad
        else:
            return (1/(1-self.p)) * grad
    
    def eval(self):
        self.dropout=False

    def train(self):
        self.dropout=True

class Softmax(Node):
    # input X: (*)
    # output softmax(X): (*), softmax at 'dim'
    def __init__(self, dim=-1):
        super().__init__("softmax")
        self.dim = dim

    def cal(self, X):
        X = X - np.max(X, axis=self.dim, keepdims=True)
        expX = np.exp(X)
        ret = expX / expX.sum(axis=self.dim, keepdims=True)
        self.cache.append(ret)
        return ret

    def backcal(self, grad):
        softmaxX = self.cache[-1]
        grad_p = np.multiply(grad, softmaxX)
        return grad_p - np.multiply(grad_p.sum(axis=self.dim, keepdims=True), softmaxX)

class LogSoftmax(Node):
    # input X: (*)
    # output logsoftmax(X): (*), logsoftmax at 'dim'
    def __init__(self, dim=-1):
        super().__init__("logsoftmax")
        self.dim = dim

    def cal(self, X):
        # TODO: YOUR CODE HERE
        X_max = np.max(X, axis=self.dim, keepdims=True)
        X_mid = X - X_max
        exp_X_mid = np.exp(X_mid)
        ret = np.sum(exp_X_mid, axis=self.dim, keepdims=True)
        self.cache.append(exp_X_mid / ret)
        return X_mid - np.log(ret)  

    def backcal(self, grad):
        # TODO: YOUR CODE HERE
        softmaxX = self.cache[-1]
        grad_p = np.sum(grad, axis=self.dim, keepdims=True)
        return grad - np.multiply(softmaxX, grad_p)




class NLLLoss(Node):
    '''
    negative log-likelihood 损失函数
    '''
    # shape X: (*, d), y: (*)
    # shape value: number 
    # 输入：X: (*) 个预测，每个预测是个d维向量，代表d个类别上分别的log概率。  y：(*) 个整数类别标签
    # 输出：NLL损失
    def __init__(self, y):
        """
        初始化
        @param y: n 样本的label
        """
        super().__init__("NLLLoss")
        self.y = y

    def cal(self, X):
        y = self.y
        self.cache.append(X)
        return - np.sum(
            np.take_along_axis(X, np.expand_dims(y, axis=-1), axis=-1))

    def backcal(self, grad):
        X, y = self.cache[-1], self.y
        ret = np.zeros_like(X)
        np.put_along_axis(ret, np.expand_dims(y, axis=-1), -1, axis=-1)
        return grad * ret



class CrossEntropyLoss(Node):
    '''
    多分类交叉熵损失函数，不同于课上讲的二分类。它与NLLLoss的区别仅在于后者输入log概率，前者输入概率。
    '''
    # shape X: (*, d), y: (*)
    # shape value: number 
    # 输入：X: (*) 个预测，每个预测是个d维向量，代表d个类别上分别的概率。  y：(*) 个整数类别标签
    # 输出：交叉熵损失
    def __init__(self, y):
        """
        初始化
        @param y: n 样本的label
        """
        super().__init__("CELoss")
        self.y = y

    def cal(self, X):
        # TODO: YOUR CODE HERE
        y = self.y
        self.cache.append(X)
        X = np.log(X)
        return - np.sum(
            np.take_along_axis(X, np.expand_dims(y, axis=-1), axis=-1))

    def backcal(self, grad):
        # TODO: YOUR CODE HERE
        X, y = self.cache[-1], self.y
        ret = np.zeros_like(X)
        np.put_along_axis(ret, np.expand_dims(y, axis=-1), -1, axis=-1)
        return grad * ret * 1/X
    
import scipy 
class Transform(Node):
    def __init__(self, p1, p2, p3):
        self.p1 = p1 # add noise
        self.p2 = p2 # shift
        self.p3 = p3
        super().__init__('transform')

    def cal(self, X):
        EPS = 1e-6
        ret = X

        p1 = np.random.uniform()
        p2 = np.random.uniform()
        p3 = np.random.uniform()

        if p1 < self.p1:
            noise = np.random.uniform(0, 1, X.shape)
            ret += noise
            value_min = np.min(ret, axis=0, keepdims=True)
            value_max = np.max(ret, axis=0, keepdims=True)
            ret = (ret - value_min) / (value_max - value_min + EPS)
        if p2 < self.p2:
            r = np.random.uniform(-1, 1)
            N, _ = X.shape
            img = X.reshape((N, 28, 28))
            for i in range(N):
                img[i] = scipy.ndimage.shift(img[i], (int(5 * r), int(5 * r)))
            ret = img.reshape((N, -1))
        if p3 < self.p3:
            N, _ = X.shape
            img = X.reshape((N, 28, 28))
            img = scipy.ndimage.rotate(img, angle=90, axes=(1, 2))
            ret = img.reshape((N, -1))
        return ret

    def backcal(self, grad):
        return grad

'''
Additional Layer
Abandoned : for the low speed
'''
class Conv2d(Node):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        weights, biaes = [], []
        for out_channel in range(out_channels):
            w = []
            for i in range(in_channels):
                w.append(np.expand_dims( kaiming_uniform(kernel_size, kernel_size), 0 ))
            if bias :
                biaes.append(kaiming_uniform(kernel_size, kernel_size))
            weights.append(np.concatenate(w, axis=0))

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.bias = bias
        self.padding = padding

        super().__init__("Conv2d", *weights, *biaes)

    def cal(self, X):
        N, Cin, H, W = X.shape
        self.cache.append(X)

        npad = ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding))
        X = np.pad(X, npad, mode='constant', constant_values=0)

        ret = np.zeros((N, self.out_channels,
                         (H+2*self.padding-self.kernel_size+1)//self.stride, 
                         (W+2*self.padding-self.kernel_size+1)//self.stride))
        
        _, _, ret_H, ret_W = ret.shape

        # ret, X = cconv2d(X, ret, N, self.out_channels, ret_H, ret_W, self.params, self.stride, self.kernel_size)
        
        for n in range(N):
            image = X[n]
            for out_n in range(self.out_channels):
                w = self.params[out_n + 0]
                b = self.params[self.out_channels + out_n]

                for i in range(ret_H):
                    for j in range(ret_W):
                        i0 = i * self.stride
                        j0 = j * self.stride

                        v = X[n, :, i0 : i0+self.kernel_size, j0 : j0+self.kernel_size]

                        res = np.sum(np.multiply(w, v), axis=0) + b
                        res = np.sum(res)

                        ret[n, out_n, i, j] = res
        
        self.cache.append(X)

        return ret
    
    def backcal(self, grad):
        X_ = self.cache[-2]
        N, Cin, H, W = X_.shape

        X = self.cache[-1]
        gx = np.zeros_like(X)
        _, _, hh, ww = gx.shape

        ret_H = (H+2*self.padding-self.kernel_size+1)//self.stride 
        ret_W = (W+2*self.padding-self.kernel_size+1)//self.stride

        grad_w = []
        grad_b = []

        for out_n in range(self.out_channels):
            w = self.params[out_n + 0]
            b = self.params[self.out_channels + out_n]

            gw = np.zeros_like(w)
            gb = np.zeros_like(b)

            for n in range(N):

                for i in range(ret_H):
                    for j in range(ret_W):
                        i0 = i * self.stride
                        j0 = j * self.stride

                        v = X[n, :, i0 : i0+self.kernel_size, j0 : j0+self.kernel_size]

                        gw += v * grad[n, out_n, i, j]
                        gb += 1 * grad[n, out_n, i, j]

                        gx[n, :, i0 : i0+self.kernel_size, j0 : j0+self.kernel_size] += grad[n, out_n, i, j] * w
            
            grad_w.append(gw)
            grad_b.append(gb)
        
        for item in grad_w:
            self.grad.append(item)
        for item in grad_b:
            self.grad.append(item)
        
        return gx[:, :, self.padding : hh-self.padding, self.padding : ww-self.padding]

class flatten(Node):
    def __init__(self):
        super().__init__("flatten")
    
    def cal(self, X):
        self.cache.append(X)
        N, _, _, _ = X.shape
        return X.reshape(N, -1)
    
    def backcal(self, grad):
        X = self.cache[-1]
        return grad.reshape(X.shape)
    
class ToImage(Node):
    def __init__(self, N, C, H, W):
        self.N = N
        self.C = C
        self.H = H
        self.W = W
        super().__init__("toimage")

    def cal(self, X):
        ret = X.reshape((-1, self.C, self.H, self.W))
        self.cache.append(ret)
        return ret

    def backcal(self, grad):
        N = self.cache[-1].shape[0]
        return grad.reshape((N, -1))
    
class MaxPool2d(Node):
    def __init__(self):
        super().__init__("maxpool2d")
    
    def cal(self, X):
        N, C, H, W = X.shape
        
        # Calculate output dimensions
        out_height = H // 2
        out_width = W // 2
        
        # Reshape input_data to (N, C, out_height, pool_height, out_width, pool_width)
        reshaped_X = X.reshape(N, C, out_height, 2, out_width, 2)
        
        # Take the maximum value along the pool_height and pool_width axes
        pooled_output = reshaped_X.max(axis=(3, 5))
        
        self.cache.append(X)
        self.cache.append(reshaped_X)
        return pooled_output

    def backcal(self, grad):
        reshaped_X = self.cache[-1]
        X = self.cache[-2]

        ret = np.zeros_like(X)

        N, C, H, W = X.shape
        out_height = H // 2
        out_width = W // 2

        reshaped_ret = ret.reshape(N, C, out_height, 2, out_height, 2)

        index = np.argmax(reshaped_ret, axis=(3, 5))

        return np.put_along_axis(reshaped_ret, index, )
