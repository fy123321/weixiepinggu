# import torch.nn as nn
# import os
import numpy as np
import pandas as pd
# from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split


# pd.set_option('display.max_rows', 500)
# pd.set_option('display.max_columns', 100)
# pd.set_option('display.width', 1000)
# np.set_printoptions(threshold=np.inf)
# import seaborn as sns
# import matplotlib.pyplot as plt


# number of samples in the data set
N_SAMPLES = 18000
# ratio between training and test sets
TEST_SIZE = 0.111111  # 为了测试模型，从所产生的18000条样本中随机抽取2000条样本作为测试数据 add by fanyu 20230910

# 每一层（包括输入层、隐含层、输出层）的神经元个数,共7层（5层隐含层）  add by fanyu 20230910
NN_ARCHITECTURE = [
    {"input_dim": 4, "output_dim": 12, "activation": "relu"},
    {"input_dim": 12, "output_dim": 11, "activation": "relu"},
    {"input_dim": 11, "output_dim": 10, "activation": "relu"},
    {"input_dim": 10, "output_dim": 9, "activation": "relu"},
    {"input_dim": 9, "output_dim": 25, "activation": "relu"},
    {"input_dim": 25, "output_dim": 8, "activation": "softmax"},
]


def init_layers(nn_architecture, seed=99):
    """
        初始化神经网络。
        输入：自定义的网络结构
        输出：每一层的权重
    """
    # random seed initiation
    np.random.seed(seed)
    # number of layers in our neural network
    # number_of_layers = len(nn_architecture)
    # parameters storage initiation
    params_values = {}

    # iteration over network layers
    for idx, layer in enumerate(nn_architecture):  # 列举；枚举；计算  add by fanyu 20230910
        # note: we number network layers from 1
        layer_idx = idx + 1

        # extracting the number of units in layers
        layer_input_size = layer["input_dim"]
        layer_output_size = layer["output_dim"]

        # initiating the values of the W matrix and vector b for subsequent layers

        ###############  Please finish this part  ################

        # [Hint]:
        # Think about the size of W & b, and replace 'XXX' with 'layer_input_size' or 'layer_output_size'
        # in the following code
        #    params_values['W' + str(layer_idx)] = np.random.randn(XXX, XXX) * 0.1
        #    params_values['b' + str(layer_idx)] = np.random.randn(XXX, 1) * 0.1
        # print(np.random.randn(layer_output_size, layer_input_size))
        # print(np.random.randn(layer_output_size, 1))
        # exit()
        params_values['W' + str(layer_idx)] = np.random.randn(
            layer_output_size, layer_input_size) / np.sqrt(layer_input_size)
        # params_values['b' + str(layer_idx)] = np.random.randn(
        #     layer_output_size, 1) * 0.01
        params_values['b' + str(layer_idx)] = np.zeros(
            layer_output_size).reshape(layer_output_size, -1)
        # params_values['W' + str(layer_idx)] = np.zeros((
        #     layer_output_size, layer_input_size))
        # params_values['b' + str(layer_idx)] = np.zeros((
        #     layer_output_size, 1))
        # 初始输入4个特征
        # w1：12 * 4；b1：12 * 1
        # print(params_values)
        #
        # Have a try: if initialize W & b as 0
        # params_values['W' + str(layer_idx)] = np.zeros((layer_output_size, layer_input_size))
        # params_values['b' + str(layer_idx)] = np.zeros((layer_output_size, 1))

        # print(params_values['W1'])
        ########################  end  ###########################

    return params_values


def sigmoid(Z):
    return 1 / (1 + np.exp(-Z))


def sigmoid_backward(dA, Z, Y):
    sig = sigmoid(Z)
    return dA * sig * (1 - sig)


def relu(Z):  # 整流线性单元(常见激活函数之一)
    return np.maximum(0, Z)


def relu_backward(dA, Z, Y):
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0
    return dZ


def softmax(Z):
    # print(Z)
    # print(Z.shape)
    ex = np.exp(Z)
    # print(ex/ex.sum(axis=0))
    return ex/ex.sum(axis=0)


def softmax_backward(dA, Z, Y):
    """
        求导函数
    """
    out = softmax(Z)
    # dout = np.diag(dZ) - np.outer(dZ, dZ)
    # print(out)
    # print(out.shape)
    # print(Y)
    # print(Y.shape)
    return (out - Y) / Y.shape[1]


def single_layer_forward_propagation(A_prev, W_curr, b_curr, activation="sigmoid"):
    """
        单层神经网络的正向传播。
    """
    # calculation of the input value for the activation function

    ###############  Please finish this part  ################

    # [Hint]:
    # - please implement the computation between W_curr, b_curr, and A_prev
    # - you will need to use the numpy function  ->  np.dot(XXX,XXX)
    # - the code will be like:   Z_curr = XXX
    # print("A_prev：", A_prev.shape)
    # print(np.dot(W_curr, A_prev).shape)
    # exit()
    Z_curr = np.dot(W_curr, A_prev) + b_curr
    # print(activation)
    ########################  end  ###########################

    # selection of activation function
    if activation == "relu":
        activation_func = relu
    elif activation == "sigmoid":
        activation_func = sigmoid
    elif activation == "softmax":
        activation_func = softmax
    else:
        raise Exception('Non-supported activation function')

    # return of calculated activation A and the intermediate Z matrix
    # print(Z_curr[0][0])
    # print(activation_func(Z_curr)[0][0])
    # exit()
    return activation_func(Z_curr), Z_curr


def full_forward_propagation(X, params_values, nn_architecture):
    """
        正向传播
    """
    # creating a temporary memory to store the information needed for a backward step
    # print(X.shape)
    memory = {}
    # X vector is the activation for layer 0 
    A_curr = X  # shape of X: (2,900)

    # iteration over network layers
    for idx, layer in enumerate(nn_architecture):
        # we number network layers from 1
        layer_idx = idx + 1
        # transfer the activation from the previous iteration
        A_prev = A_curr

        # extraction of the activation function for the current layer
        activ_function_curr = layer["activation"]

        # extraction of W for the current layer
        W_curr = params_values["W" + str(layer_idx)]  # 权重W1、W2...  add by fanyu 20230912
        # extraction of b for the current layer
        b_curr = params_values["b" + str(layer_idx)]  # 偏差b1、b2...  add by fanyu 20230912

        # calculation of activation for the current layer

        ###############  Please finish this part  ################

        # [Hint]: invoke the function single_layer_forward_propagation
        A_curr, Z_curr = single_layer_forward_propagation(A_prev, W_curr, b_curr,
                                                          activ_function_curr)  # 将上一层输出（经过激活函数转换后）作为当前层的输入 add by fanyu 20230912
        # print(A_curr)

        ########################  end  ###########################

        # saving calculated values in the memory

        ###############  Please finish this part  ################

        # [Hint]:
        #   memory["A" + str(idx)] = XXX  (A_prev or A_curr ???)
        #   memory["Z" + str(layer_idx)] = XXX

        memory["A" + str(idx)] = A_prev    # 输入层
        memory["Z" + str(layer_idx)] = Z_curr   # 激活之前

        ########################  end  ###########################

    # return of prediction vector and a dictionary containing intermediate values
    return A_curr, memory


def get_cost_value(Y_hat, Y):
    """
        损失函数
    """
    # number of examples (=900)
    # print(Y_hat)
    # print(Y_hat.shape)
    # print(Y_hat.sum(axis=0))
    # print(Y[0])
    # print(Y.shape)
    # print(np.log(Y_hat))
    # print(np.eye(Y.shape[1])[Y[0]])
    # print(np.eye(Y.shape[1])[Y[0]].shape)
    # print(np.eye(Y.shape[1])[Y[0]][:,0])
    # print(np.eye(Y.shape[1])[Y[0]][0][5])
    Y = np.eye(8)[Y[0]]
    # print(Y[0])
    # print(Y)
    # print(Y.shape)
    m = Y_hat.shape[1]  # batchsize   # 获取"Y_hat"的列数 add by fanyu 20230912
    # calculation of the cost according to the formula
    # - Y: each element is either 0 or 1
    # - shape of Y & Y_hat: (1,900)
    # - shape of cost: (1,1)
    # cost = -1 / m * (np.dot(Y, np.log(Y_hat).T) + np.dot(1 - Y, np.log(1 - Y_hat).T))
    # cost = -1 / m * np.dot(Y, np.log(Y_hat).T)
    # print(np.dot(Y, np.log(Y_hat)))
    # exit()
    cost = np.trace(-1 * np.dot(Y, np.log(Y_hat))) / m
    # print(cost)
    return cost
    # return np.squeeze(cost)  # np.squeeze() -  从数组的形状中删除单维条目，即把shape中为1的维度去掉


# an auxiliary function that converts probability into class
# def convert_prob_into_class(probs):  # 用于将概率值转换为类别标签  add by fanyu 20230912
#     probs_ = np.copy(probs)
#     # print(probs_)
#     probs_[probs_ > 0.5] = 1
#     probs_[probs_ <= 0.5] = 0
#     return probs_


def get_accuracy_value(Y_hat, Y):
    """
        计算正确率
    """
    # print(Y_hat.sum(axis=1).sum())
    # Y_hat_ = convert_prob_into_class(Y_hat)  # shape: (1,900)
    # print(Y_hat_ == Y)
    # Y = convert_prob_into_class(Y)  # 范雨加的
    # print(Y)
    # print(Y_hat[:,0:2])
    Y_hat_ = np.argmax(Y_hat, axis=0)
    # print(Y_hat_.shape)
    # print(Y_hat_)
    accuracy = (Y_hat_ == Y).astype(int).mean()
    # accuracy = (Y_hat_ == Y).all(axis=0).mean()  # numpy.all(): 测试沿给定轴的所有数组元素是否都计算为True
    # (Y_hat_ == Y).all(axis=0).shape = (900,)
    # numpy.mean(): 求取均值
    # print(accuracy)
    return accuracy


def single_layer_backward_propagation(dA_curr, W_curr, b_curr, Z_curr, A_prev, Y, activation="sigmoid"):
    """
        单层神经网络的反向传播
    """
    # number of examples
    m = A_prev.shape[1]

    # selection of activation function
    if activation == "relu":
        backward_activation_func = relu_backward
    elif activation == "sigmoid":
        backward_activation_func = sigmoid_backward
    elif activation == "softmax":
        backward_activation_func = softmax_backward
    else:
        raise Exception('Non-supported activation function')

    # step-1: {dA_curr, Z_curr} -> dZ_curr (activation function derivative)
    dZ_curr = backward_activation_func(dA_curr, Z_curr, Y)  # dA_curr 计算损失函数的导数 add by fanyu 20230914

    # print("dZ_curr", dZ_curr.shape)
    # print("dZ_curr", A_prev.T.shape)
    # exit()
    # step-2: {dZ_curr, A_prev} -> dW_curr (derivative of the matrix W)
    #         dZ_curr -> db_curr (derivative of t he vector b)

    ###############  Please finish this part  ################

    # [Hint]: get dW_curr and db_curr from dZ_curr and A_prev
    #   dW_curr = np.dot(XXX，XXX) / m

    # dW_curr = np.dot(dZ_curr, A_prev.T) / m
    dW_curr = np.dot(dZ_curr, A_prev.T)


    ########################  end  ###########################

    # db_curr = np.sum(dZ_curr, axis=1, keepdims=True) / m  # np.sum(): sum of array elements over a given axis
    db_curr = np.sum(dZ_curr, axis=1, keepdims=True)  # np.sum(): sum of array elements over a given axis
    # print(db_curr.shape)
    # or, you can write it as:
    # db_curr = np.dot(dZ_curr, np.ones((dZ_curr.shape[1], 1))) / m  # dZ_curr.shape[1] = 900

    # step-3: {dZ_curr, W_curr} -> dA_prev (derivative of the matrix A_prev)

    ###############  Please finish this part  ################

    # [Hint]: get dA_prev from dZ_curr and W_curr
    #   dA_prev = np.dot(XXX，XXX)

    dA_prev = np.dot(W_curr.T, dZ_curr)

    ########################  end  ###########################

    return dA_prev, dW_curr, db_curr


# let's see how we will access each layer in the backward pass
# for layer_idx_prev, layer in reversed(
#         list(enumerate(NN_ARCHITECTURE))):  # we will use this code line in the function 8-2
#     pass
#     # print(layer_idx_prev)  # from 4 to 0
#     # print(layer)  # from layer L to layer 1 (we number network layers from 1)


def full_backward_propagation(Y_hat, Y, memory, params_values, nn_architecture):
    """
        反向传播
    """
    grads_values = {}
    # print(Y)
    # number of examples (Y.shape=(1,900))
    m = Y.shape[1]
    # K = np.zeros(Y_hat.shape)
    # for i in range(m):
    #     K[int(Y[:, i]) - 1][i] = 1  # 只有维度是1的数组可以被转换成Python标量
    K = np.eye(8)[Y[0]].T
    # print(K[:,0:2])
    # print(K.shape)
    # print(K.shape)

    # a hack ensuring the same shape of the prediction vector and labels vector
    # Y = Y.reshape(Y_hat.shape)
    Y = K
    # initiation of gradient descent algorithm
    # dA_prev = - (np.divide(Y, Y_hat) - np.divide(1 - Y, 1 - Y_hat))
    dA_prev = 1
    # dA_prev = - Y_hat - Y
    # print(dA_prev)
    for layer_idx_prev, layer in reversed(list(enumerate(nn_architecture))):
        # we number network layers from 1
        layer_idx_curr = layer_idx_prev + 1
        # extraction of the activation function for the current layer
        activ_function_curr = layer["activation"]

        dA_curr = dA_prev

        A_prev = memory["A" + str(layer_idx_prev)]
        Z_curr = memory["Z" + str(layer_idx_curr)]

        W_curr = params_values["W" + str(layer_idx_curr)]
        b_curr = params_values["b" + str(layer_idx_curr)]

        dA_prev, dW_curr, db_curr = single_layer_backward_propagation(
            dA_curr, W_curr, b_curr, Z_curr, A_prev, Y, activ_function_curr)

        grads_values["dW" + str(layer_idx_curr)] = dW_curr
        grads_values["db" + str(layer_idx_curr)] = db_curr

    return grads_values


def update(params_values, grads_values, nn_architecture, learning_rate):
    """
        更新参数
    """
    # iteration over network layers
    for idx, layer in enumerate(nn_architecture):
        layer_idx = idx + 1

        ###############  Please finish this part  ################

        # [Hint]: update params_values
        #   params_values["W" + str(layer_idx)] = XXX
        #   params_values["b" + str(layer_idx)] = XXX

        params_values["W" + str(layer_idx)] -= learning_rate * grads_values["dW" + str(layer_idx)]
        params_values["b" + str(layer_idx)] -= learning_rate * grads_values["db" + str(layer_idx)]

        ########################  end  ###########################

    return params_values


def train(X, Y, nn_architecture, epochs, learning_rate, verbose=True, callback=None):
    """
        训练神经网络
    """
    # initiation of neural net parameters
    params_values = init_layers(nn_architecture, 2)  # 初始化参数
    # initiation of lists storing the history of metrics calculated during the learning process
    cost_history = []
    accuracy_history = []

    # performing calculations for subsequent iterations
    for i in range(epochs):
        # step forward
        Y_hat, cache = full_forward_propagation(X, params_values, nn_architecture)  # 正向传播
        # print(Y_hat)
        if i == 1:
            print("X.shape: ", X.shape)  # (2, 900)
            print("Y_hat.shape: ", Y_hat.shape)  # (1, 900)
            print("Y.shape: ", Y.shape)  # (1, 900)

        # calculating metrics and saving them in history
        cost = get_cost_value(Y_hat, Y)  # 计算损失函数
        cost_history.append(cost)
        accuracy = get_accuracy_value(Y_hat, Y)  # 计算准确率
        accuracy_history.append(accuracy)

        # step backward - calculating gradient
        grads_values = full_backward_propagation(Y_hat, Y, cache, params_values, nn_architecture)  # 反向传播
        # updating model state
        params_values = update(params_values, grads_values, nn_architecture, learning_rate)  # 更新参数

        if (i % 50 == 0):
            if (verbose):
                # print("Iteration: {:05} - cost: {:.5f} - accuracy: {:.5f}".format(i, min(cost), accuracy))
                print("Iteration: {:05} - cost: {:.5f} - accuracy: {:.5f}".format(i, cost, accuracy))

            if (callback is not None):
                callback(i, params_values)

    return params_values


b = np.loadtxt('data.txt', delimiter=',')
X = b[:, 0:4]
y = b[:, 4]
y = y.astype(int)
# print(len(set(y.tolist())))

# print("shape of X: ", X.shape)
# X, y = make_moons(n_samples = N_SAMPLES, noise=0.2, random_state=100)
# split the dataset into training set (90%) & test set (10%)
#  - test_size: 如果是浮点数，则应该在0.0和1.0之间，表示要测试集占总数据集的比例；如果是int类型，表示测试集的绝对数量。
#  - random_state: 随机数生成器使用的种子
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=42)

# let's train the neural network
# print(np.transpose(X_train).shape)
# print(np.transpose(y_train.reshape((y_train.shape[0], 1))).shape)

params_values = train(X=np.transpose(X_train), Y=np.transpose(y_train.reshape((y_train.shape[0], 1))),
                      nn_architecture=NN_ARCHITECTURE, epochs=2000, learning_rate=0.02)

# prediction
Y_test_hat, _ = full_forward_propagation(np.transpose(X_test), params_values, NN_ARCHITECTURE)

# accuracy achieved on the test set
acc_test = get_accuracy_value(Y_test_hat, np.transpose(y_test.reshape((y_test.shape[0], 1))))
print("Test set accuracy: {:.2f}".format(acc_test))
