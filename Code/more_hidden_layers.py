# Back-Propagation Neural Networks
#encoding=utf-8
import math
import random
#import string
try:
    import cPickle as pickle
except:
    import pickle

random.seed(0)

# calculate a random number where:  a <= rand < b
def rand(a, b):
    return (b-a)*random.random() + a

# our sigmoid function, tanh is a little nicer than the standard 1/(1+e^-x)
def sigmoid(x):
    return math.tanh(x)

# derivative of our sigmoid function, in terms of the output (i.e. y)
def dsigmoid(y):
    return 1.0 - y**2

class Unit:#一个神经元，一个节点
    def __init__(self, length):
        self.weight = [rand(-0.2, 0.2) for i in range(length)]
        self.change = [0.0] * length#change是什么鬼？
        self.threshold = rand(-0.2, 0.2)#这是应该是偏置值。每个神经元都有一个阈值，只有当总和输入超过这个值时，神经元才会做出反应。一般在[-1,1]之间。
        #self.change_threshold = 0.0
    def calc(self, sample):
        self.sample = sample[:]
        partsum = sum([i * j for i, j in zip(self.sample, self.weight)]) - self.threshold#每个节点都有length个输入和权值
        self.output = sigmoid(partsum)#激活函数用的sigmoid函数
        return self.output
    def update(self, diff, rate=0.5, factor=0.1):
        change = [rate * x * diff + factor * c for x, c in zip(self.sample, self.change)]
        self.weight = [w + c for w, c in zip(self.weight, change)]
        self.change = [x * diff for x in self.sample]
        #self.threshold = rateN * factor + rateM * self.change_threshold + self.threshold
        #self.change_threshold = factor
    def get_weight(self):
        return self.weight[:]
    def set_weight(self, weight):
        self.weight = weight[:]


class Layer:#一层网络
    def __init__(self, input_length, output_length):
        self.units = [Unit(input_length) for i in range(output_length)]#该层的节点
        self.output = [0.0] * output_length#该层的输出向量
        self.ilen = input_length#ilen表示输入长度
    def calc(self, sample):#
        self.output = [unit.calc(sample) for unit in self.units]
        return self.output[:]
    def update(self, diffs, rate=0.5, factor=0.1):
        for diff, unit in zip(diffs, self.units):
            unit.update(diff, rate, factor)
    def get_error(self, deltas):
        def _error(deltas, j):
            return sum([delta * unit.weight[j] for delta, unit in zip(deltas, self.units)])
        return [_error(deltas, j) for j  in range(self.ilen)]
    def get_weights(self):
        weights = {}
        for key, unit in enumerate(self.units):
            weights[key] = unit.get_weight()
        return weights
    def set_weights(self, weights):
        for key, unit in enumerate(self.units):
            unit.set_weight(weights[key])



class BPNNet:
    def __init__(self, ni, nh, no):
        # number of input, hidden, and output nodes
        self.ni = ni + 1 # +1 for bias node输入节点多了一个偏置节点
        self.nh = nh
        self.no = no
        self.hlayer1 = Layer(self.ni, self.nh)
        self.hlayer2 = Layer(self.ni, self.nh)
        self.olayer = Layer(self.nh, self.no)

    def calc(self, inputs):
        if len(inputs) != self.ni-1:
            raise ValueError('wrong number of inputs')

        # input activations
        self.ai = inputs[:] + [1.0]#输入向量增加一维，用来乘以偏置值。

        # hidden activations
        self.ah1 = self.hlayer1.calc(self.ai)#将输入层的输入作为隐藏层的输入进行计算
        _ah1=self.ah1
        _ah1=_ah1[:] + [1.0]
        self.ah2 = self.hlayer2.calc(_ah1)  # 将输入层的输入作为隐藏层的输入进行计算
        # output activations
        self.ao = self.olayer.calc(self.ah2)#将隐藏层的输入作为输出层的输入进行计算


        return self.ao[:]


    def update(self, targets, rate, factor):
        if len(targets) != self.no:
            raise ValueError('wrong number of target values')

        # calculate error terms for output #计算输出层的误差
        output_deltas = [dsigmoid(ao) * (target - ao) for target, ao in zip(targets, self.ao)]

        # calculate error terms for hidden #计算隐藏层的误差
        hidden_deltas1 = [dsigmoid(ah) * error for ah, error in zip(self.ah1, self.olayer.get_error(output_deltas))]#最后一层隐藏层
        hidden_deltas2 = [dsigmoid(ah) * error for ah, error in zip(self.ah2, self.olayer.get_error(output_deltas))]#倒数第二层隐藏层
        # update output weights
        self.olayer.update(output_deltas, rate, factor)#更新输出层的权值

        # update input weights #更新隐藏层权值
        self.hlayer1.update(hidden_deltas1, rate, factor)
        self.hlayer2.update(hidden_deltas2, rate, factor)
        # calculate error #计算误差
        return sum([0.5 * (t-o)**2 for t, o in zip(targets, self.ao)])


    def test(self, patterns):
        for p in patterns:
            print(p[0], '->', self.calc(p[0]))

    def train(self, patterns, iterations=1000, N=0.5, M=0.1):#这里的self表示整个网络
        # N: learning rate学习步长
        # M: momentum factor能量因子
        for i in range(iterations):#每次迭代
            error = 0.0#每轮计算误差
            for p in patterns:#所有用于训练的特征向量
                inputs = p[0]#特征向量
                targets = p[1]#类别

                self.calc(inputs)#将特征向量代入激活函数计算
                error = error + self.update(targets, N, M)#累加每个特征向量的误差
            if i % 100 == 0:
                print('error %-.10f' % error)
            # self.save_weights('tmp.weights')
    def save_weights(self, fn):
        weights = {
                "olayer":self.olayer.get_weights(),
                "hlayer1":self.hlayer1.get_weights(),
                "hlayer2":self.hlayer2.get_weights()
                }
        f=open(fn,"w+")
        for ele in weights:
            print(ele,weights[ele])
            f.write(ele)
            f.write(":")
            f.write(str(weights[ele]))
            f.write("\n")
        f.close()
    def load_weights(self, fn):
            with open(fn, "rb") as f:
                weights = pickle.load(f)
                self.olayer.set_weights(weights["olayer"])
                self.hlayer.set_weights(weights["hlayer1"])
                self.hlayer.set_weights(weights["hlayer2"])

def demo():
    # Teach network XOR function
    pat = [
        [[0,0], [0]],
        [[0,1], [1]],
        [[1,0], [1]],
        [[1,1], [0]]
    ]
    # pat = [
    #     [[0,0,0], [0,0]],
    #     [[0,1,0], [1,0]],
    #     [[1,0,0], [1,0]],
    #     [[1,1,0], [0,0]]
    # ]
    print("begin")
    # create a network with two input, two hidden, and one output nodes
    n = BPNNet(2, 2, 1)
    # n = BPNNet(3, 3, 2)
    # train it with some patterns
    n.train(pat)
    # test it
    n.save_weights("../Data/demo.weights")

    n.test(pat)



if __name__ == '__main__':
    demo()
    print("finish")
