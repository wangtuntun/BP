'''
Back-Propagation Neural Networks
对于隐藏层的每个节点（神经元），都会有一个bias（阈值，偏置值）。所以在原来的输入维度上，会添加以为[1.0]
所以输出的隐藏层每个节点的权值会多一个出来，那是给偏置值准备的。
偏置值表示，当该节点的输入总和超过该值时才会被触发，有了偏置值的存在，可以使得函数可以在坐标轴上进行移动，不必经过原点，可以更灵活的进行识别。
偏置值是随机生成的，一般范围在[-1.0,1.0]之间。
'''
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
def sigmoid(x):#每个神经元的激活函数
    return math.tanh(x)#双曲正切函数，计算公式差别很大，但是坐标轴上看着差不多。是一个比较常见的激活函数。（双极S型函数）

# derivative（导数） of our sigmoid function, in terms of the output (i.e. y)
def dsigmoid(y):#神经元激活函数的导数
    return 1.0 - y**2

class Unit:#一个神经元，一个节点
    def __init__(self, length):#神经元的初始值
        self.weight = [rand(-0.2, 0.2) for i in range(length)]#每个神经元的初始权值，[-0.2,0.2]的随机数
        self.change = [0.0] * length
        self.threshold = rand(-0.2, 0.2)#应该是阈值，每个神经元的输入不能超过函数
        #self.change_threshold = 0.0
    def calc(self, sample):#计算神经元的输出
        self.sample = sample[:]
        partsum = sum([i * j for i, j in zip(self.sample, self.weight)]) - self.threshold#   a=wx-t
        self.output = sigmoid(partsum)#激活函数用的sigmoid函数
        return self.output
    def update(self, diff, rate=0.5, factor=0.1):#diff就是delta
        change = [rate * x * diff + factor * c for x, c in zip(self.sample, self.change)]#梯度下降法
        self.weight = [w + c for w, c in zip(self.weight, change)]#每个神经元权值的更新w=w+change
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
        self.ilen = input_length#ilen表示输入长度。这个长度包括了偏置值。
    def calc(self, sample):#
        self.output = [unit.calc(sample) for unit in self.units]
        return self.output[:]
    def update(self, diffs, rate=0.5, factor=0.1):#diffs就是deltas
        for diff, unit in zip(diffs, self.units):
            unit.update(diff, rate, factor)
    def get_error(self, deltas):#  W * X   不是误差吗，怎么成了输入和权值的内积了？
        def _error(deltas, j):#我曹，还他么又定义了一个内部方法。
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
        self.hlayer = Layer(self.ni, self.nh)
        self.olayer = Layer(self.nh, self.no)

    def calc(self, inputs):
        if len(inputs) != self.ni-1:
            raise ValueError('wrong number of inputs')

        # input activations
        self.ai = inputs[:] + [1.0]#输入向量增加一维，用来乘以偏置值。

        # hidden activations
        self.ah = self.hlayer.calc(self.ai)#将输入层的输入作为隐藏层的输入进行计算
        # output activations
        self.ao = self.olayer.calc(self.ah)#将隐藏层的输入作为输出层的输入进行计算


        return self.ao[:]


    def update(self, targets, rate, factor):
        if len(targets) != self.no:
            raise ValueError('wrong number of target values')

        # calculate error terms for output #计算输出层的误差
        output_deltas = [dsigmoid(ao) * (target - ao) for target, ao in zip(targets, self.ao)]#   [(1-ao)**2]  *  [(target-ao)]

        # calculate error terms for hidden #计算隐藏层的误差
        hidden_deltas = [dsigmoid(ah) * error for ah, error in zip(self.ah, self.olayer.get_error(output_deltas))]# [(1-ah)**2]  *  [(w*x)]

        # update output weights #更新输出层的权值
        self.olayer.update(output_deltas, rate, factor)

        # update input weights #更新隐藏层权值
        self.hlayer.update(hidden_deltas, rate, factor)
        # calculate error #计算误差
        return sum([0.5 * (t-o)**2 for t, o in zip(targets, self.ao)])


    def test(self, patterns):
        for p in patterns:
            print(p[0], '->', self.calc(p[0]))

    def train(self, patterns, iterations=1000, N=0.5, M=0.1):#这里的self表示整个网络
        # N: learning rate学习步长，一般在[0,1]范围
        # M: momentum factor能量因子，增加动量项是为了加速算法收敛。一般在[0.1,0.8]范围
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
                "hlayer":self.hlayer.get_weights()
                }
        # with open(fn, "w+") as f:
        #     pickle.dump(weights, f)
        # print(weights)
        # f=open(fn,"w+")
        for ele in weights:
            print(ele,weights[ele])
    def load_weights(self, fn):
            with open(fn, "rb") as f:
                weights = pickle.load(f)
                self.olayer.set_weights(weights["olayer"])
                self.hlayer.set_weights(weights["hlayer"])

def demo():
    # Teach network XOR function
    # pat = [
    #     [[0,0,0], [0,0]],
    #     [[0,1,0], [1,0]],
    #     [[1,0,0], [1,0]],
    #     [[1,1,0], [0,0]]
    # ]
    # pat = [#未对数据进行归一化操作，效果很差。
    #     [[0,0], [0]],
    #     [[0,10], [1]],
    #     [[10,0], [1]],
    #     [[10,10], [0]]
    # ]
    # pat = [
    #     [[0,0], [0]],
    #     [[0,1], [1]],
    #     [[1,0], [1]],
    #     [[1,1], [0]]
    # ]
    pat = [
        [[-1.0, -1.0], [-1.0]],
        [[-1.0, 1.0], [-0.8]],
        [[1.0, -1.0], [-0.8]],
        [[1.0, 1.0], [-1.0]]
    ]
    print("begin")
    # create a network with two input, two hidden, and one output nodes
    # n = BPNNet(2, 2, 1)
    n = BPNNet(2, 2, 1)
    # train it with some patterns
    n.train(pat)
    # test it
    n.save_weights("../Data/demo.weights")

    n.test(pat)



if __name__ == '__main__':
    demo()
    print("finish")
