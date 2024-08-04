import torch
from torch import nn


class EBD(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super(EBD, self).__init__(*args, **kwargs)
        self.word_ebd = nn.Embedding(28, 24)  # 词嵌入词汇表长度28,向量长度24
        self.pos_ebd = nn.Embedding(12, 24)  # 最大的位置长度是12
        self.pos_t = torch.arange(0, 12).reshape(1, 12)  # 生成0-12的位置编码

    # X.shape应该是(batch_size, length)
    def forward(self, x: torch.Tensor):
        """
        源数据进入到模型中之后,首先经过嵌入层,进行词嵌入,
        然后在其之上附加一个位置编码(直接加和),得到最终结果.
        :param x: 源,输入数据x
        :return: 返回嵌入结果.
        """
        return self.word_ebd(x) + self.pos_ebd(self.pos_t)  # 词嵌入和位置嵌入相加,然后返回计算结果.


def attention_func(Q, K, V):
    """
    该函数用于实现注意力计算公式
    :param Q:
    :param K:
    :param V:
    :return:
    """
    A = Q @ K.transpose(-2, -1)  # 将K的最后两个维度进行交换/转置，然后与Q进行矩阵乘法。
    A = A / Q.shape[-1] ** 0.5  # d_k表示的是特征数量，也就是Q或者K的最后一个维度的长度，使用0.5次幂表示根号计算。
    A = torch.softmax(A, dim=-1)  # 在最后一个维度计算softmax
    O = A @ V  # 计算O
    return O


def transpose_qkv(qkv: torch.Tensor):
    """
    用来拆分Q,K,V
    :param qkv:
    :return:
    """
    qkv = qkv.reshape(qkv.shape[0], qkv.shape[1], 4, 6)  # 拆分四头注意力
    qkv = qkv.transpose(-2, -3)  # 交换维度
    return qkv


def transpose_o(o: torch.Tensor):
    """
    用来还原o的格式
    :param o:
    :return:
    """
    o = o.transpose(-2, -3)
    o = o.reshape(o.shape[0], o.shape[1], -1)
    return o


class AttentionBlock(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super(AttentionBlock, self).__init__(*args, **kwargs)
        self.Wq = nn.Linear(24, 24, bias=False)
        self.Wk = nn.Linear(24, 24, bias=False)
        self.Wv = nn.Linear(24, 24, bias=False)
        self.Wo = nn.Linear(24, 24, bias=False)

    def forward(self, x: torch.Tensor):
        Q, K, V = self.Wq(x), self.Wk(x), self.Wv(x)
        Q, K, V = transpose_qkv(Q), transpose_qkv(K), transpose_qkv(V)
        O = attention_func(Q, K, V)
        O = transpose_o(O)
        O = self.Wo(O)
        return O


class AddNorm(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super(AddNorm, self).__init__(*args, **kwargs)
        self.add_norm = nn.LayerNorm(24)

    def forward(self, x: torch.Tensor, x1):
        """
        加和 && 归一化
        :param x: x表示原始输入（进行词嵌入和位置嵌入后的原始输入）
        :param x1: 表示多头注意力的输出结果
        :return: 加和归一化的输出
        """
        x = x + x1  # 加和
        x = self.add_norm(x)  # 层归一化
        return x


class PosFFN(nn.Module):
    """
    逐位前馈网络
    """

    def __init__(self, *args, **kwargs) -> None:
        super(PosFFN, self).__init__(*args, **kwargs)
        self.linear1 = nn.Linear(24, 48)  # 假设输出48，这里可以修改
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(48, 24)  # 将特征数目变回24
        self.relu2 = nn.ReLU()

    def forward(self, x: torch.Tensor):
        x = self.linear1(x)
        x = self.relu1(x)
        x = self.linear2(x)
        x = self.relu2(x)
        return x


# 下面是测试代码
if __name__ == '__main__':
    aaa = torch.ones((2, 12)).long()
    ebd = EBD()
    aaa = ebd(aaa)

    attention_block = AttentionBlock()
    aaa = attention_block(aaa)

    pass
