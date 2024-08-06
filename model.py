import torch
from torch import nn


class EBD(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super(EBD, self).__init__(*args, **kwargs)
        self.word_ebd = nn.Embedding(29, 24)  # 词嵌入词汇表长度28,向量长度24
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


def attention_func(Q, K, V, i_m):
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
        super(AttentionBlock, self).__init__()
        self.Wq = nn.Linear(24, 24, bias=False)
        self.Wk = nn.Linear(24, 24, bias=False)
        self.Wv = nn.Linear(24, 24, bias=False)
        self.Wo = nn.Linear(24, 24, bias=False)

    def forward(self, x: torch.Tensor, i_m):
        Q, K, V = self.Wq(x), self.Wk(x), self.Wv(x)
        Q, K, V = transpose_qkv(Q), transpose_qkv(K), transpose_qkv(V)
        O = attention_func(Q, K, V, i_m)
        O = transpose_o(O)
        O = self.Wo(O)
        return O


class AddNorm(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super(AddNorm, self).__init__(*args, **kwargs)
        self.add_norm = nn.LayerNorm(24)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
        """
        加和 && 归一化
        :param x: x表示原始输入（进行词嵌入和位置嵌入后的原始输入）
        :param x1: 表示多头注意力的输出结果
        :return: 加和归一化的输出
        """
        x = x + x1  # 加和
        x = self.add_norm(x)  # 层归一化
        x = self.dropout(x)
        return x


class PosFFN(nn.Module):
    """
    逐位前馈网络
    """

    def __init__(self, *args, **kwargs) -> None:
        super(PosFFN, self).__init__(*args, **kwargs)
        self.linear1 = nn.Linear(24, 48, bias=False)  # 假设输出48，这里可以修改
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(48, 24, bias=False)  # 将特征数目变回24
        self.relu2 = nn.ReLU()

    def forward(self, x: torch.Tensor):
        x = self.linear1(x)
        x = self.relu1(x)
        x = self.linear2(x)
        x = self.relu2(x)
        return x


class EncoderBlock(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super(EncoderBlock, self).__init__(*args, **kwargs)
        self.multi_attention = AttentionBlock()
        self.add_norm1 = AddNorm()
        self.FFN = PosFFN()
        self.add_norm2 = AddNorm()

    def forward(self, x: torch.Tensor, i_m):
        x1 = self.multi_attention(x, i_m)  # 计算注意力
        x = self.add_norm1(x, x1)
        x1 = self.FFN(x)
        x = self.add_norm2(x, x1)
        return x


class Encoder(nn.Module):
    """
    打包编码器Encoder
    """

    def __init__(self, *args, **kwargs) -> None:
        super(Encoder, self).__init__(*args, **kwargs)
        self.ebd = EBD(*args, **kwargs)  # 嵌入
        self.encoder_blks = nn.Sequential()  # 创建一个容器
        self.encoder_blks.append(EncoderBlock())  # 假设保存两个编码器
        self.encoder_blks.append(EncoderBlock())

    def forward(self, x: torch.Tensor, i_m):
        """

        :param x:
        :param i_m: 输入掩码
        :return:
        """
        x = self.ebd(x)  # 嵌入
        for encoderBlock in self.encoder_blks:
            x = encoderBlock(x, i_m)
        return x


class CrossAttentionBlock(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super(CrossAttentionBlock, self).__init__(*args, **kwargs)
        self.Wq = nn.Linear(24, 24)
        self.Wk = nn.Linear(24, 24)
        self.Wv = nn.Linear(24, 24)
        self.Wo = nn.Linear(24, 24)

    def forward(self, x, x_en):
        q = self.Wq(x_en)  # q是来自于掩蔽多头注意力块的
        k, v = self.Wk(x), self.Wv(x)
        o = attention_func(q, k, v)
        # o = transpose_o(o)
        return o

class DecoderBlock(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super(DecoderBlock, self).__init__(*args, **kwargs)
        self.multi_attention1 = AttentionBlock()  # 掩蔽多头注意力层，先留着一行
        self.add_norm1 = AddNorm()  # 掩蔽多头注意力之后的加和归一化
        self.cross_attention = CrossAttentionBlock()  # 多头注意力机制
        self.add_norm2 = AddNorm()  # 多头注意力之后的加和归一化
        self.posFFN = PosFFN()  # 逐位前馈网络
        self.add_norm3 = AddNorm()  # 前馈网络之后的加和归一化

    def forward(self, x: torch.Tensor, x_en: torch.Tensor) -> torch.Tensor:
        """
        正向传播
        :param x: 来自编码器的输入
        :param x_en: 来自掩蔽多头注意力的输入
        :return: 输出一个解码器计算结果。
        """
        x1 = self.multi_attention1(x)
        x = self.add_norm1(x, x1)
        x1 = self.cross_attention(x, x_en)
        norm2_output = self.add_norm2(x, x1)
        posFFN_output = self.posFFN(norm2_output)
        norm3_output = self.add_norm3(posFFN_output, norm2_output)
        return norm3_output


class Decoder(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super(Decoder, self).__init__(*args, **kwargs)
        self.ebd = EBD(*args, **kwargs)  # 嵌入层和位置编码
        self.decoderBlocks = nn.Sequential()
        self.decoderBlocks.append(DecoderBlock())  # 添加两个解码层
        self.decoderBlocks.append(DecoderBlock())
        self.dense = nn.Linear(24, 28, bias=False)

    def forward(self, x: torch.Tensor, x_en: torch.Tensor):
        x = self.ebd(x)
        for decoderBlock in self.decoderBlocks:
            x = decoderBlock(x, x_en)
        x = self.dense(x)
        return x


class Transformer(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super(Transformer, self).__init__(*args, **kwargs)
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x_s: torch.Tensor, i_m, x_t: torch.Tensor, o_m):
        """
        Transformer模型向前传播
        :param x_s: 输入到编码器的源字符串
        :param i_m: 输入的掩码
        :param x_t: 输入到解码器的目标字符串
        :param o_m: 输出的掩码
        :return:
        """
        x_en = self.encoder(x_s, i_m)
        x_de = self.decoder(x_t, o_m, x_en, i_m)  # 输入encoder的输出，和目标标签
        return x_de


# 下面是测试代码
if __name__ == '__main__':
    # aaa = torch.ones((2, 12)).long()
    # ebd = EBD()
    # aaa = ebd(aaa)
    #
    # attention_block = AttentionBlock()
    # aaa = attention_block(aaa)

    # 测试Encoder的测试案例
    # encoder = Encoder()
    # input = torch.ones((2, 12)).long()
    # output = encoder(input)

    # 测试Transformer
    input_source = torch.ones((2, 12)).long()
    input_target = torch.ones((2, 1)).long()
    print(type(input_target))
    my_model = Transformer()
    output = my_model(input_source, input_target)
    print(f'output.shape: {output.shape}')
    pass
