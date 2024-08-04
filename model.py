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


# 下面是测试代码
if __name__ == '__main__':
    aaa = torch.ones((2, 12)).long()
    ebd = EBD()
    aaa = ebd(aaa)
    pass
