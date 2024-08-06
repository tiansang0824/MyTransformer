import torch
from torch.utils.data import Dataset, DataLoader
from copy import deepcopy
from data_generate import vocab_list, char_list, eos_token, bos_token, pad_token
import numpy as np


def process_data(source, target):
    max_length = 12  # 因为模型的输入长度是12
    # 截断过长的字符串
    if len(source) > max_length:
        source = source[:max_length]
    if len(target) > max_length - 1:
        target = target[:max_length - 1]
    # 将源转换成ID
    source_id = [vocab_list.index(p) for p in source]
    target_id = [vocab_list.index(p) for p in target]
    # 符号补全
    # 注意,要在前面和后面添加列表元素,所以第一个和第三个元素也要加中括号,表示这是一个列表元素.否则就会出现加法类型不匹配的问题.
    target_id = [vocab_list.index(bos_token)] + target_id + [vocab_list.index(eos_token)]
    source_mask = np.array([1] * max_length)  # 源字符串的掩码,1表示不进行掩蔽
    target_mask = np.array([1] * (max_length + 1))  # 目标字符串的掩码
    print()
    if len(source_id) < max_length:
        source_pad_length = (max_length - len(source_id))  # 填充长度:要填充的[pad]的数量
        source_id += [vocab_list.index("[PAD]")] * source_pad_length  # 为源字符串补充填充符号
        source_mask[-source_pad_length:] = 0  # 填充的部分都是无效的0
    if len(target_id) < max_length + 1:
        target_pad_length = max_length + 1 - len(target_id)
        target_id += [vocab_list.index("[PAD]")] * target_pad_length  # 为目标字符串补充填充符号
        target_mask[-target_pad_length:] = 0
    # 返回
    return source_id, source_mask, target_id, target_mask


class MyDataset(Dataset):
    def __init__(self, source_path, target_path) -> None:
        super().__init__()
        self.source_list = []
        self.target_list = []
        with open(source_path) as f_source:
            # readlines()读取文件中的行,并将每一行作为一个元素保存到列表中
            content = f_source.readlines()
            for line in content:
                # i是列表中的每个元素,其最后会跟一个换行符,通过strip()去掉换行符
                self.source_list.append(deepcopy(line.strip()))
        with open(target_path) as f_target:
            content = f_target.readlines()
            for line in content:
                self.target_list.append(deepcopy(line.strip()))

    def __len__(self):
        return len(self.source_list)  # 返回长度等同于源数据的长度

    def __getitem__(self, index):
        source_id, source_mask, target_id, target_mask = process_data(self.source_list[index], self.target_list[index])
        return (torch.tensor(source_id, dtype=torch.long), torch.tensor(source_mask, dtype=torch.long),
                torch.tensor(target_id), torch.tensor(target_mask, dtype=torch.long))


if __name__ == '__main__':
    my_dataset = MyDataset('./source.txt', './target.txt')
    source_id, source_mask, target_id, target_mask = my_dataset[2]
    print(
        f'>> test dataset:\nsource_id: {source_id}, \t\tsource mask: {source_mask}\n'
        f'target_id: {target_id}, \t\ttarget mask: {target_mask}')

    my_loader = DataLoader(my_dataset, 8, True)
    for data in my_loader:
        print(data)
    pass
