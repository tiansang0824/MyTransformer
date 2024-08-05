import random

vocab_list = ['[EOS]', '[BOS]', '[PAD]', 'a', 'b', 'c', 'd', 'e', 'f', 'g',
              'h', 'i', 'j', 'k', 'l', 'm', 'n',
              'o', 'p', 'q', 'r', 's', 't',
              'u', 'v', 'w', 'x', 'y', 'z']

char_list = ['a', 'b', 'c', 'd', 'e', 'f', 'g',
             'h', 'i', 'j', 'k', 'l', 'm', 'n',
             'o', 'p', 'q', 'r', 's', 't',
             'u', 'v', 'w', 'x', 'y', 'z']

eos_token = "[EOS]"
bos_token = "[BOS]"
pad_token = "[PAD]"

"""
26个字母和[EOS]（用在最前面）、[BOS]（用在最后面）、[PAD]组合，一共28个字符。
"""

# 打开两个文件
source_path = 'source.txt'
target_path = 'target.txt'
with open(source_path, 'w') as f_source:
    pass
with open(target_path, 'w') as f_target:
    pass

# 生成数据
# 对一个英文字符串，输出加密后的字符串。
# 加密方式：对于每个字符，使其ASCII码值循环-5，然后将整个字符串逆序。
# for循环定义10000个数据
for _ in range(10000):
    # 先定义一个源和目标字符串
    source_string = ''  # 源字符串
    target_string = ''  # 目标字符串
    # 生成数据长度
    for idx in range(random.randint(3, 13)):
        i = random.randint(0, 25)
        source_string += char_list[i]
        target_string += char_list[(i + 26 - 5) % 26]
    # 逆序目标字符串
    target_string = target_string[::-1]
    with open(source_path, 'a') as f_source:
        f_source.write(source_string + '\n')
    with open(target_path, 'a') as f_target:
        f_target.write(target_string + '\n')
