# 手写Transformer

这个项目用于学习/练习手写一个Transformer,旨在加强代码能力以及对Transformer的理解.

相关视频[参考这里](https://space.bilibili.com/568468320/channel/collectiondetail?sid=3387496.


## 数据集和训练

`data_generate.py`用于生成数据,并将其以txt的形式写入到`source.txt`和`target.txt`两个文件中.在该数据集规则下,源数据中的每一个字母循环减五后,对整个字符串进行逆序,得到最终的目标字符串.

上述两个txt被添加到了`.gitignore`中,所以当需要数据的时候,需要手动进行数据生成.
