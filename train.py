from model import Transformer
from torch.utils.data import DataLoader
from data import MyDataset

my_model = Transformer()
my_dataset = MyDataset('./source.txt', './target.txt')
my_dataLoader = DataLoader(my_dataset, batch_size=8, shuffle=True)

for input_id, input_mask, output_id, output_mask in my_dataLoader:
    pass
