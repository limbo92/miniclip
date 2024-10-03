from img_encoder import ImgEncoder
from text_encoder import TextEncoder
import torch.nn as nn
import torch
from dataset import MNISTData


class MiniClip(nn.Module):
    def __init__(self, in_channels,stride, num_embedding, embedding_dim, output_dim):
        super().__init__()
        self.img_encoder = ImgEncoder(in_channels=in_channels, stride=stride)
        self.text_encoder = TextEncoder(num_embedding=num_embedding, embedding_dim=embedding_dim, output_dim=output_dim)



    def forward(self,img_x,text_x):
        img_x = self.img_encoder(img_x)
        text_x = self.text_encoder(text_x)
        return torch.matmul(img_x,text_x.T)



if __name__ == '__main__':
    DEVICE:str = 'cuda' if torch.cuda.is_available() else 'cpu'  # 设备

    ds = MNISTData(is_local=True,is_train=True)
    images = ds.images
    print(images.dtype)
    images = images.unsqueeze(dim=1) # 给数据增加channel维度，以前的大小是N*H*W，现在是N*C*H*W
    images = images.float() # 修改数据类型
    print(images.dtype)
    labels = ds.labels
    print(labels.dtype)
    print(images.size())
    print(labels.size())
    model = MiniClip(in_channels=1, num_embedding=labels.unique().size()[0], embedding_dim=16, output_dim=8,stride=2).to(DEVICE)
    result = model(images.to(DEVICE), labels.to(DEVICE))
    print(result.size())



