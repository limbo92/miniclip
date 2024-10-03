from torch.nn import Conv2d, MaxPool2d, BatchNorm2d, LayerNorm
import torch.nn.functional as F
from torch import nn
import torch
from typing import Optional

class ResidualBlock(nn.Module):

    def __init__(self,in_channels:int=1,output_channels:int=16,kernel_size:Optional[int]=3, stride:Optional[int]=1,padding:Optional[int]=1):
        super().__init__()
        self.conv1 = Conv2d(in_channels = in_channels, out_channels = output_channels, kernel_size = kernel_size, stride=stride, padding= padding)
        self.bn1 = BatchNorm2d(output_channels)
        self.conv2 = Conv2d(in_channels = output_channels, out_channels = output_channels, kernel_size = kernel_size, stride=1, padding= padding)
        self.bn2 = BatchNorm2d(output_channels)
        self.conv3 = Conv2d(in_channels=in_channels, out_channels=output_channels, kernel_size=1,
                            stride=stride, padding=0) # In this case(residual connection), convoluted each element, I think kernel_size is 1 only for this case.
        self.bn3 = BatchNorm2d(output_channels)

    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)))
        y = F.relu(self.bn2(self.conv2(y)))
        z = F.relu(self.bn3(self.conv3(x)))
        return z+y

class ImgEncoder(nn.Module):
    def __init__(self,in_channels:int=1,output_channels:int=16,kernel_size:Optional[int]=3, stride:Optional[int]=1,padding:Optional[int]=1):
        super().__init__()
        self.res_block1 = ResidualBlock(in_channels=in_channels, output_channels=output_channels, kernel_size= kernel_size, stride=stride, padding= padding)
        self.res_block2 = ResidualBlock(in_channels=output_channels, output_channels=int(output_channels/2), kernel_size= kernel_size, stride=stride, padding= padding)
        self.res_block3 = ResidualBlock(in_channels=int(output_channels/2), output_channels=1, kernel_size= kernel_size, stride=stride, padding= padding)
        self.wi =  nn.Linear(in_features=16,out_features=8)
        self.ln = LayerNorm(normalized_shape=8)

    def forward(self,x):
        x = self.res_block1(x) #(60000,1,14,14)
        x = self.res_block2(x) #(60000,1,7,7)
        x = self.res_block3(x) #(60000,1,4,4)
        x = self.wi(x.view(x.size()[0],-1)) #reshape(view, not real reshape) the size(N,1,H,W) to (N,H*W)
        x = self.ln(x)
        return x


if __name__ == '__main__':
    from dataset import MNISTData

    test_data = torch.randn(1, 1, 28, 28)
    res = ResidualBlock(in_channels=test_data.size()[1],stride=2)
    r = res(test_data)
    print(r.size())
    img_encoder = ImgEncoder(in_channels=test_data.size()[1],stride=2)
    img_encoder_result = img_encoder(test_data)
    print(img_encoder_result)
    #
    # ds = MNISTData(is_local=True,is_train=True)
    # images = ds.images
    # print(images.dtype())
    # images = images.unsqueeze(dim=1) # 给数据增加channel维度，以前的大小是N*H*W，现在是N*C*H*W
    # images = images.to(torch.float32) # 修改数据类型
    # print(images.dtype)
    # labels = ds.labels
    # print(images.size())
    # images_size = images.size()
    # res = ResidualBlock(in_channels=1)
    # r = res(images)
    # print(r.size())
