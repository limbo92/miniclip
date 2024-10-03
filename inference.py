from cProfile import label
import os
import torch
from torch.utils.data import DataLoader
from dataset import MNISTData
from clip import MiniClip
from img_encoder import ImgEncoder


#
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
model_dir_path = os.path.realpath("./model/")
MODEL_NAME = "model_epoch_4.pth"
TARGET_COUNT = 10
# load test data
td = MNISTData(is_train=False, is_local=True)

# load trained model
model = MiniClip(in_channels=1, stride=2, num_embedding=td.labels.unique().size()[0], embedding_dim=16,
                 output_dim=8).to(DEVICE)
model.load_state_dict(torch.load(os.path.join(model_dir_path, MODEL_NAME),
                                 weights_only=True))  #加载模型, weights_only Indicates whether unpickler should be restricted to loading only tensors, primitive types, dictionaries
model.eval()  # 启动模型预测模式，禁止dropout或者batch normalization更新running estimated mean/std，使用训练集中得到的running estimated mean/std进行正则化

# setup test dataloader
ts_dataloader = DataLoader(td, shuffle=True, batch_size=100, num_workers=10, persistent_workers=True)

#测试 图像-》文字分类
with torch.no_grad():  # 让模型不计算梯度
    for i, (mini_images, mini_labels) in enumerate(ts_dataloader, 0):
        image = mini_images.unsqueeze(1).float().to(DEVICE)
        label = mini_labels.to(DEVICE)
        ## 计算单个图像与0-9数字文本的相似度，取logic的最大值作为预测值
        text_category = torch.arange(TARGET_COUNT).to(DEVICE)  # 构造0-9数字文本
        logics = model(image, text_category)
        predict = logics.argmax(dim=1)
        #print(logics)
        print("accuracy is %s"%(sum(label == predict)/100))
        print("正确分类:%s，CLIP 分类:%s" % (label, predict))


# 测试图像相似度，因为在训练的时候反向传播更新了ImageEncoder的参数，ImageEncoder也学习到了图像的表征，可以区分不同图像

# with torch.no_grad():
#     for i, (mini_images, mini_labels) in enumerate(ts_dataloader, 0):
#         other_images, other_labels = next(iter(ts_dataloader))
#         img_encoder = ImgEncoder(in_channels=1, stride=2).to(DEVICE)
#         #
#         mini_images = mini_images.unsqueeze(1).float().to(DEVICE)
#         other_images = other_images.unsqueeze(1).float().to(DEVICE)
#         mini_labels = mini_labels.to(DEVICE)
#         other_labels = other_labels.to(DEVICE)
#         #
#         mini_images_embedding = img_encoder(mini_images)
#         other_images_embedding = img_encoder(other_images)
#         logics = torch.matmul(mini_images_embedding,other_images_embedding.T)
#         predict = other_labels[logics.argmax(dim=1)]
#         print("accuracy is %s" % (sum(mini_labels == predict).item() / 100))
#         break



