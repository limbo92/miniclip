from dataset import MNISTData
from clip import MiniClip
import torch
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader
from torch.nn import functional as F
import os

# setup constant
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
model_file_path = os.path.realpath("./model/model.pth")
epoch = 10
load_model_from_local = True #
model=None
#load dataset
data = MNISTData(is_train=True, is_local=True)
images = data.images
labels = data.labels
#load checkpoint of model
if load_model_from_local:
    model = MiniClip(in_channels=1, num_embedding=labels.unique().size()[0], embedding_dim=16, output_dim=8,stride=2).to(DEVICE)
else:
    model = MiniClip(in_channels=1, num_embedding=labels.unique().size()[0], embedding_dim=16, output_dim=8,
                     stride=2).to(DEVICE)
    model.load_state_dict(torch.load(model_file_path)) #从已经训练过的模型里加载参数
#
BATCH_SIZE = 60
TARGET_COUNT=10
#setup Optimizer
optimizer = Adam(model.parameters(),lr=5e-4) #输入要被优化的参数，设定学习率learning rate
#DataLoader
dataloader = DataLoader(dataset=data, shuffle=True, batch_size=BATCH_SIZE, num_workers=10, persistent_workers=True) #persistent_workers是True的话，在所有epoch中worker都是激活的，从不terminate。只有当persistent_workers是False的时候，当一次epoch结束，worker被中断，下一次epoch开始再重新激活，False适用于开启或关闭worker不需要很多时间，或者计算资源受限，只有关闭workers, 剩下的步骤才可以顺利进行。

#Training
for i in range(epoch):
    for idx,(mini_images,mini_labels) in enumerate(dataloader,0):
        #print(idx)
        mini_images = mini_images.unsqueeze(dim=1).float()
        if torch.unique(mini_labels).shape[0] < TARGET_COUNT:  # 未覆盖10种数字
            continue
        # 挑选出10个数字，如果label里数字有重复，比如我们的target_count是5,data size也就是5. 但是第一个labels和第3个label是相同的，那么第一个图像和第一个label与第三个label点积后的值应该都是1,但是第一个图像对应的真实的y_true是[1,0,0,0,0](请记住CLIP中只有对角线位置的值是1，其余都是0),并没有告诉它第二个位置(从0算起)的y值也是1，这样会阻止把第一个图像与第三个label拉进，从而降低模型的精准度。
        target = set()
        indexes = []
        for j in range(BATCH_SIZE):
            if mini_labels[j].item() in target: #labels[j]输出的是tensor type，比如 tensor([5]). labels[j].item()输出的直接是5
                continue
            target.add(mini_labels[j].item())
            indexes.append(j)
            if len(target) == TARGET_COUNT:
                break
        mini_images = mini_images[indexes].to(DEVICE) # size是(10,1,28,28)
        mini_labels = mini_labels[indexes].to(DEVICE) # size是(10,)
        # 计算模型预测值
        predict = model(mini_images, mini_labels)
        # 构造target, 对角线的值是1，我们只需要告诉cross_entropy，第几个位置是1就可以，不需要传入一个矩阵。把它当成是一个多分类任务，图像与哪个文字最相似，我们一共有target_count个文字，在我们构造的数据集中(image-text pair)，相似的文字和图片在各自的数据中的位置是一致的，所以第n个图像，它的y_true就是[0,0,...,n,...]。
        target = torch.arange(TARGET_COUNT).to(DEVICE) #结果是[0,1,2,3,4,5,6,7,,8,9], 就告诉了cross_entropy，第一个图像的y_true是[1,0,0,...,target_num-1]
        # 计算loss，loss计算两次，一次是image->所有文字。 另一个是每个文字-> 所有Image。
        loss_i = F.cross_entropy(input = predict, target= target)
        loss_t = F.cross_entropy(input=predict.permute(1,0), target=target) #permute,对tensor重排列，二维的话就是转置。比如tensor.size是(2,3)，permute(1,0)将是tensor.size重排列为(3,2)
        loss = (loss_i+loss_t)/2
        # 优化
        optimizer.zero_grad() #清空之前的梯度
        loss.backward() #反向传播
        optimizer.step() #基于算出的梯度，对参数进行更新
        # 保存模型的check_point
        if idx % 100 == 0:
            print('epoch:{}, iter:{},loss:{}'.format(i, idx, loss))
            #torch.save(model.state_dict(), './model/model_epoch_%s_iteration_%s.pth'%(i,idx))
    # 保存模型的check_point
    torch.save(model.state_dict(), './model/model_epoch_%s.pth'%i)
