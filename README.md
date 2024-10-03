# miniclip
 A simple version of CLIP. 
 I use MNIST dataset to train a CLIP model. 
 For the image embedding, I just use a simple convolutional neural network.
 For the text embedding, I just use a simple non-linear transformation.
 主要是按照索引里的代码复现的，稍微有一些改动，比如如何从本地进行加载MNIST数据，在复现的时候，有不太明白的地方(比如卷积后维度是如何计算，BatchNorm的一些基本概念,BatchNorm和LayerNorm区别，zero_grad和model.eval的区别等等)增加了一些注释，后续会上传到自己的公众号和博客，可以通过Reference查看
### Structure

```plaintext
.
├── LICENSE
├── README.md
├── clip.py #clip模型,将img_encoder和text_encoder生成的embedding向量进行匹配
├── data  MNIST数据集
│   └── MNIST
│       └── raw
│           ├── t10k-images-idx3-ubyte
│           ├── t10k-images-idx3-ubyte.gz
│           ├── t10k-labels-idx1-ubyte
│           ├── t10k-labels-idx1-ubyte.gz
│           ├── train-images-idx3-ubyte
│           ├── train-images-idx3-ubyte.gz
│           ├── train-labels-idx1-ubyte
│           └── train-labels-idx1-ubyte.gz
├── dataset.py
├── img_encoder.py #使用卷积对图像进行embedding
├── inference.py # 推理
├── model
│   ├── model_epoch_0.pth
│   ├── model_epoch_1.pth
│   ├── model_epoch_2.pth
│   ├── model_epoch_3.pth
│   ├── model_epoch_4.pth
│   ├── model_epoch_5.pth
│   ├── model_epoch_6.pth
│   ├── model_epoch_7.pth
│   ├── model_epoch_8.pth
│   └── model_epoch_9.pth
├── text_encoder.py #使用简单的非线性转换对文本进行embedding
└── train.py #训练模型
```
### Performance
1. Image->Text 分类: 精准度差不多95%以上，学习率取0.001和0.0005的效果都差不多
2. 查找相似的Image: 精准度在20%左右，应该是CLIP的目标函数或者卷积的不够复杂导致的，CLIP的目标函数主要还是拉进相似的图像和文本的距离，拉开不相似的图像和文本，对于学习图像表征来说还是困难了点。
### Reference
#### Blog
#### Code
https://github.com/owenliang/mnist-clip/tree/main