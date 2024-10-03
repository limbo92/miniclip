import torch
import torchvision
from torch.utils.data import Dataset
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from modelscope.msdatasets import MsDataset
from torchvision.transforms.v2 import PILToTensor, Compose

print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.current_device())
print(torch.cuda.get_device_name())
print(torch.__file__)


class MNISTData(Dataset):
    def __init__(self, is_train=True, is_local=True, local_path="/home/idal-05/pythonProject/mnist-clip/data/"):
        super().__init__()
        self.train = is_train
        self.local= is_local
        if is_local:
            total_ds = torchvision.datasets.MNIST(root=local_path, train=self.train,
                                                  download=False)
            if is_train:
                self.images = total_ds.data
                self.labels = total_ds.targets
            else:
                self.images = total_ds.data
                self.labels = total_ds.targets
        else:
            self.ds = torchvision.datasets.MNIST(root="", train=False,
                                                 download=False)
            # dataset = load_dataset("ylecun/mnist")
            if is_train:
                self.ds = torchvision.datasets.MNIST(root="./mnist/", train=is_train, download=False)
                #self.ds = MsDataset.load('modelscope/mnist', subset_name='mnist', split='train', data_dir="/home/idal-05/pythonProject/mnist-clip/data")
            else:
                self.ds = torchvision.datasets.MNIST(root="./mnist/", train=is_train, download=False)
                #self.ds = MsDataset.load('modelscope/mnist', subset_name='mnist', split='train',
                #data_dir="/home/idal-05/pythonProject/mnist-clip/data")
        self.img_convert = Compose([PILToTensor()])

    def __len__(self):
        length = len(self.images)
        return length

    def __getitem__(self, item):
        image, label = self.images[item], self.labels[item]
        if self.local:
            return image/255.0, label
        else:
            return self.img_convert(image) / 255.0, label


if __name__ == '__main__':
    mnist = MNISTData()
    print(mnist.__len__())
    print(mnist.__getitem__(0))
