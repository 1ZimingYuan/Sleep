

import numpy as np
import torch
#from utils import plot_images
from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
import torch.utils.data as tdata
from torch.utils.data import Dataset
mean=[0.0049358984, 0.007318254, -9.328728e-09, 0.04608545, 0.016477318,
      0.013598264, 0.037235405, -0.025455898, -0.020063546, 0.017442517]
std=[16.951437, 17.272545, 0.0066356324, 17.932423, 16.331614,
     10.850225, 17.980978, 23.723705, 11.272454, 0.18868428]
class Mydataset(Dataset):
    def __init__(self,data_path):
        super(Mydataset, self).__init__()
        with open(data_path,'r') as f:
            lines=[line.strip().strip('\n') for line in f.readlines() if line!='' and line!='\n']
            data_path_list=[item.split()[0] for item in lines]
            lable_list=[int(item.split()[1]) for item in lines]
        assert len(data_path_list)==len(lable_list), "The number of samples doesn't equal that of lables!"
        image=[]
        for i in range(len(data_path_list)):
            image.append((data_path_list[i],lable_list[i]))
        self.image=image
        self.transform=transforms.Normalize(mean,std)
        #self.labletransform=lable_tramsform

    def __getitem__(self, index):
        image_path, lable_ = self.image[index]
        image_=np.load(image_path)
        image_temp=torch.tensor(image_)
        temp=[]
        temp.extend(image_temp.chunk(10,0))
        image_1=torch.stack(temp,0)
        image=self.transform(image_1)
        lable=torch.tensor(lable_,dtype=torch.int64)
        return image,lable

    def __len__(self):
        return len(self.image)

def get_train_valid_loader(data_dir,
                           batch_size,

                           random_seed,
                           valid_size=0.2,
                           shuffle=True,
                           show_sample=False,
                           num_workers=4,
                           pin_memory=False):

    error_msg = "[!] valid_size should be in the range [0, 1]."
    assert ((valid_size >= 0) and (valid_size <= 1)), error_msg



    # load the dataset
    train_dataset = Mydataset(data_path=data_dir)
    valid_dataset = Mydataset(data_path=data_dir)

    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))

    if shuffle:
        #np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = tdata.DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    valid_loader = tdata.DataLoader(
        valid_dataset, batch_size=batch_size, sampler=valid_sampler,
        num_workers=num_workers, pin_memory=pin_memory,
    )


    return (train_loader, valid_loader)


def get_test_loader(data_dir,
                    batch_size,
                    shuffle=True,
                    num_workers=4,
                    pin_memory=False):
    """
    Utility function for loading and returning a multi-process
    test iterator over the CIFAR-10 dataset.
    If using CUDA, num_workers should be set to 1 and pin_memory to True.
    Params
    ------
    - data_dir: path directory to the dataset.
    - batch_size: how many samples per batch to load.
    - shuffle: whether to shuffle the dataset after every epoch.
    - num_workers: number of subprocesses to use when loading the dataset.
    - pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
      True if using GPU.
    Returns
    -------
    - data_loader: test set iterator.
    """
    normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010],
    )

    # define transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    dataset = datasets.CIFAR10(
        root=data_dir, train=False,
        download=True, transform=transform,
    )

    data_loader = tdata.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    return data_loader