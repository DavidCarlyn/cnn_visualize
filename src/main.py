from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets
from torchvision.transforms import ToTensor

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.pool1(x)
        x = torch.flatten(x, 1)
        return x

class Classifier(nn.Module):
    def __init__(self, in_size, out_size):
        super().__init__()
        self.layer = nn.Linear(in_size, out_size)

    def forward(self, x):
        x = self.layer(x)
        return x

def get_data():
    
    train_dataset = datasets.MNIST(
        root = '../data/datasets',
        train = True,
        transform = ToTensor(),
        download = True,
    )
    test_dataset = datasets.MNIST(
        root = '../data/datasets',
        train = False,
        transform = ToTensor()
    )

    return train_dataset, test_dataset

if __name__ == "__main__":
    train_dset, test_dset = get_data()
    cnn = CNN()
    classifier = Classifier(12544, 10)

    optimizer = optim.SGD(list(cnn.parameters())+list(classifier.parameters()), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()

    print(f"Dataset length: {len(train_dset)}")
    for epoch in range(10):
        total_loss = 0
        for i in tqdm(range(len(train_dset))):
            imgs, lbls = train_dset[i]
            feat = cnn(imgs.unsqueeze(0))
            out = classifier(feat)
            loss = loss_fn(out, torch.tensor(lbls).unsqueeze(0))
            total_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"{epoch} | {total_loss}")


    #classifier = Classifier()