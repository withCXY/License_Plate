import cv2
import torch
from matplotlib import pyplot as plt
import torch.nn as nn
from torchvision import datasets, transforms
import torch.nn.functional as F

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(120, 84)
        self.relu4 = nn.ReLU()
        self.fc3 = nn.Linear(84, 43)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = self.relu3(self.fc1(x))
        x = self.relu4(self.fc2(x))
        x = self.fc3(x)
        return x

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.load('./model_le.pth')  # 加载模型
    model = model.to(device)
    model.eval()  # 把模型转为test模式

    # 读取要预测的图片
    img = cv2.imread("./train/chars2/W/520-3.jpg")


    trans = transforms.Compose(
        [transforms.ToTensor()])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 图片转为灰度图，因为mnist数据集都是灰度图
    img = trans(img)
    img = img.to(device)
    img = img.unsqueeze(0)
    #img = F.pad(img, (6, 6, 6, 6), mode='constant', value=0)

    output = model(img)
    # prob = F.softmax(output, dim=1)
    # print("概率：", prob)
    value, predicted = torch.max(output.data, 1)
    predict = output.argmax(dim=1)
    # print("预测类别：", chr(48+predict.item()))
