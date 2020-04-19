import dataset
import matplotlib.pyplot as plt
import numpy as np
import torchvision
from model import LeNet
import torch.optim as optim
import torch.nn as nn
import torch
import torch.nn.functional as F

def imshow(data_loader):
    # img = img / 2 + 0.5
    data_iter = iter(data_loader)
    images, labels = data_iter.next()
    img = torchvision.utils.make_grid(images)

    npimg = img.numpy()

    print(labels)
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def train(epochs, learning_rate, momentum, train_loader, save_path):
    net = LeNet()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum)

    for epoch in range(epochs):
        running_loss = 0.0

        for i, data in enumerate(train_loader):
            #  将输入放到GPU
            inputs, labels = data[0].to(device), data[1].to(device)
            #  初始化grad
            optimizer.zero_grad()

            outputs = net(inputs)

            loss = criterion(outputs, labels)  # 计算loss
            # 反向传播
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %f' %
                      (epoch + 1, i + 1, running_loss / 2000))

                running_loss = 0.0

    print("Finished Training")

    torch.save(net.state_dict(), save_path)

def test(path, test_loader):
    net = LeNet()
    net.load_state_dict(torch.load(path))
    test_loss = 0
    correct = 0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data

            output = net(inputs)
            loss = criterion(output, labels)
            test_loss += loss
            _, predicted = torch.max(output, 1)
            c = (predicted == labels).squeeze()

            for i in range(4):
                correct += c[i].item()
        
        test_loss /= len(test_loader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))


def main():
    train_loader, test_loader = dataset.get_data()
    # imshow(train_loader)
    PATH = r'D:\Programme\Python\DL\Pytorch\LeNet\model\mnist_lenet.pth'
    train(5, 0.003, 0.5, train_loader, PATH)
    test(PATH, test_loader)

main()