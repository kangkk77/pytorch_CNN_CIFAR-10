from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18


batch_size = 128
log_interval = 100
DATA_DIR = Path(__file__).resolve().parent / 'data'
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


def build_model():
    # 从头训练：使用 torchvision 的 ResNet-18，不加载 ImageNet 预训练权重。
    net = resnet18(weights=None, num_classes=10)

    # CIFAR-10 图片为 32×32，使用小卷积并取消开头的下采样。
    net.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1,
                          padding=1, bias=False)
    nn.init.kaiming_normal_(net.conv1.weight, mode='fan_out',
                            nonlinearity='relu')
    net.maxpool = nn.Identity()
    return net


def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))


def main():
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(
        root=str(DATA_DIR), train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=0)

    testset = torchvision.datasets.CIFAR10(
        root=str(DATA_DIR), train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=0)

    images, labels = next(iter(trainloader))
    imshow(torchvision.utils.make_grid(images[:4]))
    print(' '.join('%5s' % classes[labels[j]] for j in range(4)))

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    net = build_model().to(device)

    # 沿用原脚本的损失函数、优化器和 20 轮训练设置。
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

    net.train()
    for epoch in range(20):
        running_loss = 0.0
        running_samples = 0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * labels.size(0)
            running_samples += labels.size(0)
            if (i + 1) % log_interval == 0 or (i + 1) == len(trainloader):
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / running_samples))
                running_loss = 0.0
                running_samples = 0

    print('Finished Training')

    net.eval()
    images, labels = next(iter(testloader))
    imshow(torchvision.utils.make_grid(images[:4]))
    print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]]
                                   for j in range(4)))

    with torch.no_grad():
        outputs = net(images.to(device))
        predicted = outputs.argmax(dim=1).cpu()
    print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                                 for j in range(4)))

    correct = 0
    total = 0
    class_correct = [0] * len(classes)
    class_total = [0] * len(classes)
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            predicted = outputs.argmax(dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            for label, prediction in zip(labels.cpu().tolist(),
                                         predicted.cpu().tolist()):
                class_correct[label] += int(prediction == label)
                class_total[label] += 1

    print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total))
    for i in range(len(classes)):
        print('Accuracy of %5s : %2d %%' % (
            classes[i], 100 * class_correct[i] / class_total[i]))


if __name__ == '__main__':
    main()
