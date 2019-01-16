import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import argparse
from Resnet import ResNet18

# choose your device 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# argparse
parser = argparse.ArgumentParser(description='CIFAR10 model')
parser.add_argument('--output', default='./model', help='model_saving path') 
parser.add_argument('--net', default='./model/Resnet18.pth', help='continue training') 
parser.add_argument('--learning_rate', default=0.01, help='learning_rate')
parser.add_argument('--batch_size', default=128, help='batch_size')
parser.add_argument('--epoch', default=75, help='training epochs')
args = parser.parse_args()

# hyperparameter
epoch_total = args.epoch
batch_size = args.batch_size      
learning_rate = args.learning_rate        

# prepare dataset
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),  
    transforms.RandomHorizontalFlip(),  
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), 
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform_train) 
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)   

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

# CIFAR-10 labels
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# load model
net = ResNet18().to(device)

# define loss and optimizer
criterion = nn.CrossEntropyLoss()  
optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4) 

# training
if __name__ == "__main__":
    print("Start Training")  
    with open("acc.txt", "w") as f:
        with open("log.txt", "w")as f2:
            for epoch in range(epoch_total):
                print('\nEpoch: %d' % (epoch + 1))
                net.train()
                sum_loss = 0.0
                correct = 0.0
                total = 0.0
                for i, data in enumerate(trainloader, 0):
                    # data
                    length = len(trainloader)
                    inputs, labels = data
                    inputs, labels = inputs.to(device), labels.to(device)
                    optimizer.zero_grad()

                    # optimization
                    outputs = net(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    # print loss and accuracy after a batch
                    sum_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += predicted.eq(labels.data).cpu().sum()
                    print('[epoch:%d, iter:%d] Loss: %.03f | Acc: %.3f%% '
                          % (epoch + 1, (i + 1 + epoch * length), sum_loss / (i + 1), 100. * correct.float() / total))
                    f2.write('%03d  %05d |Loss: %.03f | Acc: %.3f%% '
                          % (epoch + 1, (i + 1 + epoch * length), sum_loss / (i + 1), 100. * correct.float() / total))
                    f2.write('\n')
                    f2.flush()

                # test accuracy after an epoch
                print("Start Testing!")
                with torch.no_grad():
                    correct = 0
                    total = 0
                    for data in testloader:
                        net.eval()
                        images, labels = data
                        images, labels = images.to(device), labels.to(device)
                        outputs = net(images)
                        # find the classification
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum()
                    print('test accuracy：%.3f%%' % (100 * correct.float() / total))
                    acc = 100. * correct.float() / total
                    # 将每次测试结果实时写入acc.txt文件中
                    print('Saving model......')
                    torch.save(net.state_dict(), '%s/net_%03d.pth' % (args.output, epoch + 1))
                    f.write("EPOCH=%03d,Accuracy= %.3f%%" % (epoch + 1, acc))
                    f.write('\n')
                    f.flush()
            print("Training Finished")
