import torch
import torch.nn as nn
import torch.nn.functional as F

import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
import numpy as np
import time
from collections import defaultdict
metrics_dict = defaultdict(list)
compression_dict = defaultdict(list)
percentage_of_layers = 0.4
number_of_decimal = 5
def generate_mask_array(array_len):
    num_ones = int(array_len * percentage_of_layers)
    num_zeros = array_len - num_ones
    arr = np.array([1] * num_ones + [0] * num_zeros)
    np.random.shuffle(arr)
    return arr


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        num_filter = int(planes/16)
        self.conv1 = nn.ModuleList([
            nn.Conv2d(in_planes, 16, kernel_size=3, stride=stride, padding=1, bias=False) for i in range(num_filter)])
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.ModuleList([
            nn.Conv2d(planes, 16, kernel_size=3, stride=1, padding=1, bias=False) for i in range(num_filter)])
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = [b(x) for b in self.conv1]
        out = torch.cat(out, dim=1)
        out = F.relu(self.bn1(out))
        out = [b(out) for b in self.conv2]
        out = torch.cat(out, dim=1)
        out = self.bn2(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        out = F.log_softmax(out, dim=1)
        return out


def compress_grad_single(giant_numpy_array):
    global metrics_dict 
    print ("Giant array {}".format(giant_numpy_array.shape))
    giant_numpy_array = np.round(giant_numpy_array, decimals=number_of_decimal)
    giant_string = giant_numpy_array.tostring()
    giant_string_compressed = zlib.compress(giant_string, level=9)
    compression_ratio = len(giant_string)/float(len(giant_string_compressed))
    metrics_dict['compression_giant_array'].append(compression_ratio)
    print ("Compression Ratio giant string {}".format(compression_ratio))

def do_something_grad(grad_data, layer_count):
    # import ipdb; ipdb.set_trace()
    print (grad_data.shape)
    grad_data_raster = grad_data.view(-1)
    grad_data_numpy = grad_data_raster.numpy()
    grad_data_numpy = np.round(grad_data_numpy, decimals=number_of_decimal)
    grad_data_string = grad_data_numpy.tostring()
    grad_data_compressed = zlib.compress(grad_data_string, level=9)
    compression_ratio = len(grad_data_string)/float(len(grad_data_compressed))
    print ("Compression Ratio one by one{}".format(compression_ratio))
    grad_data_uncompressed = zlib.decompress(grad_data_compressed)
    grad_data_numpy_unc  = np.frombuffer(grad_data_uncompressed, dtype=np.float32)
    grad_tensor = torch.from_numpy(grad_data_numpy_unc)
    grad_tensor = grad_tensor.reshape(grad_data.shape)
    grad_tensor = grad_tensor.float()
    return (grad_tensor)

def train(model, device, train_loader, optimizer, epoch):
    global metrics_dict
    model.train()
    step_count = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        metrics_dict['loss'].append(loss)
        loss.backward()
        temp_array = np.random.randint(0, high=2, size=10)
        
        # import ipdb; ipdb.set_trace()
        layer_count = 0
        single_list = list()
        for param in model.parameters():
            grad_val = param.grad.data
            grad_val = grad_val.view(-1)
            grad_val = grad_val.to("cpu").numpy()
            single_list.append(grad_val)
        # import ipdb; ipdb.set_trace()
        final_numpy_array = np.concatenate(single_list, axis=None)
        compress_grad_single(final_numpy_array)
        
        for param in model.parameters():
            # import ipdb; ipdb.set_trace()
            grad_val = param.grad.data.to("cpu")
            temp_mod = do_something_grad(grad_val, layer_count)
            param.grad.data = temp_mod

        optimizer.step()
        if batch_idx % 20 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test(model, device, test_loader):
    global metrics_dict
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    metrics_dict['accuracy'].append(100. * (correct/float(
        len(test_loader.dataset))))
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def main():

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

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

    model = ResNet(BasicBlock, [2,2,2,2])
    model = model.to(device)
    optimizer = optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad,
                                            model.parameters()), lr=0.01) 
    for epoch in range(30):
        tic = time.time()
        train(model, device, train_loader, optimizer, epoch)
        toc = time.time()
        print ("Time taken for an epoch = {}".format(toc-tic))
        metrics_dict["Time per epoch"].append(toc-tic)
        test(model, device, test_loader)

    with open("./5digit_bninc_resnet_stats.json", 'w') as f:
        json.dump(metrics_dict, f, indent=4)

if __name__ == '__main__':
    device = "cuda"
    main()
