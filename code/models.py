import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision.transforms as transforms
from torchvision.models import resnet18
import copy
import torchvision.utils as utils

def select_model(args, train_dataset):
    if args.model == 'cnn':
        return CNNMnist(args=args)
    elif args.model == 'resnet' : # CIFAR-10에 맞추져 있음.
        model = resnet18(pretrained=False)
        # 1. 첫 번째 conv layer를 수정
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        # 2. maxpool 제거 (CIFAR에선 필요 없음)
        model.maxpool = nn.Identity()
        # 3. 마지막 FC층을 CIFAR 클래스 수에 맞게 수정
        model.fc = nn.Linear(model.fc.in_features, 10)
        return model
    else:
        raise NotImplementedError
    

# ---------- 이미지 생성 (수정된 버전) ----------
def generate_images(generator, idxs, dataset, device='cpu', z_dim=100, num_generate=None):
    """DCGAN용 이미지 생성 함수"""
    generator.eval()
    device = torch.device(device)
    
    if num_generate is None:
        num_samples = len(idxs)
    else:
        num_samples = num_generate
    
    # DCGAN 노이즈 생성: (batch, z_dim, 1, 1)
    noise = torch.randn((num_samples, z_dim, 1, 1), device=device)
    
    with torch.no_grad():
        gen_imgs = generator(noise)
        gen_imgs = gen_imgs.cpu()
    
    # 라벨 생성
    if len(idxs) >= num_samples:
        sample_idxs = list(idxs)[:num_samples]
    else:
        sample_idxs = (list(idxs) * ((num_samples // len(idxs)) + 1))[:num_samples]
    
    labels = torch.tensor([dataset[i][1] for i in sample_idxs], dtype=torch.long)
    
    return gen_imgs, labels



# ---------- 이미지 필터링 (수정된 버전) ----------
def filter_images(discriminator, images, labels, threshold=0.7, device='cpu'):
    """
    DCGAN Discriminator로 고품질 이미지 필터링
    """
    discriminator.eval()
    device = torch.device(device)
    
    # 모든 텐서를 같은 디바이스로 이동
    images = images.to(device)
    labels = labels.to(device)
    
    with torch.no_grad():
        preds = discriminator(images).squeeze()
        mask = preds > threshold
        
        # 마스크도 같은 디바이스에 있는지 확인
        mask = mask.to(device)
        
        # 필터링 수행
        filtered_imgs = images[mask]
        filtered_labels = labels[mask]
        
        # CPU로 다시 이동 (메모리 절약)
        filtered_imgs = filtered_imgs.cpu()
        filtered_labels = filtered_labels.cpu()
    
    print(f"[DCGAN Filter] {len(images)} -> {len(filtered_imgs)} images (threshold={threshold})")
    
    return filtered_imgs, filtered_labels



# ---------- 모델 구조들 ----------
class MLP(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out):
        super(MLP, self).__init__()
        self.layer1 = nn.Linear(dim_in, dim_hidden)
        self.layer2 = nn.Linear(dim_hidden, dim_out)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.layer1(x))
        x = self.layer2(x)
        return x


class CNNMnist(nn.Module):
    def __init__(self, args):
        super(CNNMnist, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, args.num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class CNNCifar(nn.Module):
    def __init__(self, args):
        super(CNNCifar, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, args.num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# ---------- 가중치 초기화 함수 ----------
def weights_init(w):
    """DCGAN 가중치 초기화"""
    classname = w.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(w.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(w.weight.data, 1.0, 0.02)
        nn.init.constant_(w.bias.data, 0)


# ---------- DCGAN Generator (CIFAR-10용) ----------
class Generator(nn.Module):
    def __init__(self, z_dim=100, img_shape=(3, 32, 32)):
        super(Generator, self).__init__()
        self.z_dim = z_dim
        self.img_shape = img_shape

        self.tconv1 = nn.ConvTranspose2d(z_dim, 512, 4, 1, 0, bias=False)  # 1 → 4
        self.bn1 = nn.BatchNorm2d(512)
        self.tconv2 = nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False)    # 4 → 8
        self.bn2 = nn.BatchNorm2d(256)
        self.tconv3 = nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False)    # 8 → 16
        self.bn3 = nn.BatchNorm2d(128)
        self.tconv4 = nn.ConvTranspose2d(128, img_shape[0], 4, 2, 1, bias=False)  # 16 → 32

    def forward(self, x):
        if x.dim() == 2:
            x = x.view(x.size(0), x.size(1), 1, 1)
        x = F.relu(self.bn1(self.tconv1(x)))
        x = F.relu(self.bn2(self.tconv2(x)))
        x = F.relu(self.bn3(self.tconv3(x)))
        x = torch.tanh(self.tconv4(x))
        return x



# ---------- DCGAN Discriminator (CIFAR-10용) ----------
class Discriminator(nn.Module):
    def __init__(self, img_shape=(3, 32, 32)):
        super(Discriminator, self).__init__()
        self.img_shape = img_shape

        self.conv1 = nn.Conv2d(img_shape[0], 64, 4, 2, 1, bias=False)       # 32 → 16
        self.conv2 = nn.Conv2d(64, 128, 4, 2, 1, bias=False)                # 16 → 8
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, 4, 2, 1, bias=False)               # 8 → 4
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 1, 4, 1, 0, bias=False)                 # 4 → 1

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), 0.2, True)
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.2, True)
        x = F.leaky_relu(self.bn3(self.conv3(x)), 0.2, True)
        x = torch.sigmoid(self.conv4(x))
        return x



# ---------- ResNet 아키텍처 ----------
class BasicBlock(nn.Module):
    expansion = 1
    
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
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

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    #  누락된 _make_layer 메서드 추가
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
        # 64x64 입력에 맞게 조정
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18(num_classes=10):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)


