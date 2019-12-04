import os
import pdb
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import MultiStepLR

import torchvision
from torchvision.utils import make_grid
from torchvision import datasets, transforms

from util.misc import CSVLogger

from model.resnet import ResNet18, ResNet50
from model.wide_resnet import WideResNet
from tensorboardX import SummaryWriter

model_options = ['resnet18', 'resnet50', 'wideresnet']
dataset_options = ['cifar10', 'cifar100', 'svhn']

parser = argparse.ArgumentParser(description='CNN')
parser.add_argument('--dataset', '-d', default='cifar10',
                    choices=dataset_options)
parser.add_argument('--model_type', '-a', default='resnet18',
                    choices=model_options)
parser.add_argument('--batch_size', type=int, default=256,
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=120,
                    help='number of epochs to train (default: 20)')
parser.add_argument('--lr', type=float, default=0.001,
                    help='learning rate')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed (default: 1)')
parser.add_argument('--resume', type=bool, default=False,
                    help='restart')
parser.add_argument('--lr_decay', default=0.5, type=float,
                     help='learning decay for lr scheduler')
parser.add_argument('--beta', default=1.0, type=float,
                     help='beta distribution')
parser.add_argument('--prob', default=1.0, type=float,
                     help='learning decay for lr scheduler')


grad_clip = 10.0
weight_decay = 1e-4
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

args = parser.parse_args()
cudnn.benchmark = True
torch.manual_seed(args.seed)
lam = np.random.beta(args.beta, args.beta)
if device == 'cuda':
    torch.cuda.manual_seed(args.seed)

# save csv file
if not os.path.exists('logs'):
    os.makedirs('logs')
# save model
if not os.path.exists('checkpoints'):
    os.makedirs('checkpoints')
# save tensorboard file
if not os.path.exists('runs'):
    os.makedirs('runs')

test_id = str(args.dataset) + '_' + str(args.model_type)

normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                 std=[x / 255.0 for x in [63.0, 62.1, 66.7]])

train_transform = transforms.Compose([])
train_transform.transforms.append(transforms.RandomCrop(32, padding=4))
train_transform.transforms.append(transforms.RandomHorizontalFlip())
train_transform.transforms.append(transforms.ToTensor())
train_transform.transforms.append(normalize)


test_transform = transforms.Compose([
    transforms.ToTensor(),
    normalize])

if args.dataset == 'cifar10':
    num_classes = 10
    train_dataset = datasets.CIFAR10(root='data/',
                                     train=True,
                                     transform=train_transform,
                                     download=True)

    test_dataset = datasets.CIFAR10(root='data/',
                                    train=False,
                                    transform=test_transform,
                                    download=True)
elif args.dataset == 'cifar100':
    num_classes = 100
    train_dataset = datasets.CIFAR100(root='data/',
                                      train=True,
                                      transform=train_transform,
                                      download=True)

    test_dataset = datasets.CIFAR100(root='data/',
                                     train=False,
                                     transform=test_transform,
                                     download=True)
elif args.dataset == 'svhn':
    num_classes = 10
    train_dataset = datasets.SVHN(root='data/',
                                  split='train',
                                  transform=train_transform,
                                  download=True)

    extra_dataset = datasets.SVHN(root='data/',
                                  split='extra',
                                  transform=train_transform,
                                  download=True)

    # Combine both training splits (https://arxiv.org/pdf/1605.07146.pdf)
    data = np.concatenate([train_dataset.data, extra_dataset.data], axis=0)
    labels = np.concatenate([train_dataset.labels, extra_dataset.labels], axis=0)
    train_dataset.data = data
    train_dataset.labels = labels

    test_dataset = datasets.SVHN(root='data/',
                                 split='test',
                                 transform=test_transform,
                                 download=True)

# Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=args.batch_size,
                                           shuffle=True,
                                           pin_memory=True,
                                           num_workers=0)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=args.batch_size,
                                          shuffle=False,
                                          pin_memory=True,
                                          num_workers=0)

if args.model_type == 'resnet18':
    model = ResNet18(lam, num_classes=num_classes)
elif args.model_type == 'resnet50':
    cnn = ResNet50(num_classes=num_classes)
elif args.model_type == 'wideresnet':
    if args.dataset == 'svhn':
        cnn = WideResNet(depth=16, num_classes=num_classes, widen_factor=8,
                         dropRate=0.4)
    else:
        cnn = WideResNet(depth=28, num_classes=num_classes, widen_factor=10,
                         dropRate=0.3)

model = model.cuda()
criterion = nn.CrossEntropyLoss().cuda()
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
lr_step_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1) # Decay LR by a factor of 0.1 every step_size
torch.nn.utils.clip_grad_norm_(model.parameters(), 0) # Gradient Clipping

if args.dataset == 'svhn':
    scheduler = MultiStepLR(optimizer, milestones=[80, 120], gamma=0.1)
else:
    scheduler = MultiStepLR(optimizer, milestones=[60, 120, 160], gamma=0.2)

filename = 'logs/' + test_id + '.csv'
csv_logger = CSVLogger(args=args, fieldnames=['epoch', 'train_acc', 'test_acc'], filename=filename)

def test(loader, criterion):
    model.eval()    # Change model to 'eval' mode (BN uses moving mean/var).
    correct = 0.
    total = 0.
    is_train = False
    for images, labels in loader:
        images = images.cuda()
        labels = labels.cuda()

        with torch.no_grad():
            pred = model(images, is_train)

        val_loss = criterion(pred, labels)

        pred = torch.max(pred.data, 1)[1]
        total += labels.size(0)
        correct += (pred == labels).sum().item()

    val_acc = correct / total
    model.train()
    return val_acc, val_loss


for epoch in range(args.epochs):

    xentropy_loss_avg = 0. 
    correct = 0.
    total = 0.
    is_train = True

    progress_bar = tqdm(train_loader)
    for i, (images, labels) in enumerate(progress_bar):
        progress_bar.set_description('Epoch ' + str(epoch))

        images = images.cuda()
        #labels = labels.cuda()

        model.zero_grad()
        pred, lam, rand_idex, bx, by = model(images, True)

        label_a = labels
        label_b = labels[rand_idex]
        
        labels = labels.cuda()
        label_a = label_a.cuda()
        label_b = label_b.cuda()

        xentropy_loss = criterion(pred, label_a) * lam + criterion(pred, label_b) * (1. - lam)
        
        optimizer.zero_grad()
        xentropy_loss.backward()
        optimizer.step()

        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        xentropy_loss_avg += xentropy_loss.item()

        # Calculate running average of accuracy
        pred = torch.max(pred.data, 1)[1]
        total += labels.size(0)
        correct += (pred == labels.data).sum().item()
        accuracy = correct / total

        progress_bar.set_postfix(
            xentropy='%.3f' % (xentropy_loss_avg / (i + 1)),
            acc='%.3f' % accuracy)

    #grid = make_grid(images, nrow=3, padding=5, normalize=True)
    #writer.add_image('new1/images', grid, 0)
    print(bx, by)

    test_acc, test_loss = test(test_loader, criterion)
    tqdm.write('test_loss: %.3f' % (test_loss))
    tqdm.write('test_acc: %.3f' % (test_acc))

    scheduler.step(epoch)

    row = {'epoch': str(epoch), 'train_acc': str(accuracy), 'test_acc': str(test_acc)}
    csv_logger.writerow(row)


torch.save(cnn.state_dict(), 'checkpoints/' + test_id + '.pt')
csv_logger.close()