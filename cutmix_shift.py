# run train.py --dataset cifar10 --model resnet18 --data_augmentation --cutout --length 16
# run train.py --dataset cifar100 --model resnet18 --data_augmentation --cutout --length 8
# run train.py --dataset svhn --model wideresnet --learning_rate 0.01 --epochs 160 --cutout --length 20

import os
import pdb
import argparse
import numpy as np
import random
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
#from util.cutout import Cutout
#from util.cutshift import Cutshift, save_checkpoint

from model.resnet import ResNet18, ResNet50
from model.wide_resnet import WideResNet
from tensorboardX import SummaryWriter

model_options = ['resnet18', 'resnet50', 'wideresnet']
dataset_options = ['cifar10', 'cifar100', 'svhn']

parser = argparse.ArgumentParser(description='CNN')
parser.add_argument('--dataset', '-d', default='cifar100',
                    choices=dataset_options)
parser.add_argument('--model', '-a', default='resnet18',
                    choices=model_options)
parser.add_argument('--batch_size', type=int, default=128,
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=150,
                    help='number of epochs to train (default: 20)')
parser.add_argument('--learning_rate', type=float, default=0.001,
                    help='learning rate')
parser.add_argument('--data_augmentation', action='store_true', default=True,
                    help='augment data by flipping and cropping')
parser.add_argument('--cutout', action='store_true', default=False,
                    help='apply cutout')
parser.add_argument('--cutshift', action='store_true', default=False,
                    help='apply cutshift')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=0,
                    help='random seed (default: 1)')
parser.add_argument('--resume', type=bool, default=False,
                    help='restart')
parser.add_argument('--lr_decay', default=0.5, type=float,
                    help='learning decay for lr scheduler')
parser.add_argument('--beta', default=1.0, type=float,
                    help='hyperparameter beta')
parser.add_argument('--cutmix_prob', default=1.0, type=float,
                    help='cutmix probability')

grad_clip = 10.0
weight_decay = 1e-4

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
cudnn.benchmark = True  # Should make training should go faster for large models

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

if not os.path.exists('logs'):
    os.makedirs('logs')
# save model
if not os.path.exists('checkpoints'):
    os.makedirs('checkpoints')
# save tensorboard file
if not os.path.exists('runs'):
    os.makedirs('runs')


test_id = str(args.dataset) + '_' + str(args.model)
#if args.cutout:
#    test_id = test_id + '_' + 'cutout(' + str(args.n_holes) + ', ' + str(args.length) + ')'
if args.cutmix_prob > 0:
    test_id = test_id + '_' + 'cutmix'
if args.cutshift:
    test_id = test_id + '_' + 'cutshift'


print(args)



# Image Preprocessing
if args.dataset == 'svhn':
    normalize = transforms.Normalize(mean=[x / 255.0 for x in[109.9, 109.7, 113.8]],
                                     std=[x / 255.0 for x in [50.1, 50.6, 50.8]])
else:
    normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                     std=[x / 255.0 for x in [63.0, 62.1, 66.7]])

train_transform = transforms.Compose([])
if args.data_augmentation:
    train_transform.transforms.append(transforms.RandomCrop(32, padding=4))
    train_transform.transforms.append(transforms.RandomHorizontalFlip())
train_transform.transforms.append(transforms.ToTensor())
train_transform.transforms.append(normalize)
#if args.cutout:
#    train_transform.transforms.append(Cutout(n_holes=args.n_holes, length=args.length))
#if args.cutshift:
#     train_transform.transforms.append(Cutshift(n_holes=args.n_holes, length=args.length, coordinate=args.coordinate))

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

if args.model == 'resnet18':
    cnn = ResNet18(num_classes=num_classes)
elif args.model == 'resnet50':
    cnn = ResNet50(num_classes=num_classes)
elif args.model == 'wideresnet':
    if args.dataset == 'svhn':
        cnn = WideResNet(depth=16, num_classes=num_classes, widen_factor=8,
                         dropRate=0.4)
    else:
        cnn = WideResNet(depth=28, num_classes=num_classes, widen_factor=10,
                         dropRate=0.3)

if args.resume:
    #path = args.model_path + test_id + '.pt'
    checkpoint = torch.load('checkpoints/' + test_id + '.pt')
    cnn.load_state_dict(checkpoint)

cnn = cnn.cuda()
criterion = nn.CrossEntropyLoss().cuda()
cnn_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, cnn.parameters()), lr=args.learning_rate)
lr_step_scheduler = torch.optim.lr_scheduler.StepLR(cnn_optimizer, step_size=20, gamma=0.1) # Decay LR by a factor of 0.1 every step_size
torch.nn.utils.clip_grad_norm_(cnn.parameters(), 0) # Gradient Clipping

if args.dataset == 'svhn':
    scheduler = MultiStepLR(cnn_optimizer, milestones=[80, 120], gamma=0.1)
else:
    scheduler = MultiStepLR(cnn_optimizer, milestones=[60, 120, 160], gamma=0.2)

filename = 'logs/' + test_id + '.csv'
csv_logger = CSVLogger(args=args, fieldnames=['epoch', 'train_acc', 'test_acc'], filename=filename)


def test(loader, criterion):
    cnn.eval()    # Change model to 'eval' mode (BN uses moving mean/var).
    correct = 0.
    total = 0.
    for images, labels in loader:
        images = images.cuda()
        labels = labels.cuda()

        with torch.no_grad():
            pred = cnn(images)

        val_loss = criterion(pred, labels)

        pred = torch.max(pred.data, 1)[1]
        total += labels.size(0)
        correct += (pred == labels).sum().item()

    val_acc = correct / total
    cnn.train()
    return val_acc, val_loss

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

def Cutshift(img, lam2, bx1, by1, bx2, by2, length):
    img = img.cpu()
    temp = img.clone()
    
    batch = img.size(0)
    c = img.size(1)
    h = img.size(2)

    w = img.size(3)

    mask = np.ones((batch, c, h, w), np.float32)
    
    cut_rat = np.sqrt(1. - lam2)
    if np.int(w * cut_rat) != 0 and np.int(h * cut_rat) != 0:
        cut_w = random.randrange(np.int(w * cut_rat))
        cut_h = random.randrange(np.int(h * cut_rat))
    else :
        cut_w = np.int(w * cut_rat)
        cut_h = np.int(h * cut_rat)
    
    cx = np.random.randint(w)
    cy = np.random.randint(h)
    
    x1 = np.clip(cx - cut_w // 2, 0, w)
    y1 = np.clip(cy - cut_h // 2, 0, h)
    x2 = np.clip(cx + cut_w // 2, 0, w)
    y2 = np.clip(cy + cut_h // 2, 0, h)
    
    x_length = x2 - x1
    y_length = y2 - y1

    # x_coor = random.randint(-length, length)
    # y_coor = random.randint(-length, length)
        
    x_coor = (int)((random.randrange(2) - 0.5) * 2) * length
    y_coor = (int)((random.randrange(2) - 0.5) * 2) * length

    x11 = np.clip(x1 + x_coor, 0, w)
    y11 = np.clip(y1 + y_coor, 0, h)
    x22 = np.clip(x2 + x_coor, 0, w)
    y22 = np.clip(y2 + y_coor, 0, h)
    
    x_length2 = x22 - x11
    y_length2 = y22 - y11
    
    # 옮기기
    if x_length2 != x_length:
        temp_ = x_length - x_length2
        if x_coor < 0 :
            x22 = x22 + temp_
        else :
            x11 = x11 - temp_
            
    if y_length2 != y_length:
        temp_ = y_length - y_length2
        if y_coor < 0 :
            y22 = y22 + temp_
        else :
            y11 = y11 - temp_
            
    mask[:, :,  y11:y22, x11:x22] = temp[:, :, y1:y2, x1:x2]
    img[:, :,  y11:y22, x11:x22] = 1.0
    #img[:, :, y11:y22, x11:x22] = 0.0

    mask = torch.from_numpy(mask)
    img = img * mask
    return img, x1, y1, x2, y2, x11, y11, x22, y22

writer = SummaryWriter('new')
for epoch in range(args.epochs):

    xentropy_loss_avg = 0.
    correct = 0.
    total = 0.
    loss = 0.

    progress_bar = tqdm(train_loader)
    for i, (input, target) in enumerate(progress_bar):
        progress_bar.set_description('Epoch ' + str(epoch))

        input = input.cuda()
        target = target.cuda()

        cnn.zero_grad()

        r = np.random.rand(1)
        if args.beta > 0 and r < args.cutmix_prob:
            # generate mixed sample
            lam = np.random.beta(args.beta, args.beta)
            lam2 = np.random.beta(args.beta, args.beta)
            rand_index = torch.randperm(input.size()[0]).cuda()
            target_a = target
            target_b = target[rand_index]
            bbx1, bby1, bbx2, bby2 = rand_bbox(input.size(), lam)
            input[:, :, bbx1:bbx2, bby1:bby2] = input[rand_index, :, bbx1:bbx2, bby1:bby2]

            # lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (input.size()[-1] * input.size()[-2]))

            # adjust lambda to exactly match pixel ratio

            if args.cutshift :
                img, x1, y1, x2, y2, x11, y11, x22, y22 = Cutshift(input, lam2, bbx1, bbx2, bby1, bbx2, 5)
                
                img = img.cuda()
                input = img

                BBox_Area = (bbx2 - bbx1) * (bby2 - bby1)

                RB_x1 = max(bbx1, x1)
                RB_x2 = min(bbx2, x2)
                RB_y1 = max(bby1, y1)
                RB_y2 = min(bby2, y2)
                RB_w = RB_x2 - RB_x1
                RB_h = RB_y2 - RB_y1

                GB_x1 = max(bbx1, x11)
                GB_x2 = min(bbx2, x22)
                GB_y1 = max(bby1, y11)
                GB_y2 = min(bby2, y22)
                GB_w = GB_x2 - GB_x1
                GB_h = GB_y1 - GB_y2

                if RB_w > 0 and RB_h > 0 :
                    BBox_Area += RB_w * RB_h
                if GB_w > 0 and GB_h > 0 :
                    BBox_Area -= GB_w * GB_h

                lam = 1 - (BBox_Area / (input.size()[-1] * input.size()[-2]))
            else :
                lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (input.size()[-1] * input.size()[-2]))
            # compute output
            input_var = torch.autograd.Variable(input, requires_grad=True)
            target_a_var = torch.autograd.Variable(target_a)
            target_b_var = torch.autograd.Variable(target_b)
            output = cnn(input_var)
            loss = criterion(output, target_a_var) * lam + criterion(output, target_b_var) * (1. - lam)
        
        else:
            # compute output
            input_var = torch.autograd.Variable(input, requires_grad=True)
            target_var = torch.autograd.Variable(target)
            output = cnn(input_var)
            loss = criterion(output, target_var)
        
        #pred = cnn(images)

        #xentropy_loss = criterion(pred, labels)
        loss.backward()
        cnn_optimizer.step()

        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(cnn.parameters(), grad_clip)

        xentropy_loss_avg += loss.item()

        # Calculate running average of accuracy
        pred = torch.max(output.data, 1)[1]
        total += target.size(0)
        correct += (pred == target.data).sum().item()
        accuracy = correct / total

        progress_bar.set_postfix(
            xentropy='%.3f' % (xentropy_loss_avg / (i + 1)),
            acc='%.3f' % accuracy)

    grid = make_grid(input_var, nrow=3, padding=5, normalize=True)
    writer.add_image('cutmix/images3', grid, 0)

    test_acc, test_loss = test(test_loader, criterion)
    tqdm.write('test_loss: %.3f' % (test_loss))
    tqdm.write('test_acc: %.3f' % (test_acc))

    scheduler.step(epoch)

    row = {'epoch': str(epoch), 'train_acc': str(accuracy), 'test_acc': str(test_acc)}
    csv_logger.writerow(row)


torch.save(cnn.state_dict(), 'checkpoints/' + test_id + '.pt')
csv_logger.close()
writer.close()

