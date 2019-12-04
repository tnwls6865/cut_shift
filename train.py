import os
import pdb
import argparse
import numpy as np
import warnings
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import MultiStepLR

import torchvision
from torchvision.utils import make_grid
from torchvision import datasets, transforms

from utils import CSVLogger, rand_bbox, save_checkpoint

from model.resnet import ResNet18, ResNet50, ResNet34
from model.wide_resnet import WideResNet
from tensorboardX import SummaryWriter


warnings.filterwarnings("ignore")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_options = ['resnet18', 'resnet34', 'resnet50', 'wideresnet']
dataset_options = ['cifar10', 'cifar100', 'svhn', 'fashion_mnist']

parser = argparse.ArgumentParser(description='CNN')
parser.add_argument('--dataset', '-d', default='cifar10',
                    choices=dataset_options)
parser.add_argument('--model_type', '-a', default='resnet18',
                    choices=model_options)
parser.add_argument('--batch_size', type=int, default=64,
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=260,
                    help='number of epochs to train (default: 20)')
parser.add_argument('--lr', type=float, default=0.25,
                    help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9,  metavar='M',
                    help='momentum')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed (default: 1)')
parser.add_argument('--resume', type=bool, default=False,
                    help='restart')
parser.add_argument('--lr_decay', type=float, default=1e-4,
                     help='learning decay for lr scheduler')
parser.add_argument('--featuremix', action='store_true', default=True,
                     help='apply featuremix')
parser.add_argument('--beta', default=1.0, type=float,
                     help='beta distribution')
parser.add_argument('--mix_prob', default=1.0, type=float,
                     help='learning decay for lr scheduler')


def main():
    global args, lam
    
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    if device == 'cuda':
        torch.cuda.manual_seed(args.seed)

    # save file
    csv_path = 'logs/'
    csv_path = os.path.join(csv_path, 'batch_' + str(args.batch_size))
    print(csv_path)
    model_path = 'checkpoints/'
    tensorboard_path = 'runs/'
    writer = SummaryWriter(tensorboard_path)
    if not os.path.exists(csv_path):
        os.makedirs(csv_path)
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    if not os.path.exists(tensorboard_path):
        os.makedirs(tensorboard_path)
    
    # setting test_id
    test_id = str(args.dataset) + '_' + str(args.model_type) 
    if args.featuremix:
        test_id = test_id + '_changechannel'
    test_id = test_id + '_lr_' + str(args.lr) 

 
    

    # setting dataset
    if args.dataset == 'svhn':
        normalize = transforms.Normalize(mean=[x / 255.0 for x in[109.9, 109.7, 113.8]],
                                        std=[x / 255.0 for x in [50.1, 50.6, 50.8]])
    elif args.dataset == 'fashion_mnist':
        normalize = transforms.Normalize(mean=(0.5,), std=(0.5,))

    else:
        normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                        std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
    
    # data augmentation
    train_transform = transforms.Compose([])
    train_transform.transforms.append(transforms.RandomCrop(32, padding=4))
    train_transform.transforms.append(transforms.RandomHorizontalFlip())
    train_transform.transforms.append(transforms.ToTensor())
    train_transform.transforms.append(normalize)

    test_transform = transforms.Compose([transforms.ToTensor(),normalize])

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
    elif args.dataset == 'fashion_mnist':
        num_classes = 10
        train_dataset = datasets.FashionMNIST(root='data/',
                                              train=True,
                                              transform=train_transform,
                                              download=True)
        # Download and load the test data
        test_dataset = datasets.FashionMNIST(root='data/',
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

    # data loader 
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
    
    # model setting 
    if args.model_type == 'resnet18':
        model = ResNet18(num_classes=num_classes)
    elif args.model_type == 'resnet34':
        model = ResNet34(num_classes=num_classes)
    elif args.model_type == 'resnet50':
        model = ResNet50(num_classes=num_classes)
    elif args.model_type == 'wideresnet':
        if args.dataset == 'svhn':
            model = WideResNet(depth=16, num_classes=num_classes, widen_factor=8,
                            dropRate=0.4)
        else:
            model = WideResNet(depth=28, num_classes=num_classes, widen_factor=10,
                            dropRate=0.3)

    # checkpoint load
    if args.resume:
        checkpoint = torch.load('checkpoints/' + test_id + '.pt')
        model.load_state_dict(checkpoint)
    
    # loss and hyperparameter
    model = model.cuda()
    criterion = nn.CrossEntropyLoss().cuda()
    #optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.lr_decay, nesterov=True)
    #lr_step_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1) # Decay LR by a factor of 0.1 every step_size
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0) # Gradient Clipping

    if args.dataset == 'svhn':
        scheduler = MultiStepLR(optimizer, milestones=[80, 120], gamma=0.1)
    else:
        scheduler = MultiStepLR(optimizer, milestones=[60, 120, 160, 200], gamma=0.2)

    filename = csv_path + '/' + test_id + '.csv'
    csv_logger = CSVLogger(args=args, fieldnames=['epoch', 'train_loss', 'train_acc', 'test_loss', 'test_acc'], filename=filename)

    for epoch in range(args.epochs):
        
        progress_bar = tqdm(train_loader)
        
        train_acc, train_loss = train(progress_bar, model, criterion, optimizer, epoch)

        test_acc, test_loss = test(test_loader, model, criterion)

        writer.add_scalar('wideresnet/train_loss', (train_loss/len(train_loader)), epoch)
        writer.add_scalar('wideresnet/train_acc', train_acc, epoch)

        writer.add_scalar('wideresnet/test_loss', test_loss, epoch)
        writer.add_scalar('wideresnet/test_acc', test_acc, epoch)

        tqdm.write('test_loss: %.3f' % (test_loss))
        tqdm.write('test_acc: %.3f' % (test_acc))

        train_loss = '%.3f' % (train_loss/len(train_loader))
        test_loss = '%.3f' % (test_loss)

        scheduler.step(epoch)

        row = {'epoch': str(epoch), 'train_loss': str(train_loss), 'train_acc': str(train_acc), 'test_loss': str(test_loss), 'test_acc': str(test_acc)}
        csv_logger.writerow(row)
    
    save_checkpoint({
        'epoch': epoch,
        'arch': args.model_type,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict()}, model_path, test_id)
    csv_logger.close()


def train(progress_bar, model, criterion, optimizer, epoch):
    xentropy_loss_avg = 0.
    correct = 0.
    total = 0.
    bbox = []

    for i, (images, labels) in enumerate(progress_bar):
        progress_bar.set_description('Epoch ' + str(epoch))

        images = images.cuda()
        labels = labels.cuda()

        r = np.random.rand(1)
        if args.beta > 0 and r < args.mix_prob:
            is_train = True
            
            lam = np.random.beta(args.beta, args.beta)  
            
            rand_index = torch.randperm(images.size()[0]).cuda()

            label_a = labels
            label_b = labels[rand_index]
            bbox = rand_bbox(images.size(), lam)


            images[:, :, bbox[0]:bbox[2], bbox[1]:bbox[3]] = images[rand_index, :, bbox[0]:bbox[2], bbox[1]:bbox[3]]
            lam = 1 - ((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) / (images.size()[-1] * images.size()[-2]))
            
            # compute output
            input_var = torch.autograd.Variable(images, requires_grad=True)
            label_a_var = torch.autograd.Variable(label_a)
            label_b_var = torch.autograd.Variable(label_b)

            output = model(input_var, is_train, rand_index, lam)
            loss = criterion(output, label_a_var) *lam + criterion(output, label_b_var) * (1. - lam)
        
        else:
            is_train = False
            input_var = torch.autograd.Variable(images, requires_grad=True)
            label_var = torch.autograd.Variable(labels)
            output = model(input_var, is_train)
            loss = criterion(output, label_var)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        xentropy_loss_avg += loss.item()

        # Calculate running average of accuracy
        output = torch.max(output.data, 1)[1]
        total += labels.size(0)
        correct += (output == labels.data).sum().item()
        accuracy = correct / total
        #train_loss = xentropy_loss_avg / total

        progress_bar.set_postfix(xentropy='%.3f' % (xentropy_loss_avg / (i + 1)), acc='%.3f' % accuracy)

    
    return accuracy, xentropy_loss_avg


def test(loader, model, criterion):
    model.eval()    # Change model to 'eval' mode (BN uses moving mean/var).
    correct = 0.
    total = 0.
    total_loss = 0.
    loss = 0.

    is_train = False
    for images, labels in loader:
        images = images.cuda()
        labels = labels.cuda()

        with torch.no_grad():
            pred = model(images, is_train, 0)

        val_loss = criterion(pred, labels)
        loss += val_loss.item()

        pred = torch.max(pred.data, 1)[1]
        total += labels.size(0)
        correct += (pred == labels).sum().item()
        
    val_acc = correct / total
    total_loss = loss / len(loader)
    
    model.train()
    return val_acc, total_loss


if __name__ == '__main__':
    main()