import os
import sys
import time
import glob
import numpy as np
import torch
import logging
import argparse
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
from time import time
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
import configparser
from model_search import Network
from utils import *
from utils import _data_transforms_cifar10
from architect import *


# args.save = 'search-{}-{}'.format(args.saved_path, time.strftime("%Y%m%d-%H%M%S"))
# create_exp_dir(args.saved_path, scripts_to_save=glob.glob('*.py'))

args = configparser.ConfigParser()
log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.saved_path, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

PATH = ''
args.data_dir = os.path.join(PATH, "DART")
args.saved_model = os.path.join(PATH, "DART")
args.saved_path = os.path.join(PATH, "DART")
os.makedirs(args.data_dir, exist_ok=True)

args.data = '../data'  # 'location of the data corpus'
args.batch_size = 64  # 'batch size'
args.learning_rate = 0.025  # 'init learning rate'
args.learning_rate_min = 0.001  # 'min learning rate'
args.momentum = 0.9  # 'momentum'
args.weight_decay = 3e-4  # 'weight decay'
args.report_freq = 50  # 'report frequency'
args.gpu = 0  # 'gpu device id'
args.epochs = 50  # 'num of training epochs'
args.init_channels = 16  # 'num of init channels'
args.layers = 8  # 'total number of layers'
args.model_path = 'saved_models'  # 'path to save the model'
args.cutout = False  # 'use cutout'
args.cutout_length = 16  # 'cutout length'
args.drop_path_prob = 0.3  # 'drop path probability'
args.save = PATH + 'EXP'  # 'experiment name'
os.makedirs(args.save, exist_ok=True)
args.seed = 2  # 'random seed'
args.grad_clip = 5  # 'gradient clipping'
args.train_portion = 0.5  # 'portion of training data'
args.unrolled = False  # 'use one-step unrolled validation loss'
args.arch_learning_rate = 3e-4  # 'learning rate for arch encoding'
args.arch_weight_decay = 1e-3  # 'weight decay for arch encoding'

CIFAR_CLASSES = 10

writer = SummaryWriter(os.path.join(PATH, 'runs/DARTS'))

def main():
    if not torch.cuda.is_available():
        print('no gpu device available')
        sys.exit(1)

    np.random.seed(args.seed)
    torch.cuda.set_device(args.gpu)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)
    print('gpu device = ', args.gpu)
    # print("args = ", args)

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    model = Network(args.init_channels, CIFAR_CLASSES, args.layers, criterion)
    model = model.cuda()
    optimizer = torch.optim.SGD(
        model.parameters(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, float(args.epochs), eta_min=args.learning_rate_min
    )
    try:
        checkpoint = torch.load(os.path.join(args.saved_model, 'weights.pt'))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        epoch_old = checkpoint['epoch']
        print('Load previous model at : ', args.saved_model)
    except:
        epoch_old = 0
        print('Training new model!')
    print("param size = MB", count_parameters_in_MB(model))

    train_transform, valid_transform = _data_transforms_cifar10(args)
    train_data = dset.CIFAR10(root=args.data_dir, train=True, download=True, transform=train_transform)

    num_train = len(train_data)
    indices = list(range(num_train))
    split = int(np.floor(args.train_portion * num_train))

    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
        pin_memory=True, num_workers=0)

    valid_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
        pin_memory=True, num_workers=0)

    architect = Architect(model, args)

    for epoch in range(epoch_old, args.epochs):
        lr = scheduler.get_last_lr()[0]
        writer.add_scalar('learning rate', lr)
        print('epoch: ', str(epoch), ', lr: ', str(lr))

        genotype = model.genotype()
        print('genotype = ', genotype)

        print(F.softmax(model.alphas_normal, dim=-1))
        print(F.softmax(model.alphas_reduce, dim=-1))

        # training
        start_train = time()
        train_acc, train_obj = train(train_queue, valid_queue, model, architect, criterion, optimizer, lr)
        end_train = time()

        print(f'train_acc: {train_acc.item()}%, train_time: {end_train - start_train}s')
        writer.add_scalar('train accuracy', train_acc.item(), epoch)
        writer.add_scalar('train loss', train_obj.item(), epoch)
        writer.add_scalar('train time', end_train - start_train, epoch)
        scheduler.step()

        # validation
        start_valid = time()
        valid_acc, valid_obj = infer(valid_queue, model, criterion)
        end_valid = time()

        print(f'valid_acc: {valid_acc.item()}%, valid_time: {end_valid - start_valid}s')
        writer.add_scalar('valid accuracy', valid_acc.item(), epoch)
        writer.add_scalar('valid loss', valid_obj.item(), epoch)
        writer.add_scalar('valid time', end_valid - start_valid, epoch)
        save(model, epoch, optimizer, scheduler, os.path.join(args.saved_model, 'weights.pt'))


def train(train_queue, valid_queue, model, architect, criterion, optimizer, lr):
    objs = AvgrageMeter()
    top1 = AvgrageMeter()
    top5 = AvgrageMeter()

    for step, (input, target) in enumerate(train_queue):
        model.train()
        n = input.size(0)

        input = Variable(input, requires_grad=False).cuda()
        target = Variable(target, requires_grad=False).cuda(non_blocking=True)

        # get a random minibatch from the search queue with replacement
        input_search, target_search = next(iter(valid_queue))
        input_search = Variable(input_search, requires_grad=False).cuda()
        target_search = Variable(target_search, requires_grad=False).cuda(non_blocking=True)

        architect.step(input, target, input_search, target_search, lr, optimizer, unrolled=args.unrolled)

        optimizer.zero_grad()
        logits = model(input)
        loss = criterion(logits, target)

        loss.backward()
        optimizer.step()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

        prec1, prec5 = accuracy(logits, target, topk=(1, 5))
        objs.update(loss.data, n)
        top1.update(prec1.data, n)
        top5.update(prec5.data, n)

        if step % args.report_freq == 0:
            print(f'train {step}: loss = {objs.avg.item()}, top 1 = {top1.avg.item()}%, top 5 = {top5.avg.item()}%')

    return top1.avg, objs.avg


def infer(valid_queue, model, criterion):
    objs = AvgrageMeter()
    top1 = AvgrageMeter()
    top5 = AvgrageMeter()
    model.eval()

    with torch.no_grad():
        for step, (input, target) in enumerate(valid_queue):
            input = Variable(input).cuda()
            target = Variable(target).cuda(non_blocking=True)

            logits = model(input)
            loss = criterion(logits, target)

            prec1, prec5 = accuracy(logits, target, topk=(1, 5))
            n = input.size(0)
            objs.update(loss.data, n)
            top1.update(prec1.data, n)
            top5.update(prec5.data, n)

            if step % args.report_freq == 0:
                print(f'valid {step}: loss = {objs.avg.item()}, top 1 = {top1.avg.item()}%, top 5 = {top5.avg.item()}%')

    return top1.avg, objs.avg


if __name__ == '__main__':
    main()
