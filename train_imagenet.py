import os
import time
import shutil
import numpy as np
import configparser
import zlib


import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models


_CFG_NAME = 'train_cfg.ini'
config = configparser.ConfigParser()
config.read(_CFG_NAME)

best_prec1 = 0

device = config['device_conf']['train_device']


def save_checkpoint(state, is_best, epoch):
    out_file_dir = config['checkpoint_save_conf']['checkpoint_save_path']
    if not os.path.exists(out_file_dir):
        os.makedirs(out_file_dir)
    out_file_name = "{}_checkpoint.pth.tar".format(epoch)
    out_file_path = os.path.join(out_file_dir, out_file_name)
    torch.save(state, out_file_path)
    best_file_path = os.path.join(out_file_dir, 'model_best.pth.tar')
    if is_best:
        shutil.copyfile(out_file_path, best_file_path)



class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def main():
    global best_prec1
    
    model_name = config['model_conf']['model_name']
    using_pretrained = config['model_conf']['pretrained']
    if using_pretrained == 'True':
        using_pretrained = True
    else:
        using_pretrained = False
    if using_pretrained:
        print ("Using pretrained model")
        model = models.__dict__[model_name](pretrained=using_pretrained)
    else:
        print ("We are training from scratch")
        model = models.__dict__[model_name]()
    if config['device_conf']['train_device'] == 'gpu':
        model = model.to(device)
    # loss function 
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=float(config['train_conf']['learning_rate']),
                                momentum=float(config['train_conf']['momentum']),
                                weight_decay=float(config['train_conf']['weight_decay']))
    start_epoch = 0
    if config['checkpoint_load_conf']['checkpoint_load'] == 'True':
        load_file_path = config['checkpoint_load_conf']['checkpoint_to_load']
        if os.path.exists(load_file_path):
            print ("Loading from checkpoint {}".format(load_file_path))
            checkpoint = torch.load(load_file_path)
            start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print ("Loaded checkpoint {} from epoch {}".format(load_file_path,
                                                               start_epoch))
        else:
            print ("File {} doesn't exist")



    cudnn.benchmark = True
    train_dir = config['data_path']['train_dir']
    val_dir = config['data_path']['val_dir']
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_dataset = datasets.ImageFolder( train_dir, transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ]))
    train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=int(config['train_conf']['batch_size']),
        shuffle=True, num_workers=int(config['data_path']['num_workers']),
        pin_memory=True, sampler=None)
    val_dataset = datasets.ImageFolder(val_dir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=int(config['train_conf']['batch_size']),
        shuffle=False, num_workers=int(config['data_path']['num_workers']),
        pin_memory=True)
    num_epochs = int(config['train_conf']['num_epochs'])
    for epoch in range(start_epoch, num_epochs):
        train(train_loader, model, criterion, optimizer, epoch)
        prec1 = validate(val_loader, model, criterion)
        print ("Best prec1 after epoch {} is {}".format(epoch, prec1))
        # Saving is a, first we save the best precision model,
        # second we save at every specified epoch
        is_best = prec1 > best_prec1
        best_prec1 = max(best_prec1, prec1)
        save_checkpoint({ 'epoch' : epoch + 1,
                         'arch' : model_name,
                         'state_dict': model.state_dict(),
                         'best_prec1': best_prec1,
                         'optimizer' : optimizer.state_dict()
                         },is_best, epoch)

def do_something_grad(grad_data, layer_count):
    # import ipdb; ipdb.set_trace()
    print (grad_data.shape)
    grad_data_raster = grad_data.view(-1)
    grad_data_numpy = grad_data_raster.numpy()
    grad_data_numpy = np.round(grad_data_numpy, decimals=5)
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

def compress_grad_single(giant_numpy_array):
    print ("Giant array {}".format(giant_numpy_array.shape))
    giant_numpy_array = np.round(giant_numpy_array, decimals=5)
    giant_string = giant_numpy_array.tostring()
    giant_string_compressed = zlib.compress(giant_string, level=9)
    compression_ratio = len(giant_string)/float(len(giant_string_compressed))
    print ("Compression Ratio giant string {}".format(compression_ratio))

def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.train()
    
    end = time.time()
    for i, (input_img, target) in enumerate(train_loader):
        data_time.update(time.time() - end)
        if config['device_conf']['train_device'] == 'gpu':
            input_img =  input_img.to(device)
            target = target.to(device)
        output = model(input_img)
        loss = criterion(output, target)
        prec1, prec5 = accuracy(output, target, topk=(1,5))
        losses.update(loss.item(), input_img.size(0))
        top1.update(prec1[0], input_img.size(0))
        top5.update(prec5[0], input_img.size(0))
        optimizer.zero_grad()
        loss.backward()
        layer_count = 0
        single_list = list()
        # import ipdb; ipdb.set_trace()
        for param in model.parameters():
            grad_val = param.grad.data
            grad_val = grad_val.view(-1)
            grad_val = grad_val.numpy()
            single_list.append(grad_val)
        final_numpy_array = np.concatenate(single_list, axis=None)
        compress_grad_single(final_numpy_array)
        
        for param in model.parameters():
            # import ipdb; ipdb.set_trace()
            temp_mod = do_something_grad(param.grad.data, layer_count)
            param.grad.data = temp_mod


        optimizer.step()
        batch_time.update(time.time() - end)
        end = time.time()

        if i % int(config['print_conf']['print_freq']) == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      data_time=data_time, loss=losses, top1=top1, top5=top5))


def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input_img, target) in enumerate(val_loader):
            if config['device_conf']['val_device'] == 'gpu':
                input_img = input_img.to(device)
                target = target.to(device)
            output = model(input_img)
            loss = criterion(output, target)
            prec1, prec5 = accuracy(output, target, topk=(1,5))
            losses.update(loss.item(), input_img.size(0))
            top1.update(prec1[0], input_img.size(0))
            top5.update(prec5[0], input_img.size(0))

            batch_time.update(time.time() - end)
            end = time.time()

            if i % int(config['print_conf']['print_freq']) == 0:
                print('Test: [{0}/{1}]\n'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\n'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       i, len(val_loader), batch_time=batch_time, loss=losses,
                       top1=top1, top5=top5))
        print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
                            .format(top1=top1, top5=top5))
    return top1.avg

if __name__ == '__main__':
    main()
