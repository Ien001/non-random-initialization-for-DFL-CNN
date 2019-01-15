import torch
import time
import sys
from utils.util import *
from utils.save import *
from torch.autograd import Variable


def validate_new(args, val_loader, model, criterion, epoch):
    print('begin validate!')
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    log = Log()
    model.eval()   
    end = time.time()

    total_output= []
    total_label = []
    start_test = True

    # we may have ten d in data
    for i, (data, target, paths) in enumerate(val_loader):
        if i % 1000 == 0:
            print(i)
        if args.gpu is not None:
            data = Variable(data.cuda().reshape(data.size(0) * 10, 3, 448, 448)) # bs*10, 3, 448, 448
            #print('data.shape', data.shape)
            target = Variable(target.cuda())
            #print(target.shape)
            target = target.resize(int(data.size(0)/10), 1).expand(int(data.size(0)/10),10).resize(data.size(0))

        output1, output2, output3, _ = model(data)
        output = output1 + output2 + 0.1 * output3

        if start_test:
           total_output = output.data.float()
           total_label = target.data.float()
           start_test = False
        else:
           total_output = torch.cat((total_output, output.data.float()) , 0)
           total_label = torch.cat((total_label , target.data.float()) , 0)
        
    _,predict = torch.max(total_output,1)

    acc = torch.sum(torch.squeeze(predict).float() == total_label).item() / float(total_label.size()[0])
    print(' test acc == ' + str(acc))
    return acc


def validate_simple_ian(args, val_loader, model):    
    model.eval()   
    for i, (data, target, paths) in enumerate(val_loader):
        if args.gpu is not None:
            data = Variable(data.cuda())
            target = Variable(target.cuda())

        result = torch.zeros(2000)
        for idx, d in enumerate(data):      # data [batchsize, 10_crop, 3, 448, 448]
            d = d.unsqueeze(0) # d [1, 3, 448, 448]
            center = model(d)
            center = center.expand(10)
            empty[i*10 : i*10 + 10] = center 

    return result


def validate(args, val_loader, model, criterion, epoch):
    print('begin validate!')
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    log = Log()
    model.eval()   
    end = time.time()

    total_output= []
    total_label = []
    start_test = True
    # we may have ten d in data
    for i, (data, target, paths) in enumerate(val_loader):

        if args.gpu is not None:
            data = Variable(data.cuda())
            if data.dim() == 4:
                data = data.unsqueeze(0)
                #print('data',data.shape)
            target = Variable(target.cuda())

        # compute output
        for idx, d in enumerate(data[0]):      # data [batchsize, 10_crop, 3, 448, 448]
            d = d.unsqueeze(0) # d [1, 3, 448, 448]
            output1, output2, output3, _ = model(d)
            output = output1 + output2 + 0.1 * output3

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            top1.update(prec1[0], 1)
            top5.update(prec5[0], 1)
            if i % 1000 == 0:
                print('DFL-CNN <==> Test <==> Img:{} No:{} Top1 {:.3f} Top5 {:.3f}'.format(i, idx, prec1.cpu().numpy()[0], prec5.cpu().numpy()[0]))
        
        if epoch == 0:
            break

    print('DFL-CNN <==> Test Total <==> Top1 {:.3f}% Top5 {:.3f}%'.format(top1.avg, top5.avg))
    log.save_test_info(epoch, top1, top5)
    return top1.avg


def validate_simple(args, val_loader, model, criterion, epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    log = Log()
    model.eval()   
    end = time.time()

    # we may have ten d in data
    for i, (data, target, paths) in enumerate(val_loader):
        if args.gpu is not None:
            data = Variable(data.cuda())
            target = Variable(target.cuda())

        # compute output
        for idx, d in enumerate(data):      # data [batchsize, 10_crop, 3, 448, 448]
            d = d.unsqueeze(0) # d [1, 3, 448, 448]

            output1, output2, output3, _ = model(d)
            output = output1 + output2 + 0.1 * output3

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output, target, topk=(1, 5))

            top1.update(prec1[0], 1)
            top5.update(prec5[0], 1)
            if i % 1000 == 0:
                print('DFL-CNN <==> Test <==> Img:{} Top1 {:.3f} Top5 {:.3f}'.format(i, prec1.cpu().numpy()[0], prec5.cpu().numpy()[0]))

    print('DFL-CNN <==> Test Total <==> Top1 {:.3f}% Top5 {:.3f}%'.format(top1.avg, top5.avg))
    log.save_test_info(epoch, top1, top5)
    return top1.avg