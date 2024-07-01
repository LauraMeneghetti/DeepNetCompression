import shutil
import torch
import time
import argparse
from datasets import get_dataloaders
from cnn_models import get_model
from datetime import datetime
import os



def train(train_loader, model, criterion, optimizer, epoch, log, writer):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()
    end = time.time()

    for i, (input, target) in enumerate(train_loader):
        
        # measure data loading time
        data_time.update(time.time() - end)
        
        target = target.cuda()
        input_var = input.cuda()
        
        # compute output
        if args.model == "inceptionv3":
            output, _ = model(input_var)
        else:
            output = model(input_var)
                
        loss = criterion(output, target)
        
        # measure accuracy and record loss
        prec1, prec5 = accuracy(output, target, topk=(1,5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            output = ('Epoch: [{0}][{1}/{2}], lr: {lr:.5f}\t'
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                        'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                        'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                        'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5, lr=optimizer.param_groups[-1]['lr']))
            writer.add_scalar('train/batch_loss', losses.avg, epoch * len(train_loader) + i)
            writer.add_scalar('train/batch_top1Accuracy', top1.avg, epoch * len(train_loader) + i)
            print(output)
            log.write(output + '\n')
            log.flush()
    writer.add_scalar('train/loss', losses.avg, epoch + 1)
    writer.add_scalar('train/top1Accuracy', top1.avg, epoch + 1)
    writer.add_scalar('train/top5Accuracy', top5.avg, epoch + 1)
    return top1.avg


def validate(val_loader, model, criterion, best_prec1, iter, log, epoch, print_freq, writer):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            
            target = target.cuda()
            input_var = input.cuda()

            output = model(input_var)
            loss = criterion(output, target)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output, target, topk=(1,5))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))
            top5.update(prec5.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            
            if i % print_freq == 0:
                output = ('Val: [{0}/{1}]\t'
                            'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                            'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                            'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                            'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                            i, len(val_loader), batch_time=batch_time, loss=losses, top1=top1, top5=top5))
                print(output)
                log.write(output + '\n')
                log.flush()

    output = ('Validation Results: Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f} Loss {loss.avg:.5f}'.format(top1=top1, top5=top5, loss=losses))
    print(output)
    output_best = '\nBest Prec@1: %.3f'%(best_prec1)
    print(output_best)
    writer.add_scalar('val/loss', losses.avg, epoch + 1)
    writer.add_scalar('val/top1Accuracy', top1.avg, epoch + 1)
    writer.add_scalar('val/top5Accuracy', top5.avg, epoch + 1)
    log.write(output + ' ' + output_best + '\n')
    log.flush()

    return top1.avg


def save_checkpoint(state, is_best, model_dir):
    torch.save(state, '%s/%s_checkpoint.pth.tar' % (model_dir, args.store_name))
    if is_best:
        shutil.copyfile('%s/%s_checkpoint.pth.tar' % (model_dir, args.store_name), '%s/%s_best.pth.tar' % (model_dir, args.store_name))

def save_checkpoint_epoch(state, epoch, model_dir):
    torch.save(state, '%s/%s_checkpoint_%d.pth.tar' % (model_dir, args.store_name, epoch))


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
            correct_k = correct[:k].contiguous().view(-1).float().sum(0)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res