import shutil
import torch
import time
import argparse
from datasets import get_dataloaders
from cnn_models import get_model
from datetime import datetime
import os
from torch.utils.tensorboard import SummaryWriter
from models.utils_reduction import get_seq_model_general
from models.netadapter import NetAdapter
from utils import validate, train, save_checkpoint

torch.backends.cudnn.benchmark = True
torch.autograd.set_detect_anomaly(False)
torch.autograd.profiler.profile(False)
torch.autograd.profiler.emit_nvtx(False)


device = torch.device('cuda')

parser = argparse.ArgumentParser(description='Train a model on a dataset')
parser.add_argument('--model', type=str, required=True, choices=['vgg16', 'resnet101', 'inceptionv3'])
parser.add_argument('--dataset', type=str, required=True, choices=['cifar10', 'cifar100', 'stl10'])
parser.add_argument('--dataset_path', type=str, default='./dataset')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('--epochs', type=int, required=True)
parser.add_argument('--batch_size', type=int, required=True)

parser.add_argument('--lr', type=float, required=True)
parser.add_argument('--epsilon', default=1e-08, type=float)
parser.add_argument('--beta1', type=float, default=0.9, help='The beta1 value for AdamW optimizer')
parser.add_argument('--beta2', type=float, default=0.999, help='The beta2 value for AdamW optimizer')
parser.add_argument('--weight_decay', default=0.01, type=float)
parser.add_argument('--momentum', default=0, type=float)

parser.add_argument('--experiment_name', type=str, default='')
parser.add_argument('--print-freq', '-p', default=20, type=int, metavar='N', help='print frequency (default: 10)')
parser.add_argument('--eval-freq', '-ef', default=2, type=int, metavar='N', help='evaluation frequency (default: 5)')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
parser.add_argument('--root_log',type=str, default='log')
parser.add_argument('--iter_size', default=1, type=int, help='Number of iterations to wait before updating the weights')
parser.add_argument('--store_name', type=str, default="")
  
  
global args#, best_prec1
global best_prec1
best_prec1 = 0
args = parser.parse_args()

if args.dataset == 'cifar10':
    num_classes = 10
elif args.dataset == 'cifar100':
    num_classes = 100
elif args.dataset == 'stl10':
    num_classes = 10
else:
    raise ValueError('Unknow Dataset ' + args.dataset)
    
global model_dir

model_dir = os.path.join('experiments', args.dataset, args.model, datetime.now().strftime('%b%d_%H-%M-%S')+'_'+args.experiment_name)
os.makedirs(model_dir)
os.makedirs(os.path.join(model_dir, args.root_log))

writer = SummaryWriter(model_dir)


# Load model
model, input_size = get_model(args.model, num_classes, pretrained=True)
model = model.cuda()
model = torch.nn.DataParallel(model).cuda()
print(model.module)

#optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2), eps=args.epsilon, weight_decay=args.weight_decay)
optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

criterion = torch.nn.CrossEntropyLoss().cuda()

train_loader, test_loader, train_dataset = get_dataloaders(args.dataset, args.batch_size, input_size, args.dataset_path)

if args.resume:
    if os.path.isfile(args.resume):
        print(("=> loading checkpoint '{}'".format(args.resume)))
        checkpoint = torch.load(args.resume)
        args.start_epoch = checkpoint['epoch']
        best_prec1 = checkpoint['best_prec1']
        model.load_state_dict(checkpoint['model_state_dict'])
        print(("=> loaded checkpoint '{}' (epoch {})".format(args.evaluate, checkpoint['epoch'])))
    else:
        print(("=> no checkpoint found at '{}'".format(args.resume)))


log_training = open(os.path.join(model_dir, args.root_log, '%s.csv' % args.store_name), 'a')

if args.evaluate:
    validate(test_loader, model, criterion, best_prec1, iter=0, epoch=args.start_epoch, log=log_training, print_freq=args.print_freq, writer=writer)
    #return
    

seq_model = get_seq_model_general(model.module).to(device)
train_labels = torch.tensor(train_loader.dataset.targets)

cutoff_idx = 7 
red_dim = 50 
red_method = 'POD' 
inout_method = 'FNN'
n_class = 10
netadapter = NetAdapter(cutoff_idx, red_dim, red_method, inout_method)
red_model = netadapter.reduce_net(seq_model, train_dataset, train_labels, train_loader, num_classes)
print(red_model)

model = red_model

for epoch in range(args.start_epoch, args.epochs):
    writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch + 1)
    train_prec1 = train(train_loader, model, criterion, optimizer, epoch, log_training, writer=writer)
    scheduler.step()
        
    # evaluate on validation set
    if (epoch + 1) % args.eval_freq == 0:
        prec1 = validate(test_loader, model, criterion, (epoch + 1) * len(train_loader), log_training, writer=writer, epoch=epoch)
        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.model,
            'model_state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'current_prec1': prec1,
            'lr': optimizer.param_groups[-1]['lr'],
        }, is_best, model_dir)
    else:
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.model,
            'model_state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'current_prec1': train_prec1,
            'lr': optimizer.param_groups[-1]['lr'],
        }, False, model_dir)





