import argparse
import os
import time
import torch

from torch.utils.data import DataLoader
import torch.optim
import torchvision.transforms as transforms
from torchvision.utils import make_grid
from torch import nn

from tensorboardX import SummaryWriter

import RedNet_model
import RedNet_data
from utils import utils
from utils.utils import save_ckpt
from utils.utils import load_ckpt
from utils.utils import print_log
from torch.optim.lr_scheduler import LambdaLR

parser = argparse.ArgumentParser(description='RedNet Indoor Sementic Segmentation')
parser.add_argument('--data-dir', default=None, metavar='DIR',
                    help='path to SUNRGB-D')
parser.add_argument('--cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')
parser.add_argument('--epochs', default=1500, type=int, metavar='N',
                    help='number of total epochs to run (default: 1500)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=5, type=int,
                    metavar='N', help='mini-batch size (default: 10)')
parser.add_argument('--lr', '--learning-rate', default=2e-3, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--print-freq', '-p', default=200, type=int,
                    metavar='N', help='print batch frequency (default: 50)')
parser.add_argument('--save-epoch-freq', '-s', default=5, type=int,
                    metavar='N', help='save epoch frequency (default: 5)')
parser.add_argument('--last-ckpt', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--lr-decay-rate', default=0.8, type=float,
                    help='decay rate of learning rate (default: 0.8)')
parser.add_argument('--lr-epoch-per-decay', default=100, type=int,
                    help='epoch of per decay of learning rate (default: 150)')
parser.add_argument('--ckpt-dir', default='./model/', metavar='DIR',
                    help='path to save checkpoints')
parser.add_argument('--summary-dir', default='./summary', metavar='DIR',
                    help='path to save summary')
parser.add_argument('--checkpoint', action='store_true', default=False,
                    help='Using Pytorch checkpoint or not')

args = parser.parse_args()
device = torch.device("cuda:0" if args.cuda and torch.cuda.is_available() else "cpu")
image_w = 640
image_h = 480
def train():
    train_data = RedNet_data.SUNRGBD(transform=transforms.Compose([RedNet_data.scaleNorm(),
                                                                   RedNet_data.RandomScale((1.0, 1.4)),
                                                                   RedNet_data.RandomHSV((0.9, 1.1),
                                                                                         (0.9, 1.1),
                                                                                         (25, 25)),
                                                                   RedNet_data.RandomCrop(image_h, image_w),
                                                                   RedNet_data.RandomFlip(),
                                                                   RedNet_data.ToTensor(),
                                                                   RedNet_data.Normalize()]),
                                     phase_train=True,
                                     data_dir=args.data_dir)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=False)

    num_train = len(train_data)

    if args.last_ckpt:
        model = RedNet_model.RedNet(pretrained=False)
    else:
        model = RedNet_model.RedNet(pretrained=True)
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    CEL_weighted = utils.CrossEntropyLoss2d()
    model.train()
    model.to(device)
    CEL_weighted.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                momentum=args.momentum, weight_decay=args.weight_decay)

    global_step = 0

    if args.last_ckpt:
        global_step, args.start_epoch = load_ckpt(model, optimizer, args.last_ckpt, device)

    lr_decay_lambda = lambda epoch: args.lr_decay_rate ** (epoch // args.lr_epoch_per_decay)
    scheduler = LambdaLR(optimizer, lr_lambda=lr_decay_lambda)

    writer = SummaryWriter(args.summary_dir)

    for epoch in range(int(args.start_epoch), args.epochs):

        scheduler.step(epoch)
        local_count = 0
        last_count = 0
        end_time = time.time()
        if epoch % args.save_epoch_freq == 0 and epoch != args.start_epoch:
            save_ckpt(args.ckpt_dir, model, optimizer, global_step, epoch,
                      local_count, num_train)

        for batch_idx, sample in enumerate(train_loader):

            image = sample['image'].to(device)
            depth = sample['depth'].to(device)
            target_scales = [sample[s].to(device) for s in ['label', 'label2', 'label3', 'label4', 'label5']]
            optimizer.zero_grad()
            pred_scales = model(image, depth, args.checkpoint)
            loss = CEL_weighted(pred_scales, target_scales)
            loss.backward()
            optimizer.step()
            local_count += image.data.shape[0]
            global_step += 1
            if global_step % args.print_freq == 0 or global_step == 1:

                time_inter = time.time() - end_time
                count_inter = local_count - last_count
                print_log(global_step, epoch, local_count, count_inter,
                          num_train, loss, time_inter)
                end_time = time.time()

                for name, param in model.named_parameters():
                    writer.add_histogram(name, param.clone().cpu().data.numpy(), global_step, bins='doane')
                grid_image = make_grid(image[:3].clone().cpu().data, 3, normalize=True)
                writer.add_image('image', grid_image, global_step)
                grid_image = make_grid(depth[:3].clone().cpu().data, 3, normalize=True)
                writer.add_image('depth', grid_image, global_step)
                grid_image = make_grid(utils.color_label(torch.max(pred_scales[0][:3], 1)[1] + 1), 3, normalize=False,
                                       range=(0, 255))
                writer.add_image('Predicted label', grid_image, global_step)
                grid_image = make_grid(utils.color_label(target_scales[0][:3]), 3, normalize=False, range=(0, 255))
                writer.add_image('Groundtruth label', grid_image, global_step)
                writer.add_scalar('CrossEntropyLoss', loss.data, global_step=global_step)
                writer.add_scalar('Learning rate', scheduler.get_lr()[0], global_step=global_step)
                last_count = local_count

    save_ckpt(args.ckpt_dir, model, optimizer, global_step, args.epochs,
              0, num_train)

    print("Training completed ")

if __name__ == '__main__':
    if not os.path.exists(args.ckpt_dir):
        os.mkdir(args.ckpt_dir)
    if not os.path.exists(args.summary_dir):
        os.mkdir(args.summary_dir)

    train()
