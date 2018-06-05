import argparse
import torch
import imageio
import skimage.transform
import torchvision

import torch.optim
import RedNet_model
from utils import utils
from utils.utils import load_ckpt

parser = argparse.ArgumentParser(description='RedNet Indoor Sementic Segmentation')
parser.add_argument('-r', '--rgb', default=None, metavar='DIR',
                    help='path to image')
parser.add_argument('-d', '--depth', default=None, metavar='DIR',
                    help='path to depth')
parser.add_argument('-o', '--output', default=None, metavar='DIR',
                    help='path to output')
parser.add_argument('--cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--last-ckpt', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')

args = parser.parse_args()
device = torch.device("cuda:0" if args.cuda and torch.cuda.is_available() else "cpu")
image_w = 640
image_h = 480
def inference():

    model = RedNet_model.RedNet(pretrained=False)
    load_ckpt(model, None, args.last_ckpt, device)
    model.eval()
    model.to(device)

    image = imageio.imread(args.rgb)
    depth = imageio.imread(args.depth)

    # Bi-linear
    image = skimage.transform.resize(image, (image_h, image_w), order=1,
                                     mode='reflect', preserve_range=True)
    # Nearest-neighbor
    depth = skimage.transform.resize(depth, (image_h, image_w), order=0,
                                     mode='reflect', preserve_range=True)

    image = image / 255
    image = torch.from_numpy(image).float()
    depth = torch.from_numpy(depth).float()
    image = image.permute(2, 0, 1)
    depth.unsqueeze_(0)

    image = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])(image)
    depth = torchvision.transforms.Normalize(mean=[19050],
                                             std=[9650])(depth)

    image = image.to(device).unsqueeze_(0)
    depth = depth.to(device).unsqueeze_(0)

    pred = model(image, depth)

    output = utils.color_label(torch.max(pred, 1)[1] + 1)[0]

    imageio.imsave(args.output, output.cpu().numpy().transpose((1, 2, 0)))

if __name__ == '__main__':
    inference()
