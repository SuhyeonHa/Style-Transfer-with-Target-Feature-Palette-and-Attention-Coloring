import warnings
warnings.simplefilter("ignore", UserWarning)
import argparse
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.utils.data as data
import numpy as np
from PIL import Image, ImageFile
from torchvision import transforms
from function import calc_mean_std
from function import denorm, denorm_device, tensor2im
from torch.optim import Adam

import net
from sampler import InfiniteSamplerWrapper
import visdom

cudnn.benchmark = True
Image.MAX_IMAGE_PIXELS = None  # Disable DecompressionBombError
ImageFile.LOAD_TRUNCATED_IMAGES = True

def train_transform():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform_list = [
        transforms.Resize(size=(512, 512)),
        transforms.RandomCrop(256),
        transforms.ToTensor(),
        normalize
    ]
    return transforms.Compose(transform_list)


def test_transform():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform_list = [
        transforms.Resize(size=(512, 512)),
        #transforms.RandomCrop(256),
        transforms.ToTensor(),
        normalize
    ]
    return transforms.Compose(transform_list)

class FlatFolderDataset(data.Dataset):
    # Get all images from the root and transform
    def __init__(self, root, transform):
        super(FlatFolderDataset, self).__init__()
        self.root = root
        self.paths = list(Path(self.root).glob('*'))
        self.transform = transform

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(str(path)).convert('RGB')
        img = self.transform(img)
        return img

    def __len__(self):
        return len(self.paths)

    def name(self):
        return 'FlatFolderDataset'


def vis_patches(style, patch_index, groups):
    crop_ratio = 256//32  # 512/32
    patches = []
    group = groups[0]
    for i in range(len(group)):
        temp = []
        for elem in group[i]:  # elem is sample index
            points = patch_index[elem]
            w1 = points[0] * crop_ratio  # x
            w2 = points[1] * crop_ratio  # x + crop_width
            h1 = points[2] * crop_ratio  # y
            h2 = points[3] * crop_ratio  # y + crop_height
            patch = style[:, w1:w2, h1:h2]
            temp.append(patch)
        patches.append(temp)
    return patches

def get_args_parser():
    parser = argparse.ArgumentParser()
    # Basic options
    parser.add_argument('--train_content_dir', type=str, required=True, help='content images directory for train')  # COCO directory
    parser.add_argument('--train_style_dir', type=str, required=True, help='style images directory for train')  # PBN directory
    parser.add_argument('--test_content_dir', type=str, default='input/content',
                        help='content images directory for test')
    parser.add_argument('--test_style_dir', type=str, default='input/style',
                        help='style images directory for test')
    parser.add_argument('--vgg', type=str, default='models/vgg_normalised.pth', help='pre-trained vgg model')
    parser.add_argument('--model_dir', default='models', help='Directory to save the model')
    parser.add_argument('--output_dir', type=str, default='./output')
    parser.add_argument('--save_ext', type=str, default='.jpg',
                        help='The extension name of the output image')

    # training options
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--max_iter', type=int, default=95000)
    parser.add_argument('--style_weight', type=float, default=1.0)
    parser.add_argument('--content_weight', type=float, default=30.0)
    parser.add_argument('--save_model_interval', type=int, default=5000)
    parser.add_argument('--test_interval', type=int, default=10)
    parser.add_argument('--cluster_size', type=int, default=3)
    parser.add_argument('--patch_size', type=int, default=8)
    parser.add_argument('--num_patch', type=int, default=20)

    return parser


if __name__ == '__main__':
    args = get_args_parser().parse_args()

    device = torch.device('cuda')
    model_dir = Path(args.model_dir)
    model_dir.mkdir(exist_ok=True, parents=True)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # A pair image (content, style) displaying on Visdom
    test_tf = test_transform()
    test_content_path = Path('./input/content/sailboat.jpg')
    test_content_img = test_tf(Image.open(test_content_path))
    test_style_path = Path('./input/style/05.jpg')
    test_style_img = test_tf(Image.open(test_style_path))

    # Visdom display initialization
    vis = visdom.Visdom()
    vis.close(env="main")
    plot = vis.line(Y=torch.Tensor(1).zero_(), opts=dict(title='loss_c'))
    plot2 = vis.line(Y=torch.Tensor(1).zero_(), opts=dict(title='loss_s'))
    vis_content = vis.image(denorm(test_content_img), opts=dict(title="Content Image"))
    vis_style = vis.image(denorm(test_style_img), opts=dict(title="Style Image"))
    vis_img = vis.image(denorm(test_content_img), opts=dict(title="Stylized Image"))

    decoder = net.decoder
    encoder = net.vgg

    encoder.load_state_dict(torch.load(args.vgg))
    encoder = nn.Sequential(*list(encoder.children())[:31])
    network = net.Net(encoder, decoder)
    network.train()
    network.to(device)

    optimizer = Adam(decoder.parameters(), lr=args.lr)

    content_tf = train_transform()
    style_tf = train_transform()

    content_dataset = FlatFolderDataset(args.train_content_dir, content_tf)
    style_dataset = FlatFolderDataset(args.train_style_dir, style_tf)

    # InfiniteSamplerWrapper: get 256*256 patches from random permutation order
    # Get image using PIL
    # content_iter = iter(data.DataLoader(
    #     content_dataset, batch_size=args.batch_size,
    #     sampler=InfiniteSamplerWrapper(content_dataset),
    #     num_workers=args.n_threads))
    # style_iter = iter(data.DataLoader(
    #     style_dataset, batch_size=args.batch_size,
    #     sampler=InfiniteSamplerWrapper(style_dataset),
    #     num_workers=args.n_threads))

    content_iter = iter(data.DataLoader(content_dataset, batch_size=args.batch_size, shuffle=True))
    style_iter = iter(data.DataLoader(style_dataset, batch_size=args.batch_size, shuffle=True))

    loss_list = []

    #################################################

    for itr in range(args.max_iter):
        content_images = next(content_iter).to(device)
        style_images = next(style_iter).to(device)

        loss_c, loss_s = network(content_images, style_images, args)

        loss_c = args.content_weight * loss_c
        loss_s = args.style_weight * loss_s
        loss = loss_c + loss_s

        loss_list.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print('[{:d}/{:d}]: {:f}'.format(itr, args.max_iter, loss.item()))

        if itr % args.test_interval == 0 and itr!= 0:
            content = test_content_img.to(device).unsqueeze(0)  # whole image(512)
            style = test_style_img.to(device).unsqueeze(0)  # whole image(512)

            with torch.no_grad():
                out, patch_index, groups, repr_patches = network(content, style, args, is_test=True)

            out = denorm_device(out, device)
            style = denorm_device(style, device)
            output = out.to('cpu')
            style = style.to('cpu')
            output = tensor2im(output)
            style = tensor2im(style)

            vis.image(output, win=vis_img)

            patches = vis_patches(style, patch_index, groups)

            vis_patch = ['cluster0', 'cluster1', 'cluster2']
            for i in range(len(patches)):
                vis.images(patches[i], opts=dict(height=200, width=500), win=vis_patch[i])

        if (itr + 1) % args.save_model_interval == 0:
            torch.save(network.state_dict(), './models/patchpalette_{:d}_iters.pth'.format(itr+1))

        vis.line(Y=torch.Tensor([loss_c]), X=torch.Tensor([itr]), win=plot, update='append')
        vis.line(Y=torch.Tensor([loss_s]), X=torch.Tensor([itr]), win=plot2, update='append')