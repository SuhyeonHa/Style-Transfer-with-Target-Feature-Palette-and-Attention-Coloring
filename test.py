import argparse
from pathlib import Path
import torch
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image
from function import adaptive_instance_normalization as adain
from function import denorm_device
import net

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

trans = transforms.Compose([transforms.Resize((512, 512)),
                            transforms.ToTensor(),
                            normalize])

mask_trans = transforms.Compose([transforms.Resize((512, 512)),
                            transforms.ToTensor()])

def style_transfer(net, content, style, args, mask=None, alpha=1.0):
    assert (0.0 <= alpha <= 1.0)
    content_f = net.encode(content)
    style_f = net.encode(style)

    patches, patch_index = net.make_patches(style_f, args.patch_size, args.num_patch)
    centroids, groups = net.patch_cluster(patches, args.cluster_size, args.patch_size)

    t0 = adain(content_f, centroids[:, 0, :, :, :].cuda())
    t0 = alpha * t0 + (1 - alpha) * content_f
    t1 = adain(content_f, centroids[:, 1, :, :, :].cuda())
    t1 = alpha * t1 + (1 - alpha) * content_f
    t2 = adain(content_f, centroids[:, 2, :, :, :].cuda())
    t2 = alpha * t2 + (1 - alpha) * content_f

    cs0 = net.sanet_0(content_f, t0)
    cs1 = net.sanet_1(content_f, t1)
    cs2 = net.sanet_2(content_f, t2)

    t = net.merge_conv(net.merge_conv_pad(cs0 + cs1 + cs2))

    output = net.decoder(t)

    if mask is not None:
        output = output * mask

    return output

def multi_style_transfer(net, content, style_paths, args, alpha=1.0):
    assert (0.0 <= alpha <= 1.0)
    content_f = net.encode(content)

    style0 = trans(Image.open(str(style_paths[0]))).to(device).unsqueeze(0)
    style1 = trans(Image.open(str(style_paths[1]))).to(device).unsqueeze(0)
    style2 = trans(Image.open(str(style_paths[2]))).to(device).unsqueeze(0)

    style_f_0 = net.encode(style0)
    style_f_1 = net.encode(style1)
    style_f_2 = net.encode(style2)

    patches_0, patch_index_0 = net.make_patches(style_f_0, args.patch_size, args.num_patch)
    centroids_0, groups_0 = net.patch_cluster(patches_0, args.cluster_size, args.patch_size)

    patches_1, patch_index_1 = net.make_patches(style_f_1, args.patch_size, args.num_patch)
    centroids_1, groups_1 = net.patch_cluster(patches_1, args.cluster_size, args.patch_size)

    patches_2, patch_index_2 = net.make_patches(style_f_2, args.patch_size, args.num_patch)
    centroids_2, groups_2 = net.patch_cluster(patches_2, args.cluster_size, args.patch_size)

    t0 = adain(content_f, centroids_0[:, 0, :, :, :].cuda())
    t0 = alpha * t0 + (1 - alpha) * content_f
    t1 = adain(content_f, centroids_1[:, 0, :, :, :].cuda())
    t1 = alpha * t1 + (1 - alpha) * content_f
    t2 = adain(content_f, centroids_2[:, 0, :, :, :].cuda())
    t2 = alpha * t2 + (1 - alpha) * content_f

    cs0 = net.sanet_0(content_f, t0)
    cs1 = net.sanet_1(content_f, t1)
    cs2 = net.sanet_2(content_f, t2)

    t = net.merge_conv(net.merge_conv_pad(cs0 + cs1 + cs2))

    output = net.decoder(t)
    out_t0 = net.decoder(t0)
    out_t1 = net.decoder(t1)
    out_t2 = net.decoder(t2)

    return output

def get_args_parser():
    parser = argparse.ArgumentParser()
    # Basic options
    parser.add_argument('--content', type=str, help='File path to the content image')
    parser.add_argument('--content_dir', type=str, help='Directory path to a batch of content images')
    parser.add_argument('--style', type=str, help='File path to the style image')
    parser.add_argument('--style_dir', type=str, help='Directory path to a batch of style images')
    parser.add_argument('--style_mask', type=str, help='File path to the style mask')
    parser.add_argument('--model', type=str, default='models/patchpalette_95000_iters.pth')
    parser.add_argument('--test_mode', type=str, default='single_style_transfer',
                        choices=['single_style_transfer', 'multi_style_transfer', 'spatial_control'])

    # Additional options
    parser.add_argument('--content_size', type=int, default=512)
    parser.add_argument('--style_size', type=int, default=512)
    parser.add_argument('--crop', action='store_true',
                        help='do center crop to create squared image')
    parser.add_argument('--save_ext', default='.jpg',
                        help='The extension name of the output image')
    parser.add_argument('--output', type=str, default='output',
                        help='Directory to save the output image(s)')
    parser.add_argument('--alpha', type=float, default=1.0,
                        help='The weight that controls the degree of \
                                 stylization. Should be between 0 and 1')
    parser.add_argument('--style_interpolation_weights', type=str, default='',
        help='The weight for blending the style of multiple style images')
    parser.add_argument('--cluster_size', type=int, default=3)
    parser.add_argument('--patch_size', '-p', type=int, default=8,
                        help='Size of extracted patches from style features')
    parser.add_argument('--num_patch', type=int, default=100)
    return parser

if __name__ == '__main__':
    args = get_args_parser().parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Either --content or --contentDir should be given.
    assert (args.content or args.content_dir)
    if args.content:
        content_paths = [Path(args.content)]
    else:
        content_dir = Path(args.content_dir)
        content_paths = [f for f in content_dir.glob('*')]

    # Either --style or --styleDir should be given.
    assert (args.style or args.style_dir)
    if args.style:
        style_paths = [Path(args.style)]
    else:
        style_dir = Path(args.style_dir)
        style_paths = [f for f in style_dir.glob('*')]

    decoder = net.decoder
    encoder = net.vgg

    network = net.Net(encoder, decoder)
    network.load_state_dict(torch.load(args.model))
    network.to(device)

    if args.test_mode == 'single_style_transfer':
        for content_path in content_paths:
            for style_path in style_paths:
                content = trans(Image.open(str(content_path)).convert('RGB')).to(device).unsqueeze(0)
                style = trans(Image.open(str(style_path)).convert('RGB')).to(device).unsqueeze(0)

                with torch.no_grad():
                    output = style_transfer(network, content, style, args)

                output = denorm_device(output, device)
                res = output.cpu()
                output_name = output_dir / '{:s}_stylized_{:s}{:s}'.format(
                    content_path.stem, style_path.stem, args.save_ext)
                save_image(res, str(output_name))

    elif args.test_mode == 'multi_style_transfer':
        # Content: The first image of content_dir or content in args
        # Styles: From first to third images of style_dir
        content = trans(Image.open(str(content_paths[0])).convert('RGB')).to(device).unsqueeze(0)

        with torch.no_grad():
            output = multi_style_transfer(network, content, style_paths, args)

        output = denorm_device(output, device)
        res = output.cpu()
        output_name = output_dir / '{:s}_multi_stylized_{:s}'.format(content_paths[0].stem, args.save_ext)
        save_image(res, str(output_name))

    elif args.test_mode == 'spatial_control':
        assert (args.content and args.style_dir and args.style_mask)
        content = trans(Image.open(str(content_paths[0])).convert('RGB')).to(device).unsqueeze(0)
        style1 = trans(Image.open(str(style_paths[0])).convert('RGB')).to(device).unsqueeze(0)
        style2 = trans(Image.open(str(style_paths[1])).convert('RGB')).to(device).unsqueeze(0)
        style_mask = mask_trans(Image.open(str(Path(args.style_mask))).convert('1')).to(device).unsqueeze(0)

        with torch.no_grad():
            output1 = style_transfer(network, content, style1, args, style_mask)
            output2 = style_transfer(network, content, style2, args, 1-style_mask)

        output1 = denorm_device(output1, device).cpu()
        output2 = denorm_device(output2, device).cpu()
        style_mask = style_mask.cpu()
        res = output1*style_mask + output2*(1-style_mask)
        output_name = output_dir / '{:s}_spatial_control_{:s}'.format(content_paths[0].stem, args.save_ext)
        save_image(res, str(output_name))