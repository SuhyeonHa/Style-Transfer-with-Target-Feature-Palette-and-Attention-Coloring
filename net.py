import torch.nn as nn
import torch

from function import adaptive_instance_normalization as adain
from function import calc_mean_std, mean_variance_norm
from sklearn.cluster import KMeans
import numpy as np
import random

decoder = nn.Sequential(
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 256, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 128, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 64, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 3, (3, 3)),
)

vgg = nn.Sequential(
    nn.Conv2d(3, 3, (1, 1)),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(3, 64, (3, 3)),
    nn.ReLU(),  # relu1-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),  # relu1-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 128, (3, 3)),
    nn.ReLU(),  # relu2-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),  # relu2-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 256, (3, 3)),
    nn.ReLU(),  # relu3-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 512, (3, 3)),
    nn.ReLU(),  # relu4-1, this is the last layer used
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU()  # relu5-4
)

class SANet(nn.Module):
    def __init__(self, in_planes):
        super(SANet, self).__init__()
        self.f = nn.Conv2d(in_planes, in_planes, (1, 1))
        self.g = nn.Conv2d(in_planes, in_planes, (1, 1))
        self.h = nn.Conv2d(in_planes, in_planes, (1, 1))
        self.sm = nn.Softmax(dim = -1)
        self.out_conv = nn.Conv2d(in_planes, in_planes, (1, 1))

    def forward(self, content, style):
        F = self.f(mean_variance_norm(content))
        G = self.g(mean_variance_norm(style))
        H = self.h(style)
        b, c, h, w = F.size()
        F = F.view(b, -1, w * h).permute(0, 2, 1)
        b, c, h, w = G.size()
        G = G.view(b, -1, w * h)
        S = torch.bmm(F, G)
        S = self.sm(S)
        b, c, h, w = H.size()
        H = H.view(b, -1, w * h)
        O = torch.bmm(H, S.permute(0, 2, 1))
        b, c, h, w = content.size()
        O = O.view(b, c, h, w)
        O = self.out_conv(O)
        O += content
        return O

class Net(nn.Module):
    def __init__(self, encoder, decoder):
        super(Net, self).__init__()
        enc_layers = list(encoder.children())
        self.enc_1 = nn.Sequential(*enc_layers[:4])  # input -> relu1_1
        self.enc_2 = nn.Sequential(*enc_layers[4:11])  # relu1_1 -> relu2_1
        self.enc_3 = nn.Sequential(*enc_layers[11:18])  # relu2_1 -> relu3_1
        self.enc_4 = nn.Sequential(*enc_layers[18:31])  # relu3_1 -> relu4_1
        self.decoder = decoder
        self.relu = nn.LeakyReLU()
        self.mse_loss = nn.MSELoss()
        self.sanet_0 = SANet(in_planes=512)
        self.sanet_1 = SANet(in_planes=512)
        self.sanet_2 = SANet(in_planes=512)
        self.merge_conv_pad = nn.ReflectionPad2d((1, 1, 1, 1))
        self.merge_conv = nn.Conv2d(512, 512, (3, 3))

        # fix the encoder
        for name in ['enc_1', 'enc_2', 'enc_3', 'enc_4']:
            for param in getattr(self, name).parameters():
                param.requires_grad = False

    # extract relu1_1, relu2_1, relu3_1, relu4_1 from input image
    def encode_with_intermediate(self, input):
        results = [input]
        for i in range(4):
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            results.append(func(results[-1]))
        return results[1:]

    # extract relu4_1 from input image
    def encode(self, input):
        for i in range(4):
            input = getattr(self, 'enc_{:d}'.format(i + 1))(input)
        return input

    def calc_content_loss(self, input, target):
        assert (input.size() == target.size())
        # assert (target.requires_grad is False)
        # return self.mse_loss(input, target) # Original AdaIN
        return self.mse_loss(mean_variance_norm(input), mean_variance_norm(target))

    def calc_style_loss(self, input, target):
        assert (input.size() == target.size())
        assert (target.requires_grad is False)
        input_mean, input_std = calc_mean_std(input)
        target_mean, target_std = calc_mean_std(target)
        return self.mse_loss(input_mean, target_mean) + \
               self.mse_loss(input_std, target_std)

    def make_patches(self, image, patch_size, num_patch):
        B, _, H, W = image.shape
        batch_size = B
        width, height = H, W
        crop_width, crop_height = patch_size, patch_size
        patches = [] # save cropped patches
        patch_index = [] # save coordinates of the patch

        for b in range(batch_size):
            for i in range(num_patch):
                x, y = random.randrange(0, max(0, width - crop_width) + 1), random.randrange(0, max(0, height - crop_height) + 1)
                patch = image[b, :, y:y+crop_height, x:x+crop_width].unsqueeze(0)
                patch_index.append([x, x+crop_width, y, y+crop_height])
                # save patches as tensor
                if i == 0:
                    patches = patch
                else:
                    patches = torch.cat((patches, patch))
            if b == 0:
                result = patches.unsqueeze(0)
            else:
                result = torch.cat((result, patches.unsqueeze(0)))

            # result_patches (8, 20, 512, 8, 8)
            # patch_index (8*20 [w1, w2, h1, h2])

        return result, patch_index

    def patch_cluster(self, patch_group, num_cluster, patch_size):
        B, N, C, H, W = patch_group.shape  # (8, 20, 512, 8, 8)
        batch_size = B
        groups = []

        for b in range(batch_size):
            patches = patch_group[b, :, :, :, :]
            s = patches.reshape(N, -1)
            km = KMeans(n_clusters=num_cluster)
            clustering = km.fit(s)
            y_pred = clustering.labels_

            num_sample = len(y_pred)
            selected_patch = torch.zeros(3, 512, patch_size, patch_size)
            for i in range(num_cluster):
                d = km.fit_transform(s)[:, i]
                ind = np.argsort(d)[::][:50]
                selected_patch[i] = patches[ind[0]]

            groups.append(list())
            for i in range(num_cluster):
                groups[b].append(list())

            # groups have samples with the group's label
            for i in range(num_sample):
                for j in range(num_cluster):
                    if y_pred[i] == j:
                        groups[b][j].append(i)

            if b == 0:
                result_patches = torch.Tensor(selected_patch.unsqueeze(0))
            else:
                result_patches = torch.cat((result_patches, selected_patch.unsqueeze(0)))

        return result_patches, groups

    def forward(self, content, style, args, alpha=1, is_test=False):
        assert 0 <= alpha <= 1

        content_feat = self.encode(content)
        style_feats = self.encode_with_intermediate(style)

        patches, patch_index = self.make_patches(style_feats[-1], args.patch_size, args.num_patch)
        repr_patches, groups = self.patch_cluster(patches, args.cluster_size, args.patch_size)

        t0 = adain(content_feat, repr_patches[:, 0, :, :, :].cuda())
        t0 = alpha * t0 + (1 - alpha) * content_feat
        t1 = adain(content_feat, repr_patches[:, 1, :, :, :].cuda())
        t1 = alpha * t1 + (1 - alpha) * content_feat
        t2 = adain(content_feat, repr_patches[:, 2, :, :, :].cuda())
        t2 = alpha * t2 + (1 - alpha) * content_feat

        cs0 = self.sanet_0(content_feat, t0)
        cs1 = self.sanet_1(content_feat, t1)
        cs2 = self.sanet_2(content_feat, t2)

        t = self.merge_conv(self.merge_conv_pad(cs0+cs1+cs2))

        out = self.decoder(t)
        out_feats = self.encode_with_intermediate(out)

        if is_test == True:
            return out, patch_index, groups, repr_patches

        loss_c = self.calc_content_loss(out_feats[-1], content_feat)
        loss_s = self.calc_style_loss(out_feats[0], style_feats[0])
        for i in range(1, 4):
            loss_s += self.calc_style_loss(out_feats[i], style_feats[i])

        return loss_c, loss_s