import os
import cv2
import numpy as np
import random
import matplotlib.pyplot as plt
import lpips

import torch
import torch.nn as nn
import torchvision

from torch.distributions.normal import Normal
from skimage.metrics import structural_similarity as ssim
from model import resnet34
from utils import foveat_img

class Actor(nn.Module):

    def __init__(self):
        super().__init__()
        self.backbone = resnet34(pretrained=True)
        self.fixation_conv = nn.Sequential(nn.Conv2d(64, 16, 3, padding=1),
                                           nn.LeakyReLU(inplace=True),
                                           nn.Conv2d(16, 4, 3, padding=1),
                                           nn.LeakyReLU(inplace=True),
                                           nn.Conv2d( 4, 1, 1, padding=0))

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.blur_conv = nn.Sequential(nn.Linear(64, 16),
                                       nn.LeakyReLU(inplace=True),
                                       nn.Linear(16, 4),
                                       nn.LeakyReLU(inplace=True),
                                       nn.Linear( 4, 1))
        self.distribution = Normal(torch.Tensor([0.0]), torch.Tensor([1.0]))

    def forward(self, x, no_blur=False):

        feats, _ = self.backbone(x)
        feats = nn.functional.interpolate(feats, (335, 447))
        center = nn.functional.softmax(self.fixation_conv(feats).flatten())
        blur_mean = self.blur_conv(self.avgpool(feats).flatten(start_dim=1)) * (no_blur == False)
        blur_mean = nn.functional.sigmoid(blur_mean) * 3.5
        self.distribution = Normal(blur_mean.flatten(), torch.Tensor([1.0]).cuda())

        return center

    def pi(self, s, a):
        probs_center = self.forward(s)
        return probs_center[a]

    def update_weight(self, state, action, blur_p, reward, optimizer):
        loss = (-1.0) * reward * torch.log(self.pi(state, action)) +\
            (-1.0) * reward * self.distribution.log_prob(blur_p)
        # update policy parameter \theta
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def eval(img_name, actor, epoch, resnet, loss_fn_vgg):

    with torch.no_grad():

        img = cv2.imread(img_name)
        if img.shape != (335, 447, 3):
            img = cv2.resize(img, (447, 335))
        img = img[:, :, ::-1]

        orig_state = torch.Tensor(img.copy()).permute(2, 0, 1).unsqueeze(0) / 255
        tr = torchvision.transforms.Normalize((0.485, 0.456, 0.406),
                                              (0.229, 0.224, 0.225))
        orig_state[0] = tr(orig_state[0])
        orig_state = orig_state.cuda()
        orig_preds = resnet(orig_state)
        log_orig_preds = nn.functional.log_softmax(orig_preds, dim=1)

        actor.eval()

        fixation_probs = actor(orig_state)
        action = torch.argmax(fixation_probs).item()
        blur_p = actor.distribution.mean

        fov_img, num_full_res_pixels = foveat_img(img, [(action%447, action//447)], blur_p.item()+7, 3, 1.5)
        state = torch.Tensor(fov_img.copy()).permute(2, 0, 1).unsqueeze(0) / 255
        state[0] = tr(state[0])
        state = state.cuda()
        fov_preds = resnet(state)
        log_fov_preds = nn.functional.log_softmax(fov_preds, dim=1)

        kl_div = nn.functional.kl_div(log_fov_preds, log_orig_preds, reduction='sum', log_target=True).item()

        ssi = ssim(fov_img, img, multichannel=True)

        perceptual_loss = loss_fn_vgg(state.cpu(), orig_state.cpu()).item()

        # cv2.imshow("frame", fov_img[:, :, ::-1])
        # key = cv2.waitKey(100)
        if not os.path.exists(os.path.join("outputs", str(epoch))):
            os.makedirs(os.path.join("outputs", str(epoch)))

        img_name = os.path.basename(img_name).split(".")[0]
        cv2.imwrite(os.path.join("outputs", str(epoch), "{}_{}_{}_{}.jpg".format(img_name, action%447, action//447, blur_p.item()+7)),
            fov_img[:, :, ::-1])

        return kl_div, ssi, perceptual_loss, num_full_res_pixels

if __name__ == '__main__':

    resnet = torchvision.models.resnet34(pretrained=True).cuda()
    resnet.eval()
    loss_fn_vgg = lpips.LPIPS(net='vgg')

    actor = Actor().cuda()
    actor.load_state_dict(torch.load("ckpt/25.pth"))
    img_names = os.listdir(os.path.abspath(os.path.join(__file__ ,"../../dataset/COCO-2017-test-500/data")))
    img_names = [name for name in img_names if name.endswith(".jpg") or name.endswith(".png")]
    metrics = []
    random.shuffle(img_names)
    for img_name in img_names:
        print(img_name)
        img_path = os.path.join(os.path.join(__file__ ,"../../dataset/COCO-2017-test-500/data"), img_name)
        result = eval(img_path, actor, 0, resnet, loss_fn_vgg)
        metrics.append(result)
    print(np.mean(metrics, axis=0))

# if __name__ == '__main__':
#     actor = Actor().cuda()
#     actor.load_state_dict(torch.load("ckpt/999.pth"))
#     img_names = os.listdir("data")
#     img_names = [name for name in img_names if name.endswith(".jpg") or name.endswith(".png")]
#     metrics = []
#     random.shuffle(img_names)
#     for img_name in img_names:
#         img_path = os.path.join("data", img_name)
#         result = eval(img_path, actor, 0)
#         metrics.append(result)
#     print(np.mean(metrics, axis=0))
