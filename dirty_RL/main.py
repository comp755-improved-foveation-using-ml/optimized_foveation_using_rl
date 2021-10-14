import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torchvision

from model import resnet34
from utils import foveat_img

FIXATION_STEPSIZE = 10

class Actor(nn.Module):

    def __init__(self):
        super().__init__()
        self.backbone = resnet34(pretrained=True)
        self.fixation_conv = nn.Sequential(nn.Conv2d(64, 16, 3, padding=1),
                                           nn.LeakyReLU(inplace=True),
                                           nn.Conv2d(16, 4, 3, padding=1),
                                           nn.LeakyReLU(inplace=True),
                                           nn.Conv2d( 4, 1, 1, padding=0))

    def forward(self, x):

        feats, _ = self.backbone(x)
        feats = nn.functional.interpolate(feats, (335, 447))
        center = nn.functional.softmax(self.fixation_conv(feats).flatten())

        return center

    def pi(self, s, a):
        probs_center = self.forward(s)
        return probs_center[a]

    def update_weight(self, state, action, reward, optimizer):
        loss = (-1.0) * reward * torch.log(self.pi(state, action))
        # update policy parameter \theta
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def gen_gaussian():
    x, y = np.meshgrid(np.linspace(1,447,447), np.linspace(1,335,335))
    dst = np.sqrt((x-186)**2+(y-163)**2)
    
    # Intializing sigma and muu
    sigma = 40
    muu = 0.000
    
    # Calculating Gaussian array
    return np.exp(-( (dst-muu)**2 / ( 2.0 * sigma**2 ) ) )

def main():

    actor = Actor().cuda()
    resnet = torchvision.models.resnet34(pretrained=True).cuda()
    resnet.eval()
    optimizer = torch.optim.AdamW(actor.parameters(), lr=1e-3)


    tr = torchvision.transforms.Normalize((0.485, 0.456, 0.406),
                                          (0.229, 0.224, 0.225))

    img = cv2.imread("29393.png")
    img = img[:, :, ::-1]

    orig_state = torch.Tensor(img.copy()).permute(2, 0, 1).unsqueeze(0) / 255
    orig_state[0] = tr(orig_state[0])
    orig_state = orig_state.cuda()
    with torch.no_grad():
        orig_preds = resnet(orig_state)
        log_orig_preds = nn.functional.log_softmax(orig_preds, dim=1)
    
    gauss = gen_gaussian().flatten()

    for episode in range(10000):

        actor.train()

        probs = actor(orig_state)
        action = probs.multinomial(1).item()

        with torch.no_grad():
            fov_img, num_full_res_pixels = foveat_img(img, [(action%447, action//447)], 7, 3, 1.5)
            state = torch.Tensor(fov_img.copy()).permute(2, 0, 1).unsqueeze(0) / 255
            state[0] = tr(state[0])
            state = state.cuda()
            fov_preds = resnet(state)
            log_fov_preds = nn.functional.log_softmax(fov_preds, dim=1)

        reward = 1. / nn.functional.kl_div(log_fov_preds, log_orig_preds, reduction='sum', log_target=True).item()
        reward = reward**3
        # try:
        #     reward = -torch.log(nn.functional.kl_div(log_fov_preds, log_orig_preds, reduction='sum', log_target=True) + 1e-8).item()/10.
        # except:
        #     import pdb; pdb.set_trace()
        print("({}, {}) {}".format(action%447, action//447, reward))

        # reward = gauss[action]

        actor.update_weight(orig_state, action, reward, optimizer)

        # if episode%10 == 0:
        #     probs = probs.reshape(335, 447).detach().cpu().numpy()
        #     heatmap = cv2.applyColorMap((probs * 255).astype(np.uint8), cv2.COLORMAP_JET)
        #     cv2.imshow("123", heatmap)
        #     key = cv2.waitKey(500)

        # rewards.append(reward)

        # if done:
        #     print("Episode {} finished after {} timesteps".format(i_episode, timesteps+1))
        #     break

        # actor.update_weight(states, actions, rewards, optimizer)

        # evaluation
        with torch.no_grad():

            actor.eval()

            probs = actor(orig_state)
            action = torch.argmax(probs).item()
            if episode%10 == 0:
                fov_img, num_full_res_pixels = foveat_img(img, [(action%447, action//447)], 7, 3, 1.5)
                cv2.imshow("frame", fov_img[:, :, ::-1])
                key = cv2.waitKey(100)
                print("({}, {})".format(action%447, action//447))
                cv2.imwrite(os.path.join("kl_outputs", "{}.png".format(episode//10)), fov_img[:, :, ::-1])


if __name__ == '__main__':
    main()
