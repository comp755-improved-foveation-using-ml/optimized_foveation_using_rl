#!/usr/bin/env python3

from skimage.metrics import structural_similarity as ssim
from PIL import Image
from torchvision import transforms
from resnet import resnet34
import numpy as np
import torch.nn.functional as F
import torch
import pdb
import lpips

def main():
    model = resnet34(True, True)
    # model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet34', pretrained=True)
    model.eval()
    filename = "../sample_images/29393"
    input_image = Image.open(filename + '.png').convert('RGB')
    input_image_fov = Image.open(filename + '_GD.jpg').convert('RGB')
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_image = preprocess(input_image)
    input_image_fov = preprocess(input_image_fov)
    input_tensor = input_image.unsqueeze(0)
    input_tensor_fov = input_image_fov.unsqueeze(0)
    # black = torch.zeros_like(input_tensor)

    with torch.no_grad():
        _, output = model(input_tensor)
        _, output_fov = model(input_tensor_fov)
        # _, output_black = model(black)

    print(np.moveaxis(input_image.numpy(), 0, -1).shape)
    prob = F.softmax(output[0], dim=0)
    prob_fov = F.softmax(output_fov[0], dim=0)
    # prob_black = F.softmax(output_black[0], dim=0)
    logprob = F.log_softmax(output[0], dim=0)
    logprob_fov = F.log_softmax(output_fov[0], dim=0)
    # logprob_black = F.log_softmax(output_black[0], dim=0)
    div = F.kl_div(logprob_fov, logprob, reduction='sum', log_target=True) # lower is more similar

    ssi = ssim(np.moveaxis(input_image_fov.numpy(), 0, -1), np.moveaxis(input_image.numpy(), 0, -1), multichannel=True) # higher is more similar

    loss_fn_alex = lpips.LPIPS(net='alex') # best forward scores
    loss_fn_vgg = lpips.LPIPS(net='vgg') # closer to "traditional" perceptual loss, when used for optimization
    da = loss_fn_alex(input_tensor_fov, input_tensor) # lower is more similar
    dv = loss_fn_vgg(input_tensor_fov, input_tensor) # lower is more simlar

    pdb.set_trace()
    print(div)


if __name__ == '__main__':
    main()
