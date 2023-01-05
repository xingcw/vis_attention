import os
import cv2
import torch
import numpy as np
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

from trans_exp.baselines.ViT.ViT_LRP import vit_base_patch16_224 as vit_LRP
from trans_exp.baselines.ViT.ViT_explanation_generator import LRP
from trans_exp.utils.load_weights import load_weights


# create heatmap from mask on image
def show_cam_on_image(img, mask):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return cam


def generate_visualization(attribution_generator, original_image, class_index=None, use_thresholding=False):
    transformer_attribution = attribution_generator.generate_LRP(original_image.unsqueeze(0).cuda(), 
                                                                 method="transformer_attribution", 
                                                                 index=class_index).detach()
    transformer_attribution = transformer_attribution.reshape(1, 1, 14, 14)
    transformer_attribution = torch.nn.functional.interpolate(transformer_attribution, scale_factor=16, mode='bilinear')
    transformer_attribution = transformer_attribution.reshape(224, 224).data.cpu().numpy()
    transformer_attribution = (transformer_attribution - 
                               transformer_attribution.min()) / (
                                   transformer_attribution.max() - 
                                   transformer_attribution.min())

    if use_thresholding:
      transformer_attribution = transformer_attribution * 255
      transformer_attribution = transformer_attribution.astype(np.uint8)
      ret, transformer_attribution = cv2.threshold(transformer_attribution, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
      transformer_attribution[transformer_attribution == 255] = 1

    image_transformer_attribution = original_image.permute(1, 2, 0).data.cpu().numpy()
    image_transformer_attribution = (image_transformer_attribution - 
                                     image_transformer_attribution.min()) / (
                                         image_transformer_attribution.max() - 
                                         image_transformer_attribution.min())
                                     
    vis = show_cam_on_image(image_transformer_attribution, transformer_attribution)
    vis =  np.uint8(255 * vis)
    vis = cv2.cvtColor(np.array(vis), cv2.COLOR_RGB2BGR)
    return vis


def main():

    flightmare_path = Path(os.environ["FLIGHTMARE_PATH"])
    sample_path = flightmare_path / "flightpy/results/students/teacher_PPO_5/12-26-14-24-47/data/000/000437.npz"

    image = np.load(sample_path)
    image = Image.fromarray(image["rgb"].squeeze())

    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(image)
    axs[0].axis('off')

    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])
    image = transform(image)

    # initialize ViT pretrained
    model = vit_LRP(pretrained=True).cuda()
    
    # get trained weights in the correct format
    weight_path = flightmare_path / "flightpy/results/students/teacher_PPO_5/vit_01-03-13-54-10/model/vit_ep20_data_data_584640.pth"
    weights = load_weights(weight_path, "vit", "cuda")
    
    model.load_state_dict(weights, strict=False)
    model.eval()
    
    attribution_generator = LRP(model)
    output = model(image.unsqueeze(0).cuda())
    heatmap = generate_visualization(attribution_generator, image)

    axs[1].imshow(heatmap)
    axs[1].axis('off')
    
    plt.show()


if __name__ == "__main__":
    main()