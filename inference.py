import os
import json
import argparse
import torch
import PIL
import numpy as np
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from collections import OrderedDict
import cv2
import argparse


def loadmodel(model,modelpath,device):
    checkpoint = torch.load(modelpath, map_location=device)
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint.keys():
        checkpoint = checkpoint['state_dict']
    # If during training, we used data parallel
    if 'module' in list(checkpoint.keys())[0] and not isinstance(model, torch.nn.DataParallel):
        # for gpu inference, use data parallel
        if "cuda" in device.type:
            model = torch.nn.DataParallel(model)
        else:
        # for cpu inference, remove module
            new_state_dict = OrderedDict()
            for k, v in checkpoint.items():
                name = k[7:]
                new_state_dict[name] = v
            checkpoint = new_state_dict
    # load
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    return model 

def show_focus(img_o,focus,color=(64,244,208),thickness=1):

    contours, im = cv2.findContours(focus, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    res1=cv2.drawContours(img_o.copy(), contours=contours, contourIdx=-1, color=color, thickness=thickness)
    return res1


def main(ftseg_modelpath,ftseg_config,img_folder,output_folder):

    with open(ftseg_config, 'r') as config_file:
        ftseg_c = json.load(config_file)


    transform_val = transforms.Compose([
                                        transforms.Resize((800, 800)),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                                       ])
    
    ftseg_model = getattr(smp, ftseg_c["arch"]["smp"]["decoder_name"])(
                                                        encoder_name=ftseg_c["arch"]["smp"]['encoder_name'],
                                                        encoder_weights='imagenet',
                                                        classes=ftseg_c["class_num"],
                                                        activation=None
                                                        )
    

    availble_gpus = list(range(torch.cuda.device_count()))
    device = torch.device('cuda:0' if len(availble_gpus) > 0 else 'cpu')

    ftsegmodel = loadmodel(ftseg_model,ftseg_modelpath,device)

    for imgname in tqdm(os.listdir(img_folder)):
        i_imgpath = os.path.join(img_folder, imgname)
        o_imgpath = os.path.join(output_folder, imgname[:-4] + '.png')
        img = Image.open(i_imgpath).convert('RGB')
        img1 = cv2.imread(i_imgpath)
        img2 = cv2.resize(img1,(800,800))
        w, h = img.size
        input = transform_val(img).unsqueeze(0)

        ftseg_prediction = ftsegmodel(input.to(device))
        ftseg_prediction = ftseg_prediction.squeeze(0).cpu().detach().numpy()
        ftseg_prediction = F.softmax(torch.from_numpy(ftseg_prediction), dim=0).argmax(0).cpu().numpy()
        ftsegmask_0 = (ftseg_prediction == 0) * 0
        ftsegmask_1 = (ftseg_prediction == 1) * 255
        ftsegmask = ftsegmask_0 + ftsegmask_1
        ftsegmask = np.asarray(ftsegmask.astype(np.uint8))

        img_ftseg = show_focus(img2,ftsegmask,(229,238,0),1)

        img_ftseg = cv2.resize(img_ftseg,(w,h))

        cv2.imwrite(o_imgpath,img_ftseg)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Segmentation Model Inference")
    parser.add_argument('--model_path', type=str, required=True, help='Path to the pretrained model')
    parser.add_argument('--img_folder', type=str, required=True, help='Path to the folder containing input images')
    parser.add_argument('--output_folder', type=str, required=True, help='Path to the folder to save output images')
    parser.add_argument('--config_path', type=str, required=True, help='Path to the configuration file')

    args = parser.parse_args()

    main(args.model_path, args.config_path, args.img_folder, args.output_folder)


