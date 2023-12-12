import os, time
import numpy as np
import streamlit as st
import tempfile
from scipy import ndimage
from scipy.ndimage import label
from functools import partial
import monai
from monai.inferers import sliding_window_inference
from monai.data import load_decathlon_datalist
from monai.transforms import AsDiscrete,AsDiscreted,Compose,Invertd,SaveImaged
from monai import transforms, data
from networks.swin3d_unetrv2 import SwinUNETR as SwinUNETR_v2
import nibabel as nib
import torch

import warnings
warnings.filterwarnings("ignore")

import argparse
parser = argparse.ArgumentParser(description='liver tumor validation')

# file dir
parser.add_argument('--val_dir', default=None, type=str)
parser.add_argument('--json_dir', default=None, type=str)
parser.add_argument('--save_dir', default='out', type=str)
parser.add_argument('--checkpoint', action='store_true')

parser.add_argument('--log_dir', default=None, type=str)
parser.add_argument('--feature_size', default=16, type=int)
parser.add_argument('--val_overlap', default=0.5, type=float)
parser.add_argument('--num_classes', default=3, type=int)

parser.add_argument('--model', default='unet', type=str)
parser.add_argument('--swin_type', default='tiny', type=str)

def denoise_pred(pred: np.ndarray):
    """
    # 0: background, 1: liver, 2: tumor.
    pred.shape: (3, H, W, D)
    """
    denoise_pred = np.zeros_like(pred)

    live_channel = pred[1, ...]
    labels, nb = label(live_channel)
    max_sum = -1
    choice_idx = -1
    for idx in range(1, nb+1):
        component = (labels == idx)
        if np.sum(component) > max_sum:
            choice_idx = idx
            max_sum = np.sum(component)
    component = (labels == choice_idx)
    denoise_pred[1, ...] = component

    # 膨胀然后覆盖掉liver以外的tumor
    liver_dilation = ndimage.binary_dilation(denoise_pred[1, ...], iterations=30).astype(bool)
    denoise_pred[2,...] = pred[2,...].astype(bool) * liver_dilation

    denoise_pred[0,...] = 1 - np.logical_or(denoise_pred[1,...], denoise_pred[2,...])

    return denoise_pred

def cal_dice(pred, true):
    intersection = np.sum(pred[true==1]) * 2.0
    dice = intersection / (np.sum(pred) + np.sum(true))
    return dice



def _get_model(val_dir, json_dir, save_dir, log_dir, checkpoint, feature_size, val_overlap, num_classes, model, swin_type):
    inf_size = [96, 96, 96]
    print(model)
    if model == 'swin_unetrv2':
        if swin_type == 'tiny':
            feature_size=12
        elif swin_type == 'small':
            feature_size=24
        elif swin_type == 'base':
            feature_size=48

        model = SwinUNETR_v2(in_channels=1,
                          out_channels=3,
                          img_size=(96, 96, 96),
                          feature_size=feature_size,
                          patch_size=2,
                          depths=[2, 2, 2, 2],
                          num_heads=[3, 6, 12, 24],
                          window_size=[7, 7, 7])
        
    elif model == 'unet':
        from monai.networks.nets import UNet 
        model = UNet(
                    spatial_dims=3,
                    in_channels=1,
                    out_channels=3,
                    channels=(16, 32, 64, 128, 256),
                    strides=(2, 2, 2, 2),
                    num_res_units=2,
                )
    
    else:
        raise ValueError('Unsupported model ' + str(model))


    if checkpoint:
        checkpoint = torch.load(os.path.join(log_dir, 'model.pt'), map_location='cpu')

        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in checkpoint['state_dict'].items():
            new_state_dict[k.replace('backbone.','')] = v
        # load params
        model.load_state_dict(new_state_dict, strict=False)
        print('Use logdir weights')
    else:
        model_dict = torch.load(os.path.join(log_dir, 'model.pt'))
        model.load_state_dict(model_dict['state_dict'])
        print('Use logdir weights')

    model = model.cuda()
    model_inferer = partial(sliding_window_inference, roi_size=inf_size, sw_batch_size=1, predictor=model,  overlap=val_overlap, mode='gaussian')
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Total parameters count', pytorch_total_params)

    return model, model_inferer

def _get_loader(val_dir, json_dir):
    val_data_dir = val_dir
    datalist_json = json_dir 
    val_org_transform = transforms.Compose(
    [
        transforms.LoadImaged(keys=["image"]),
        transforms.AddChanneld(keys=["image"]),
        transforms.Orientationd(keys=["image"], axcodes="RAS"),
        transforms.Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear")),
        transforms.ScaleIntensityRanged(keys=["image"], a_min=-21, a_max=189, b_min=0.0, b_max=1.0, clip=True),
        transforms.SpatialPadd(keys=["image"], mode="minimum", spatial_size=[96, 96, 96]),
        transforms.ToTensord(keys=["image"]),
    ]
    )
    val_files = load_decathlon_datalist(datalist_json, True, "validation", base_dir=val_data_dir)
    val_org_ds = data.Dataset(val_files, transform=val_org_transform)
    val_org_loader = data.DataLoader(val_org_ds, batch_size=1, shuffle=False, num_workers=4, sampler=None, pin_memory=True)

    post_transforms = Compose([
        Invertd(
            keys="pred",
            transform=val_org_transform,
            orig_keys="image",
            meta_keys="pred_meta_dict",
            orig_meta_keys="image_meta_dict",
            meta_key_postfix="meta_dict",
            nearest_interp=False,
            to_tensor=True,
        ),
        # AsDiscreted(keys="pred", argmax=True, to_onehot=3),
        AsDiscreted(keys="pred", argmax=True, to_onehot=3),
        AsDiscreted(keys="label", to_onehot=3),
        # SaveImaged(keys="pred", meta_keys="pred_meta_dict", output_dir=output_dir, output_postfix="seg", resample=False,output_dtype=np.uint8,separate_folder=False),
    ])
    
    return val_org_loader, post_transforms

def main(val_dir, json_dir, save_dir, log_dir, checkpoint, feature_size, val_overlap, num_classes, model, swin_type):
    model_name = log_dir.split('/')[-1]

    print("MAIN Argument values:")
    print('val_dir =>', val_dir)
    print('json_dir =>', json_dir)
    print('save_dir =>', save_dir)
    print('log_dir =>', log_dir)
    print('checkpoint =>', checkpoint)
    print('feature_size =>', feature_size)
    print('val_overlap =>', val_overlap)
    print('num_classes =>', num_classes)
    print('model =>', model)
    print('swin_type =>', swin_type)
    print('-----------------')

    torch.cuda.set_device(0) #use this default device (same as args.device if not distributed)
    torch.backends.cudnn.benchmark = True

    ## loader and post_transform
    val_loader, post_transforms = _get_loader(val_dir, json_dir)

    ## NETWORK
    model, model_inferer = _get_model(val_dir, json_dir, save_dir, log_dir, checkpoint, feature_size, val_overlap, num_classes, model, swin_type)

    model.eval()
    start_time = time.time()
    with torch.no_grad():
        for idx, val_data in enumerate(val_loader):
            val_inputs = val_data["image"].cuda()
            # name = val_data['label_meta_dict']['filename_or_obj'][0].split('/')[-1].split('.')[0]
            # original_affine = val_data["label_meta_dict"]["affine"][0].numpy()
            # pixdim = val_data['label_meta_dict']['pixdim'].cpu().numpy()
            # spacing_mm = tuple(pixdim[0][1:4])

            val_data["pred"] = model_inferer(val_inputs)
            val_data = [post_transforms(i) for i in data.decollate_batch(val_data)]
            # val_outputs, val_labels = from_engine(["pred", "label"])(val_data)
            val_outputs = val_data[0]['pred']
            
            # val_outpus.shape == val_labels.shape  (3, H, W, Z)
            val_outputs = val_outputs.detach().cpu().numpy()

            # denoise the ouputs 
            val_outputs = denoise_pred(val_outputs)

            # save the prediction
            output_dir = os.path.join(save_dir, model_name, str(val_overlap), 'pred')
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            val_outputs = np.argmax(val_outputs, axis=0)

            nib.save(
                nib.Nifti1Image(val_outputs.astype(np.uint8), original_affine), os.path.join(output_dir, f'{name}.nii.gz')
            )

# save path: save_dir/log_dir_name/str(args.val_overlap)/pred/
st.title("Liver CT Scan Tumor Prediction")
uploaded_file = st.file_uploader("Upload a CT scan (.nii.gz)", type=["nii.gz"])

if uploaded_file is not None:
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_file_path = os.path.join(temp_dir, "temp.nii.gz")
        with open(temp_file_path, "wb") as file:
            file.write(uploaded_file.getbuffer())
        val_dir = os.path.dirname(temp_file_path)
        json_dir = "datafolds/lits.json"  # Update this path as needed
        save_dir = temp_dir  # Directory where predictions will be saved
        log_dir = "runs/synt.pretrain.swin_unetrv2_base"  # Update this path as needed
        main(val_dir, json_dir, save_dir, log_dir, checkpoint=False, feature_size=16, val_overlap=0.75, num_classes=3, model='swin_unetrv2', swin_type='base')
        output_dir = os.path.join(save_dir, model_name, str(val_overlap), 'pred')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        prediction_file = os.path.join(output_dir, "temp.nii.gz")
        if os.path.exists(prediction_file):
            with open(prediction_file, "rb") as file:
                st.download_button(
                    label="Download Prediction",
                    data=file,
                    file_name="prediction_result.nii.gz",
                    mime="application/gzip"
                )
