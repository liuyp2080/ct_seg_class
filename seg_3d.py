import os
import nibabel as nib
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from monai.data import ITKWriter
from scipy.ndimage import zoom
import glob
from monai.losses import GeneralizedDiceFocalLoss
from monai.inferers import sliding_window_inference
from monai.transforms import (
    AsDiscrete,
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandFlipd,
    RandCropByPosNegLabeld,
    RandShiftIntensityd,
    ScaleIntensityRanged,
    Spacingd,
    RandRotate90d,
)
import torchvision.transforms as transforms
from monai.config import print_config
from monai.metrics import DiceMetric
from monai.networks.nets import UNETR
import torch.nn.functional as F
from torch.optim.lr_scheduler import ExponentialLR

from monai.data import (
    DataLoader,
    CacheDataset,
    load_decathlon_datalist,
    decollate_batch,
)


import torch

print_config()


train_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(
            keys=["image", "label"],
            pixdim=(1, 1, 5.0),
            mode=("bilinear", "nearest"),
        ),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=-175,
            a_max=250,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        # CropForegroundd(keys=["image", "label"], source_key="image"),
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=(128,128, 16),
            pos=1,
            neg=1,
            num_samples=4,
            image_key="image",
            image_threshold=0,
        ),
        RandFlipd(
            keys=["image", "label"],
            spatial_axis=[0],
            prob=0.10,
        ),
        RandFlipd(
            keys=["image", "label"],
            spatial_axis=[1],
            prob=0.10,
        ),
        RandFlipd(
            keys=["image", "label"],
            spatial_axis=[2],
            prob=0.10,
        ),
        RandRotate90d(
            keys=["image", "label"],
            prob=0.10,
            max_k=3,
        ),
        RandShiftIntensityd(
            keys=["image"],
            offsets=0.10,
            prob=0.50,
        ),
    ]
)
val_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(
            keys=["image", "label"],
            pixdim=(1, 1, 5.0),
            mode=("bilinear", "nearest"),
        ),
        ScaleIntensityRanged(keys=["image"], a_min=-175, a_max=250, b_min=0.0, b_max=1.0, clip=True),
        # CropForegroundd(keys=["image", "label"], source_key="image"),
    ]
)
test_transforms = Compose(
    [
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        Orientationd(keys=["image"], axcodes="RAS"),
        Spacingd(keys=["image"], pixdim=(1, 1, 5.0), mode="bilinear"),
        ScaleIntensityRanged(keys=["image"], a_min=-175, a_max=250, b_min=0.0, b_max=1.0, clip=True),
        # CropForegroundd(keys=["image"],source_key="image"),
    ]
)


root_dir = "/work/home/pazhou_045/"
split_json = "dataset_0.json"

datasets = root_dir + split_json
datalist = load_decathlon_datalist(datasets, True, "training")
val_files = load_decathlon_datalist(datasets, True, "validation")
test_files=load_decathlon_datalist(datasets, True, "test")
train_ds = CacheDataset(
    data=datalist,
    transform=train_transforms,
    cache_num=24,
    cache_rate=1.0,
    num_workers=8,
)
train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=8, pin_memory=True)
val_ds = CacheDataset(data=val_files, transform=val_transforms, cache_num=6, cache_rate=1.0, num_workers=4)
val_loader = DataLoader(val_ds, batch_size=2, shuffle=False, num_workers=4, pin_memory=True)

test_ds=CacheDataset(data=test_files, transform=test_transforms, cache_num=6, cache_rate=1.0, num_workers=4)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = UNETR(
    in_channels=1,
    out_channels=5,
    img_size=(128,128, 16),
    feature_size=16,
    hidden_size=768,
    mlp_dim=3072,
    num_heads=12,
    pos_embed="perceptron",
    norm_name="instance",
    res_block=True,
    dropout_rate=0.0,
).to(device)

loss_function = GeneralizedDiceFocalLoss(to_onehot_y=True, softmax=True)
torch.backends.cudnn.benchmark = True
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4,weight_decay=1e-5)

def validation(epoch_iterator_val):
    model.eval()
    with torch.no_grad():
        for batch in epoch_iterator_val:
            val_inputs, val_labels = (batch["image"].to(device), batch["label"].to(device))
            val_outputs = sliding_window_inference(val_inputs, (128,128, 16), 4, model)
            val_labels_list = decollate_batch(val_labels)
            val_labels_convert = [post_label(val_label_tensor) for val_label_tensor in val_labels_list]
            val_outputs_list = decollate_batch(val_outputs)
            val_output_convert = [post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list]
            dice_metric(y_pred=val_output_convert, y=val_labels_convert)
            epoch_iterator_val.set_description("Validate (%d / %d Steps)" % (global_step, 10.0))
        mean_dice_val = dice_metric.aggregate().item()
        dice_metric.reset()
    return mean_dice_val


def train(global_step, train_loader, dice_val_best, global_step_best):
    model.train()
    epoch_loss = 0
    step = 0
    epoch_iterator = tqdm(train_loader, desc="Training (X / X Steps) (loss=X.X)", dynamic_ncols=True)
    for step, batch in enumerate(epoch_iterator):
        step += 1
        x, y = (batch["image"].to(device), batch["label"].to(device))
        logit_map = model(x)
        loss = loss_function(logit_map, y)
        loss.backward()
        epoch_loss += loss.item()
        optimizer.step()
        optimizer.zero_grad()
        epoch_iterator.set_description("Training (%d / %d Steps) (loss=%2.5f)" % (global_step, max_iterations, loss))
        if (global_step % eval_num == 0 and global_step != 0) or global_step == max_iterations:
            epoch_iterator_val = tqdm(val_loader, desc="Validate (X / X Steps) (dice=X.X)", dynamic_ncols=True)
            dice_val = validation(epoch_iterator_val)# 评价函数
            epoch_loss /= step
            epoch_loss_values.append(epoch_loss)
            metric_values.append(dice_val)
            if dice_val > dice_val_best:
                dice_val_best = dice_val
                global_step_best = global_step
                torch.save(model.state_dict(), os.path.join(root_dir, "new_best_metric_model_seg.pth"))
                print(
                    "Model Was Saved ! Current Best Avg. Dice: {} Current Avg. Dice: {}".format(dice_val_best, dice_val)
                )
            else:
                print(
                    "Model Was Not Saved ! Current Best Avg. Dice: {} Current Avg. Dice: {}".format(
                        dice_val_best, dice_val
                    )
                )
        global_step += 1
    return global_step, dice_val_best, global_step_best

#训练,避免重复运行训练
model_path = os.path.join(root_dir,"new_best_metric_model_seg.pth")
if not os.path.exists(model_path):
    max_iterations = 25000
    eval_num = 2000
    post_label = AsDiscrete(to_onehot=5)#用于评价
    post_pred = AsDiscrete(argmax=True, to_onehot=5)
    dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
    global_step = 0
    dice_val_best = 0.0
    global_step_best = 0
    epoch_loss_values = []
    metric_values = []
    while global_step < max_iterations:
        global_step, dice_val_best, global_step_best = train(global_step, train_loader, dice_val_best, global_step_best)
        # lr_scheduler = ExponentialLR(optimizer, gamma=0.95)
    model.load_state_dict(torch.load(root_dir+"new_best_metric_model_seg.pth",map_location=device))

    print(f"train completed, best_metric: {dice_val_best:.4f} " f"at iteration: {global_step_best}")

#预测部分，验证或者测试数据集，需要保存预测结果
test_dir='/public/pazhou/pazhou_data/preliminary_test/'
if not os.path.exists(root_dir+'segment9'):
    os.mkdir(root_dir+'segment9')#存储预测结果
model.load_state_dict(torch.load(model_path,map_location=device))
model.eval()
with torch.no_grad():
    i=449
    post_processor = AsDiscrete(argmax=True, to_onehot=None)#用于预测的后处理,正确的
    for item in test_ds:
        num_labels=5
        img_name = os.path.split(item["image"].meta["filename_or_obj"])[1]
        img = item["image"].to(device)
        # print(img.shape)
        _, _, original_depth = img.shape[1:]
        img_save = torch.unsqueeze(img, dim=1)
        img_save = img_save.squeeze(0)  # Remove the batch dimension
        img_save = img_save.squeeze(0)
        img_save = img_save.detach().cpu().numpy().astype(np.int16) 
        img_save = zoom(img_save, (512 / img_save.shape[0], 512 / img_save.shape[1], 1), order=0)
        img_save = nib.Nifti1Image(img_save,img.affine)

        print(img_save.shape)
        test_inputs = torch.unsqueeze(img, 1).to(device)
        # print(test_inputs.shape)
        test_outputs = sliding_window_inference(test_inputs, (128,128,16), 4, model, overlap=0.8)
        seg_output=torch.argmax(test_outputs, dim=1)#argmax,correct
        seg_output=seg_output.squeeze(0)
        seg_output_np = seg_output.detach().cpu().numpy().astype(np.int16) #detach是分离的意思
        # unique_labels = np.unique(seg_output_np)
        seg_output_resized = zoom(seg_output_np, (512 / seg_output_np.shape[0], 512 / seg_output_np.shape[1], 1), order=0)
        print(seg_output_resized.shape)
        multi_channel_nifti = nib.Nifti1Image(seg_output_resized,img.affine)
        nifti_shape = multi_channel_nifti.get_fdata().shape
        output_file_path = os.path.join(root_dir, 'segment9', img_name.split('.')[0] + '_mask.nii.gz')
        nib.save(multi_channel_nifti, output_file_path)
        output_file_path_img = os.path.join(root_dir, 'segment9', img_name.split('.')[0] + '.nii.gz')
        nib.save(img_save, output_file_path_img)
        i+=1
        print(i)
        
        
    