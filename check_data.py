import pandas as pd 
import numpy as np
train_df=pd.read_csv('/public/pazhou/pazhou_data/train/train.csv')

print(train_df['liver'].value_counts())
print(train_df['spleen'].value_counts())
print(train_df['left kidney'].value_counts())
print(train_df['right kidney'].value_counts())

import nibabel as nib

# 读取NIfTI文件train
nifti_img = nib.load('/work/home/pazhou_045/segment9/case450.nii.gz')
#test
pred_img=nib.load('/work/home/pazhou_045/segment9/case450_mask.nii.gz')
# 打印NIfTI文件的头信息
# print(nifti_img.header)
print(pred_img.header)
# 打印NIfTI文件的数据形状
print(nifti_img.shape)
print(pred_img.shape)
# 打印NIfTI文件的像素尺寸
print(nifti_img.header.get_zooms())
print(pred_img.header.get_zooms())
# 打印NIfTI文件的数据类型
print(nifti_img.header.get_data_dtype())
print(pred_img.header.get_data_dtype())
msk='/public/pazhou/pazhou_data/train/mask/case12_mask.nii.gz'
msk_img=nib.load(msk)
data_train=msk_img.get_fdata()
data_pred=pred_img.get_fdata()
print(msk_img.get_fdata().shape)
print(msk_img.header)
print(np.unique(data_pred))  # 打印数据中的唯一值（标签）
print(np.unique(data_train))  # 打印数据中的唯一值（标签）
# result_df=pd.read_csv('result.csv')
# print(result_df)
# print(result_df['ID'])