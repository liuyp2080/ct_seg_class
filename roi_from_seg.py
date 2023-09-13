

from glob import glob
import numpy as np
from pathlib import Path
import numpy as np
import nibabel as nib
from scipy import ndimage
import os
import nibabel as nib
import numpy as np

def segment_image_with_roi(segmentation_file, raw_image_file, output_dir):
    img = nib.load(segmentation_file)
    segmentation_data = img.get_fdata()
    unique_labels = np.unique(segmentation_data)
    rois = []
    for label in unique_labels:
        if label != 0:#去掉了background
            labeled_mask = segmentation_data == label
            indices = np.nonzero(labeled_mask)
            if indices[0].size > 0:
                rois.append({
                        'label': label,#0,1,2,3
                        'indices': indices,
                    })

    sorted_rois=rois
    print(len(rois))
    raw_image = nib.load(raw_image_file).get_fdata()
    file_name=Path(segmentation_file).stem
    for roi in sorted_rois:#
        roi_indices = roi['indices']
        label=roi['label']        
        mask = np.zeros_like(raw_image)
        mask[roi_indices] = 1
        roi_image = raw_image * mask
        output_filename = f"{label}_{file_name}.gz"
        output_path = os.path.join(output_dir, output_filename)
        nib.save(nib.Nifti1Image(roi_image, img.affine), output_path)

    print(f"ROI images saved to {output_dir}")
    
    

if __name__ == "__main__":  
    train_dir='/public/pazhou/pazhou_data/train/'
    test_dir='/work/home/pazhou_045/segment9/'
    destiny_dir='/work/home/pazhou_045/'
    
    if not os.path.exists(destiny_dir+'data_roi9'):
        os.makedirs(destiny_dir+'data_roi9')
    output_dir=destiny_dir+'data_roi9'
    for i in range(400):
        segmentation_file_name='case{}_mask.nii.gz'.format(i+1)
        raw_image_file_name='case{}.nii.gz'.format(i+1)
        segmentation_file=train_dir+'mask/'+segmentation_file_name
        raw_image_file=train_dir+'data/'+raw_image_file_name
        segment_image_with_roi(segmentation_file, raw_image_file, output_dir)
    if not os.path.exists(destiny_dir+'test_data_roi9'):#测试数据
        os.makedirs(destiny_dir+'test_data_roi9')
    output_dir=destiny_dir+'test_data_roi9'
    for i in range(449,549):
        segmentation_file_name='case{}_mask.nii.gz'.format(i+1)
        raw_image_file_name='case{}.nii.gz'.format(i+1)
        segmentation_file=os.path.join(test_dir,segmentation_file_name)
        raw_image_file=os.path.join(test_dir,raw_image_file_name)
        segment_image_with_roi(segmentation_file, raw_image_file, output_dir)
    
    
