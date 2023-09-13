

import logging
import os
import sys
import glob
import numpy as np
import torch
import pandas as pd
import monai
from monai.data import CSVSaver, ImageDataset, DataLoader
from monai.transforms import EnsureChannelFirst, Compose, Resized, ScaleIntensityd, LoadImaged
def test(start_with=3, organ='liver'):
    monai.config.print_config()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    root_dir = "/work/home/pazhou_045/"
    data_dir = os.path.join(root_dir,'test_data_roi11')
    images=[]
    for file in glob.glob(os.path.join(data_dir, '*')): 
        
        file_name = os.path.basename(file).startswith(str(start_with))

        images.append(file) if file_name else None

    val_files = [{"img": img} for img in images]
     # Define transforms for image
    val_transforms = Compose(
        [
            LoadImaged(keys=["img"], ensure_channel_first=True),
            ScaleIntensityd(keys=["img"]),
            Resized(keys=["img"], spatial_size=(128, 128, 32)),
        ]
    )

    # create a validation data loader
    val_ds = monai.data.Dataset(data=val_files, transform=val_transforms)
    val_loader = DataLoader(val_ds, batch_size=2, num_workers=4, pin_memory=torch.cuda.is_available())

    # Create DenseNet121
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = monai.networks.nets.DenseNet121(spatial_dims=3, in_channels=1, out_channels=2).to(device)

    model.load_state_dict(torch.load("best_metric_model_classification3d_{}_{}.pth".format(organ.replace(' ','_'),start_with)))
    model.eval()


    results_df = pd.DataFrame(columns=["ID", organ])
    with torch.no_grad():
        for val_data in val_loader:
            # print(val_data['img_meta_dict'].keys())
            val_images= val_data["img"].to(device)
            val_preds = model(val_images).argmax(dim=1)
            file_names = [os.path.basename(file).split('_')[1] for file in val_data['img_meta_dict']['filename_or_obj']]
            batch_result=pd.DataFrame({
                'ID':file_names,
                organ:val_preds.cpu().numpy()
            })
            results_df = pd.concat([results_df, batch_result])
    return results_df

if __name__ == "__main__":
    id=['case{}'.format(i) for i in range(450,550)]
    basic_df=pd.DataFrame({'ID':id})
    result_liver=test(start_with=1, organ='liver')
    # print(result_liver)
    result_spleen=test(start_with=1, organ='spleen')
    # print(result_spleen)
    result_left_kidney=test(start_with=1, organ='left kidney')
    # print(result_left_kidney)
    result_right_kidney=test(start_with=1, organ='right kidney')
    # print(result_right_kidney)
    result=pd.merge(basic_df,result_liver)
    result=pd.merge(result,result_spleen)
    result=pd.merge(result,result_left_kidney)
    result=pd.merge(result,result_right_kidney)
    result=result.fillna(value=0)
    result.to_csv('result.csv',index=False)
    # print(result)
