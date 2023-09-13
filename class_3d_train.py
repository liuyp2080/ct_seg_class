
import logging
import os
import sys
from pathlib import Path
import glob
import numpy as np
import torch
import pandas as pd
import torch.optim.lr_scheduler as lr_scheduler
import monai
from monai.data import decollate_batch, DataLoader
from monai.metrics import ROCAUCMetric,ConfusionMatrixMetric
from monai.transforms import Activations, AsDiscrete, Compose, LoadImaged, RandRotate90d, Resized, ScaleIntensityd


def classification_train(start_with=1.0, organ='liver',lr=1e-5,threshold=0.5):
    monai.config.print_config()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    data_path = '/work/home/pazhou_045/data_roi9/'#
    train_dir='/public/pazhou/pazhou_data/train/'
    images=[]
    for file in glob.glob(os.path.join(data_path, '*')):
        
        file_name = os.path.basename(file).startswith(str(start_with))

        images.append(file) if file_name else None

    # Define transforms for image
    train_transforms = Compose(
        [
            LoadImaged(keys=["img"], ensure_channel_first=True),
            ScaleIntensityd(keys=["img"]),
            Resized(keys=["img"], spatial_size=(256, 256, 60)),
            # RandRotate90d(keys=["img"], prob=0.8, spatial_axes=[0, 2]),
        ]
    )
    val_transforms = Compose(
        [
            LoadImaged(keys=["img"], ensure_channel_first=True),
            ScaleIntensityd(keys=["img"]),
            Resized(keys=["img"], spatial_size=(384,384, 80)),
        ]
    )
    post_pred = Compose([Activations(softmax=True)])
    post_pred_f1=Compose([AsDiscrete(threshold=0.5)])
    post_label = Compose([AsDiscrete(to_onehot=2)])       
     # Create DenseNet121, CrossEntropyLoss and Adam optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = monai.networks.nets.DenseNet121(spatial_dims=3, in_channels=1, out_channels=2).to(device)
    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
    auc_metric = ROCAUCMetric()
    f1_metric = ConfusionMatrixMetric(metric_name='f1 score')
    epochs=15
    #match the slice and organ
    #  labels for class: 1 and 0
    labels_df=pd.read_csv(train_dir+'train.csv')
    labels = np.array(labels_df[organ], dtype=np.int64)

    train_files = [{"img": img, "label": label} for img, label in zip(images[:360], labels[:360])]
    val_files = [{"img": img, "label": label} for img, label in zip(images[-40:], labels[-40:])]


    # Define dataset, data loader
    check_ds = monai.data.Dataset(data=train_files, transform=train_transforms)
    check_loader = DataLoader(check_ds, batch_size=2, num_workers=4, pin_memory=torch.cuda.is_available())
    check_data = monai.utils.misc.first(check_loader)
    # print(check_data["img"].shape, check_data["label"])

    # create a training data loader
    train_ds = monai.data.Dataset(data=train_files, transform=train_transforms)
    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=8, pin_memory=torch.cuda.is_available())

    # create a validation data loader
    val_ds = monai.data.Dataset(data=val_files, transform=val_transforms)
    val_loader = DataLoader(val_ds, batch_size=2, num_workers=8, pin_memory=torch.cuda.is_available())

    # start a typical PyTorch training
    val_interval = 1
    best_metric = -1
    best_metric_epoch = -1
    for epoch in range(epochs):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{epochs}")
        model.train()
        epoch_loss = 0
        step = 0
        for batch_data in train_loader:
            step += 1
            inputs, labels = batch_data["img"].to(device), batch_data["label"].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_len = len(train_ds) // train_loader.batch_size
            print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")
        epoch_loss /= step
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

        if (epoch + 1) % val_interval == 0:
            model.eval()
            with torch.no_grad():
                y_pred = torch.tensor([], dtype=torch.float32, device=device)
                y = torch.tensor([], dtype=torch.long, device=device)
                for val_data in val_loader:
                    val_images, val_labels = val_data["img"].to(device), val_data["label"].to(device)
                    y_pred = torch.cat([y_pred, model(val_images)], dim=0)
                    y = torch.cat([y, val_labels], dim=0)
                # print(len(y))
                # print(len(y_pred)) 
                acc_value = torch.eq(y_pred.argmax(dim=1), y)
                acc_metric = acc_value.sum().item() / len(acc_value)
                y_onehot = [post_label(i) for i in decollate_batch(y, detach=False)]
                y_pred_act = [post_pred(i) for i in decollate_batch(y_pred)]
                              
                auc_metric(y_pred_act, y_onehot)
                auc_result = auc_metric.aggregate()
                print(auc_result)
                auc_metric.reset()
                y_pred_binary = [post_pred_f1(pred) for pred in y_pred_act]  
                f1_metric(y_pred_binary, y_onehot)
                f1_result = f1_metric.aggregate()
                print(f1_result)
                f1_metric.reset()
                del y_pred_act, y_onehot
                if f1_result[0].item() > best_metric:
                    best_metric = f1_result[0].item()
                    best_metric_epoch = epoch + 1
                    torch.save(model.state_dict(), "best_metric_model_classification3d_{}_{}.pth".format(organ.replace(' ','_'),start_with))#保存模型
                    print("saved new best metric model")
                print(
                    "current epoch: {} current accuracy: {:.4f} current AUC: {:.4f}  best metric: {:.4f} at epoch {}".format(
                        epoch + 1, acc_metric, auc_result, best_metric, best_metric_epoch
                    )
                )
        scheduler.step()




if __name__ == "__main__":
    #获得了model
    root_dir='/work/home/pazhou_045/'
    
    # classification_train(start_with=1.0, organ='liver')
    classification_train(start_with=2.0, organ='spleen',lr=1e-5)#lr是最重要的参数
    # classification_train(start_with=3.0, organ='left kidney',lr=1e-5)
    classification_train(start_with=4.0, organ='right kidney')

