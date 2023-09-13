#将segment9 文件夹中的_mask 文件拷贝到新的文件夹segment10
import os
import shutil

source_folder = "segment9"
target_folder = "segment10"

# 创建目标文件夹
os.makedirs(target_folder, exist_ok=True)

# 遍历源文件夹中的文件
for file_name in os.listdir(source_folder):
    # 如果文件名以"_mask"结尾，则进行拷贝
    if file_name.endswith("_mask.nii.gz"):
        source_file = os.path.join(source_folder, file_name)
        target_file = os.path.join(target_folder, file_name)
        shutil.copy2(source_file, target_file)