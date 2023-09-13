# ct_seg_class
tasks of segmentation and classification of CT images
代码文件说明：

seg_3d,完成分割任务,包括模型的训练和测试集的mask预测,形成segment9；
roi_from_seg,根据mask区域提取原始图片中各个器官的图片，为下一步训练分类模型做准备；
class_3d_train, 使用上述提取到的数据训练四个分类模型，每个器官一个；
class_3d_test,预测测试集的图片，形成result.csv
zip，将segment和result.csv 形成submit.zip

copy_seg, 将segment9 内测试集的mask图片单独拷贝到segment10内，为了提交结果。

