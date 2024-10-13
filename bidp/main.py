import os
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import random_split
import torch.nn.functional as F
from torch.utils.data import Dataset
from PIL import Image
from torch.optim.lr_scheduler import StepLR
import os
from customdataset import CustomDataset
from PECA_U_Net import PECA_U_Net
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

if torch.cuda.is_available():
    # 创建CUDA设备对象
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
# Instantiate the dataset
img_dir = r'F:\cnndata\data\data1\DUTS-TE\images'  # Update with your path
seg_dir = r'F:\cnndata\data\data1\DUTS-TE\GT'  # Update with your path
bian=r"F:\cnndata\data\data1\DUTS-TE\bian"
output_folder2=r"F:\cnndata\newdata\swinduizhao\ecssd"
os.makedirs(output_folder2, exist_ok=True)
# Define your transform for both image and segmentation
class Transform(object):
    def __init__(self,):
        self.transform = transforms.Compose([
            transforms.Resize((352,352)),
            transforms.ToTensor()
        ])
        self.transform1 = transforms.Compose([
            transforms.Resize((352, 352)),
            transforms.Normalize(([0.485, 0.456, 0.406],
                        [0.229, 0.224, 0.225])),
            transforms.ToTensor()
        ])
    def __call__(self, sample):
        image, segmentation ,img_name,bian= sample['image'], sample['segmentation'],sample["filename"],sample["bian"]
        image = self.transform1(image)
        segmentation = self.transform(segmentation)
        bian=self.transform(bian)
        return {'image': image, 'segmentation': segmentation,"filename":img_name,"bian":bian}
transformed_dataset = CustomDataset(img_dir=img_dir, seg_dir=seg_dir,bian=bian,transform=Transform())
def iou(pred, labels, num_classes):
    """
    Calculate Intersection over Union (IoU) for each class and return the average IoU.

    Args:
        pred (torch.Tensor): Model predictions, shape (N, H, W), where N is batch size,
                             each element in tensor should be class index for each pixel.
        labels (torch.Tensor): Ground truth labels, shape (N, H, W), each element should
                               be class index for each pixel.
        num_classes (int): Number of classes in the segmentation task.

    Returns:
        mean_iou (float): The mean IoU over all classes.
    """
    # Initialize list to store IoU for all classes
    ious = []

    # Avoid division by zero in IoU computation
    eps = 1e-6

    # Iterate over each class
    for cls in range(num_classes):
        # Intersection: True Positive (TP)
        intersection = ((pred == cls) & (labels == cls)).float().sum()

        # Union: TP + False Positive (FP) + False Negative (FN)
        union = ((pred == cls) | (labels == cls)).float().sum()

        # IoU: Intersection over Union
        iou = (intersection + eps) / (union + eps)  # Added epsilon to avoid division by zero

        # Append the IoU to the list
        ious.append(iou.item())

    # Compute the mean IoU by averaging over all classes
    mean_iou = sum(ious) / len(ious)
    return mean_iou

train_loader = DataLoader(transformed_dataset, batch_size=6, shuffle=True)
test_loader = DataLoader(transformed_dataset, batch_size=1, shuffle=True)


cnn_model = PECA_U_Net(in_chans=3,num_classes=2)
cnn_model=cnn_model.to(device)



criterion = nn.CrossEntropyLoss()
optimizer_cnn = torch.optim.SGD(cnn_model.parameters(), lr=0.005,momentum=0.90)
num_epochs =65

for epoch in range(num_epochs):
    cnn_model.train()

    # 初始化累积损失
    total_loss_cnn_accumulated = 0.0
    num_batches = 0  # 追踪批次的数量

    for i, batch in enumerate(train_loader):
        inputs = batch['image']
        labels = batch['segmentation']
        edge_dir = batch["bian"]
        inputs = inputs.to(device)
        labels = labels.to(device)
        edge_dir = edge_dir.to(device)

        outputs_cnn=cnn_model(inputs)

        # 初始化批次的总损失
        total_loss_cnn = 0.0
        loss_cnn_labeled = criterion(outputs_cnn,labels.squeeze().long())
        total_loss_cnn += loss_cnn_labeled
        optimizer_cnn.zero_grad()

        loss =  total_loss_cnn  # 确保这是标量
        loss.backward()  # 先计算梯度
        optimizer_cnn.step()

        # 累积总损失用于报告
        total_loss_cnn_accumulated += total_loss_cnn.item()
        num_batches += 1

    # 每个epoch结束后输出平均损失
    print(
        f"Epoch {epoch + 1}, Average Total Loss CNN: {total_loss_cnn_accumulated / num_batches:.4f}")



p=[]

cnn_model.eval()
for j, batch in enumerate((test_loader)):
    image = batch["image"]
    segment_image = batch["segmentation"]
    finame = batch["filename"]
    image, segment_image = image.to(device), segment_image.to(device)

    out = cnn_model(image)
    out = torch.argmax(out, dim=1, keepdim=True)
    a = iou(out, segment_image, num_classes=2)
    print(a)
    p.append(a)
    image = out.squeeze()  # 现在 image 的形状应为 (224, 224)
    image = image.cpu()
    # 转换为PIL图像
    # 注意：我们假设这个张量是在0到1之间，如果是0到255，应先转换为uint8
    image = image.mul(255).byte()  # 转换为0-255的范围
    image = Image.fromarray(image.cpu().numpy(), 'L')  # 'L' 模式表示灰度图像

    # 保存图像
    par, parts = finame[0].split('.')
    par = par + ".png"
    image.save(os.path.join(output_folder2, par))
print("cnn的iou是",sum(p)/len(p))
