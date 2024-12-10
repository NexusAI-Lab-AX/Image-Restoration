import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import cv2
import numpy as np
import os
from tqdm import tqdm
import zipfile
from skimage.metrics import structural_similarity as ssim

# GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 모델 정의
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        # 인코더
        self.enc1 = self.conv_block(3, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)
        self.enc5 = self.conv_block(512, 1024)

        # 디코더
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec1 = self.conv_block(1024 + 512, 512)

        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec2 = self.conv_block(512 + 256, 256)

        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec3 = self.conv_block(256 + 128, 128)

        self.up4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec4 = self.conv_block(128 + 64, 64)

        self.final = nn.Conv2d(64, 3, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(nn.MaxPool2d(2)(e1))
        e3 = self.enc3(nn.MaxPool2d(2)(e2))
        e4 = self.enc4(nn.MaxPool2d(2)(e3))
        e5 = self.enc5(nn.MaxPool2d(2)(e4))

        d1 = self.dec1(torch.cat([self.up1(e5), e4], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d1), e3], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d2), e2], dim=1))
        d4 = self.dec4(torch.cat([self.up4(d3), e1], dim=1))

        return torch.sigmoid(self.final(d4))

# 랜덤 마스크 생성
def create_random_mask(image_shape):
    """
    주어진 이미지 크기에서 랜덤 다각형 마스크를 생성
    """
    height, width = image_shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)

    # 다각형 개수 (랜덤)
    num_polygons = np.random.randint(1, 4)

    for _ in range(num_polygons):
        # 다각형 정점 개수
        num_points = np.random.randint(3, 8)
        points = np.array([
            [
                np.random.randint(0, width),
                np.random.randint(0, height)
            ]
            for _ in range(num_points)
        ])
        # 다각형 > 마스크에 추가
        cv2.fillPoly(mask, [points], color=255)

    return mask

# 데이터셋 클래스 정의
class ImageDataset(Dataset):
    def __init__(self, input_dir, gt_dir, use_mask=True):
        self.input_dir = input_dir
        self.gt_dir = gt_dir
        self.input_images = sorted(os.listdir(input_dir))
        self.gt_images = sorted(os.listdir(gt_dir))
        self.use_mask = use_mask 

    def __len__(self):
        return len(self.input_images)

    def __getitem__(self, idx):
        input_path = os.path.join(self.input_dir, self.input_images[idx])
        gt_path = os.path.join(self.gt_dir, self.gt_images[idx])
        
        # 손상된 입력 이미지
        input_image = cv2.imread(input_path)  # train_input
        gt_image = cv2.imread(gt_path)       # train_gt
        
        if self.use_mask:
            mask = create_random_mask(gt_image.shape)
            # train_gt에 마스크를 적용하여 손상 이미지 생성
            masked_image = cv2.cvtColor(gt_image, cv2.COLOR_BGR2GRAY)
            masked_image = cv2.bitwise_and(masked_image, masked_image, mask=cv2.bitwise_not(mask))
            masked_image = np.expand_dims(masked_image, axis=-1)
            masked_image = np.repeat(masked_image, 3, axis=-1)
        else:
            mask = np.zeros(gt_image.shape[:2], dtype=np.uint8)
            masked_image = input_image
        
        # Tensor 변환 및 정규화
        input_tensor = torch.tensor(masked_image).permute(2, 0, 1).float() / 255.0
        gt_tensor = torch.tensor(gt_image).permute(2, 0, 1).float() / 255.0
        mask_tensor = torch.tensor(mask).unsqueeze(0).float() / 255.0

        return input_tensor, gt_tensor, mask_tensor

# train_input 데이터셋
input_dataset = ImageDataset("train_input", "train_gt", use_mask=False)
# 랜덤 마스크 데이터셋
masked_dataset = ImageDataset("train_input", "train_gt", use_mask=True)

# 데이터셋 결합
from torch.utils.data import ConcatDataset
combined_dataset = ConcatDataset([input_dataset, masked_dataset])
train_loader = DataLoader(combined_dataset, batch_size=64, shuffle=True, num_workers=48, pin_memory=True)

# 모델 초기화
image_model = UNet().to(device)
image_model = nn.DataParallel(image_model, device_ids=[0, 1])

image_optimizer = optim.AdamW(image_model.parameters(), lr=0.0001)
criterion = nn.MSELoss()


# 학습 루프
for epoch in range(120):
    if epoch == 40:  # 40에폭부터 학습률 변경
        for param_group in image_optimizer.param_groups:
            param_group['lr'] = 0.00001
        tqdm.write(f"Epoch {epoch+1}: Learning Rate changed to {param_group['lr']}")

    image_model.train()
    running_loss = 0.0
    
    with tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/120", unit="batch") as pbar:
        for input_images, gt_images, masks in train_loader:
            input_images, gt_images, masks = input_images.to(device), gt_images.to(device), masks.to(device)
            
            image_optimizer.zero_grad()
            image_outputs = image_model(input_images)
            image_loss = criterion(image_outputs, gt_images)
            image_loss.backward()
            image_optimizer.step()

            running_loss += image_loss.item()
            pbar.set_postfix(loss=image_loss.item())
            pbar.update(1)

    avg_loss = running_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/120], Average Loss: {avg_loss:.4f}")

    # 평가 및 결과 저장
    image_model.eval()

    with torch.no_grad():
        test_output_dir = f"result/epoch_{epoch+1}"
        os.makedirs(test_output_dir, exist_ok=True)

        for img_name in sorted(os.listdir("test_input")):
            img_path = os.path.join("test_input", img_name)
            input_image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            input_image = np.expand_dims(input_image, axis=-1) / 255.0
            input_tensor = torch.tensor(input_image).permute(2, 0, 1).unsqueeze(0).float().to(device)
            
            restored_image = image_model(input_tensor.expand(-1, 3, -1, -1)).squeeze().permute(1, 2, 0).cpu().numpy() * 255
            restored_image = restored_image.astype(np.uint8)
            cv2.imwrite(os.path.join(test_output_dir, img_name), restored_image)

    # 에폭 결과를 ZIP 파일로 저장
    zip_path = f"result/epoch_{epoch+1}.zip"
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        for root, _, files in os.walk(test_output_dir):
            for file in files:
                zipf.write(os.path.join(root, file), arcname=os.path.join(os.path.basename(root), file))
    print(f"Epoch {epoch+1} result saved to {zip_path}")

print("Training complete.")

