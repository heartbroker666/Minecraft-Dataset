import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import os
import time

# 定义你的本地数据集类
class Datasets(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []  # 存储所有图片路径
        self.labels = []  # 存储对应的标签

        # 遍历数据集文件夹，获取所有图片路径和对应标签
        for label, class_folder in enumerate(os.listdir(root_dir)):
            class_folder_path = os.path.join(root_dir, class_folder)
            for image_name in os.listdir(class_folder_path):
                image_path = os.path.join(class_folder_path, image_name)
                self.image_paths.append(image_path)
                self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        # 读取图像文件
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

# 数据转换
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# 创建数据集实例
dataset = Datasets(root_dir='./data/MC_datasets', transform=transform)

# 分割数据集为训练集和测试集
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

# 创建数据加载器
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 定义卷积神经网络模型
class CustomCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)  # 输入图片是 RGB 三通道
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 7 * 7, 256)  # 输入图片大小是 28x28
        self.fc2 = nn.Linear(256, 30)  # 有30个类别

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 32 * 7 * 7)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 训练和评估模型的函数
def train_and_evaluate(model, train_loader, test_loader, epochs, learning_rate):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    LOSS = []
    criterion = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)  # 将数据移动到GPU
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)

        train_loss = train_loss / len(train_loader.dataset)
        LOSS.append(train_loss)
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}")
    # 训练完毕
    torch.save(model, 'Minecraft_CNN_Model')
    print('Model has been saved succeed!')

    # 绘制损失函数图像
    plt.figure(figsize=(10, 8))
    plt.plot(LOSS)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss over Epochs with Minecraft_datasets')
    plt.savefig('plot_loss_Minecraft.png')
    plt.show()
    # 在测试集上评估模型
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    print(f"Accuracy on test set: {100 * accuracy:.2f}%")

# 实例化模型并训练
model = CustomCNN()
epochs = 100
learning_rate = 0.001
train_and_evaluate(model, train_loader, test_loader, epochs, learning_rate)

