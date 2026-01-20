#Bonus 1

"""import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATASET_DIR = "/home/nikhil/.cache/kagglehub/datasets/moazeldsokyx/dogs-vs-cats/versions/1/dataset"
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
model.fc = nn.Linear(model.fc.in_features, 2)

model.load_state_dict(torch.load("best_resnet18.pth"))
model = model.to(device)

for param in model.parameters():
    param.requires_grad = False
for param in model.layer4.parameters():
    param.requires_grad = True

for param in model.fc.parameters():
    param.requires_grad = True

optimizer = optim.Adam([{'params': model.layer4.parameters(), 'lr': 0.0001},{'params': model.fc.parameters(), 'lr': 0.0001}])
criterion = nn.CrossEntropyLoss()

train_transforms = transforms.Compose([
    transforms.Resize(256),                    # 1
    transforms.RandomResizedCrop(224),         # 2
    transforms.RandomHorizontalFlip(),          # 3
    transforms.RandomRotation(15),              # 4
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2
    ),                                         # 5
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

val_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

# Training and Validation DataLoaders
train_ds= datasets.ImageFolder(
    root=f"{DATASET_DIR}/train",
    transform=train_transforms
)
train_dl = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=4)

val_ds= datasets.ImageFolder(
    root=f"{DATASET_DIR}/validation",
    transform=val_transforms
)
val_dl = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=4)

model.train()
epochs = 5

for epoch in range(epochs):
    running_loss = 0.0

    for images, labels in train_dl:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_dl):.4f}")

# Validation Loop
model.eval()
correct = 0
total = 0   
with torch.no_grad():
    for images, labels in val_dl:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
accuracy = 100 * correct / total
print(f'Validation Accuracy: {accuracy:.2f}%')

torch.save(model.state_dict(), "resnet18_finetuned.pth")
"""

#Bonus 2

"""import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATASET_DIR = "/home/nikhil/.cache/kagglehub/datasets/moazeldsokyx/dogs-vs-cats/versions/1/dataset"
model1 = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
model1.fc = nn.Linear(model1.fc.in_features, 2)
model1 = model1.to(device)

model2 = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
model2.classifier[1] = nn.Linear(model2.classifier[1].in_features, 2)
model2 = model2.to(device)  

for param in model1.parameters():
    param.requires_grad = False
model1.fc.requires_grad_(True)

for param in model2.parameters():
    param.requires_grad = False
for param in model2.classifier.parameters():
    param.requires_grad = True

train_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])
val_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

# Training and Validation DataLoaders
train_ds= datasets.ImageFolder(
    root=f"{DATASET_DIR}/train",
    transform=train_transforms
)
train_dl = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=4)

val_ds= datasets.ImageFolder(
    root=f"{DATASET_DIR}/validation",
    transform=val_transforms
)
val_dl = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=4)
criterion = nn.CrossEntropyLoss()
optimizer1 = optim.Adam(model1.parameters(), lr=0.001)
optimizer2 = optim.Adam(model2.parameters(), lr=0.001)

epochs = 5

for epoch in range(epochs):
    model1.train()
    model2.train()
    running_loss1 = 0.0
    running_loss2 = 0.0

    for images, labels in train_dl:
        images, labels = images.to(device), labels.to(device)

        # Train model1
        optimizer1.zero_grad()
        outputs1 = model1(images)
        loss1 = criterion(outputs1, labels)
        loss1.backward()
        optimizer1.step()
        running_loss1 += loss1.item()

        # Train model2
        optimizer2.zero_grad()
        outputs2 = model2(images)
        loss2 = criterion(outputs2, labels)
        loss2.backward()
        optimizer2.step()
        running_loss2 += loss2.item()

    print(f"Epoch [{epoch+1}/{epochs}], Model1 Loss: {running_loss1/len(train_dl):.4f}, Model2 Loss: {running_loss2/len(train_dl):.4f}")

# Validation Loop
model1.eval()
model2.eval()
correct1 = 0
correct2 = 0
total = 0   
with torch.no_grad():   
    for images, labels in val_dl:
        images, labels = images.to(device), labels.to(device)
        outputs1 = model1(images)
        outputs2 = model2(images)
        _, predicted1 = torch.max(outputs1.data, 1)
        _, predicted2 = torch.max(outputs2.data, 1)
        correct1 += (predicted1 == labels).sum().item()
        correct2 += (predicted2 == labels).sum().item()
        total += labels.size(0)
accuracy1 = correct1 / total if total > 0 else 0
accuracy2 = correct2 / total if total > 0 else 0

        
print(f'Model1 Validation Accuracy: {accuracy1*100:.2f}%')
print(f'Model2 Validation Accuracy: {accuracy2*100:.2f}%')
if accuracy1 > accuracy2:
    print("Model1 (ResNet18) performs better.")
else:
    print("Model2 (MobileNetV2) performs better.")"""
        

#Bonus 3

"""import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model.eval()

def denormalize(img_tensor):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    img = img_tensor.cpu().numpy().transpose((1, 2, 0))
    img = std * img + mean
    img = np.clip(img, 0, 1)
    return img


correct_samples = []
incorrect_samples = []

class_names = val_ds.classes  # ['cats', 'dogs']

with torch.no_grad():
    for images, labels in val_dl:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)

        for i in range(images.size(0)):
            sample = (
                images[i],
                labels[i].item(),
                preds[i].item()
            )

            if preds[i] == labels[i] and len(correct_samples) < 10:
                correct_samples.append(sample)

            elif preds[i] != labels[i] and len(incorrect_samples) < 10:
                incorrect_samples.append(sample)

            if len(correct_samples) == 10 and len(incorrect_samples) == 10:
                break

        if len(correct_samples) == 10 and len(incorrect_samples) == 10:
            break

fig, axes = plt.subplots(2, 10, figsize=(20, 5))

# Correct predictions
for i, (img, label, pred) in enumerate(correct_samples):
    axes[0, i].imshow(denormalize(img))
    axes[0, i].set_title(
        f"P: {class_names[pred]}\nT: {class_names[label]}",
        fontsize=9
    )
    axes[0, i].axis("off")

# Incorrect predictions
for i, (img, label, pred) in enumerate(incorrect_samples):
    axes[1, i].imshow(denormalize(img))
    axes[1, i].set_title(
        f"P: {class_names[pred]}\nT: {class_names[label]}",
        fontsize=9,
        color="red"
    )
    axes[1, i].axis("off")

axes[0, 0].set_ylabel("Correct", fontsize=12)
axes[1, 0].set_ylabel("Incorrect", fontsize=12)

plt.tight_layout()
plt.show()"""

#Note:
# Use bonus 1 code to fine-tune and save the model as "best_resnet18.pth"
# Use bonus 3 code to visualize correct and incorrect predictions after training
# Use bonus 3 with bonus 1 code to plot correct and incorrect predictions

# Use bonus 2 code to compare ResNet18 and MobileNetV2 performance on the same dataset (use it alone)