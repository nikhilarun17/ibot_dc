import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

DATASET_DIR = "/home/nikhil/.cache/kagglehub/datasets/moazeldsokyx/dogs-vs-cats/versions/1/dataset"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

# Basic setup (Freeze layers, modify final layer, loss function, optimizer)
for param in model.parameters():
    param.requires_grad = False
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 2)
model = model.to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Data Augmentation and Normalization
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

# Learning rate scheduler 
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

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

num_epochs = 5
best_val_acc = 0.0

train_losses = []
val_losses = []
train_accs = []
val_accs = []

#Training Loop

for epoch in range(num_epochs):
    print(f"\nEpoch {epoch+1}/{num_epochs}")

    # Training Phase
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_dl:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)              
        loss = criterion(outputs, labels)    

        loss.backward()                     
        optimizer.step()                     

        running_loss += loss.item()

        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    train_loss = running_loss / len(train_dl)
    train_acc = correct / total

    train_losses.append(train_loss)
    train_accs.append(train_acc)
    
    # Validation Phase
    model.eval()
    val_running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_dl:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            val_running_loss += loss.item()

            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    val_loss = val_running_loss / len(val_dl)
    val_acc = correct / total

    val_losses.append(val_loss)
    val_accs.append(val_acc)

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "best_resnet18.pth")
        print("Saved new best model")

    if scheduler is not None:
        scheduler.step()

    print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
    print(f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f}")

# Plot training curves
def plot_training_curves(train_losses, val_losses, train_accs, val_accs):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss plot
    epochs = range(len(train_losses))

    ax1.plot(epochs, train_losses, label='Train Loss')
    ax1.plot(epochs, val_losses, label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()


    # Accuracy plot
    ax2.plot(train_accs, label='Train Acc')
    ax2.plot(val_accs, label='Val Acc')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()

    plt.tight_layout()
    plt.savefig('training_curves.png')
    plt.show()

plot_training_curves(train_losses, val_losses, train_accs, val_accs)

# Evaluate model and plot confusion matrix
def evaluate_and_confusion_matrix(model, dataloader, device, class_names):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    print(f'Validation Accuracy: {acc:.4f}')

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    plt.show()  

evaluate_and_confusion_matrix(model, val_dl, device, class_names=train_ds.classes)