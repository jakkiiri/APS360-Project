# APS360 Skin Disease Detection - Full Pipeline with Fine-Tuning, Augmentation, Balanced Sampling
from charset_normalizer import is_binary
from tqdm import tqdm

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import models, transforms
from torchvision.models import MobileNet_V2_Weights, ResNet50_Weights, EfficientNet_B0_Weights
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from glob import glob
import cv2
import random

# === Paths and Device ===
split_dir = r'C:\Users\shore\Desktop\APS360\Datasets\DataSplit'
binary_cache_root = r'C:\Users\shore\Desktop\APS360\Datasets\Cache_Binary'
multi_cache_root = r'C:\Users\shore\Desktop\APS360\Datasets\Cache_Multi'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Label Mapping ===
label_mapping = {
    'nevus': 0, 'melanoma': 1, 'bcc': 2, 'seborrheic_keratosis': 3,
    'actinic_keratosis': 4, 'scc': 5, 'dermatofibroma': 6,
    'lentigo': 7, 'vascular_lesion': 8
}

# === Transformations ===
base_transform = transforms.Compose([
    transforms.Resize((320, 320)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])
no_aug_transform = transforms.Compose([
    transforms.Resize((320, 320)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# === Hair Removal ===
def apply_hair_removal(image_bgr, threshold=10, kernel_size=17):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    _, mask = cv2.threshold(blackhat, threshold, 255, cv2.THRESH_BINARY)
    return cv2.inpaint(image_bgr, mask, 1, cv2.INPAINT_TELEA)

def pil_to_cv2(pil_img):
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

def cv2_to_pil(cv_img):
    return Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))

# === Build Cache with Oversampling and Undersampling ===
def build_cache(split_dir, cache_root, transform, oversample_factor=2, undersample_factor=0.5, is_binary=False):
    os.makedirs(cache_root, exist_ok=True)
    for phase in ['train', 'val', 'test']:
        split_path = os.path.join(split_dir, phase)
        cache_path = os.path.join(cache_root, phase)
        os.makedirs(cache_path, exist_ok=True)
        print(f"\n📦 Caching {phase} data...")

        for class_name in os.listdir(split_path):
            class_dir = os.path.join(split_path, class_name)
            if not os.path.isdir(class_dir): continue
            class_cache_dir = os.path.join(cache_path, class_name)
            os.makedirs(class_cache_dir, exist_ok=True)
            label = label_mapping[class_name]

            images = [f for f in os.listdir(class_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
            img_paths = [os.path.join(class_dir, f) for f in images]
            # Switch transform if binary model (no augmentation)
            effective_transform = no_aug_transform if is_binary else transform


            if phase == 'train':
                if class_name == 'nevus':
                    img_paths = random.sample(img_paths, int(len(img_paths) * undersample_factor))
                elif len(img_paths) < 100:  # small class
                    img_paths *= oversample_factor

            for i, img_path in enumerate(tqdm(img_paths, desc=f"{phase}/{class_name}")):
                try:
                    img = Image.open(img_path).convert("RGB")
                    img_cv = pil_to_cv2(img)
                    img_no_hair = apply_hair_removal(img_cv)
                    img_pil = cv2_to_pil(img_no_hair)

                    img_tensor = effective_transform(img_pil)

                    fname = os.path.basename(img_path)
                    if i >= len(images):  # synthetic sample
                        fname = f"aug_{i}_{fname}"
                    torch.save({'image': img_tensor, 'label': label},
                               os.path.join(class_cache_dir, fname + '.pt'))
                except Exception as e:
                    print(f"❌ Failed on {img_path}: {e}")

# === Cached Dataset ===
class CachedDataset(Dataset):
    def __init__(self, cache_dir):
        self.files = sorted(glob(os.path.join(cache_dir, '*', '*.pt')))
        self.labels = [torch.load(f, map_location='cpu', weights_only=True)['label'] for f in self.files]


    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = torch.load(self.files[idx], map_location='cpu', weights_only=True)
        image = data['image']
        label = data['label']
        binary_label = 0 if label == 0 else 1
        return image, binary_label, label

# === DataLoaders ===
def get_loaders(train_dir, val_dir, batch_size=32):
    train_dataset = CachedDataset(train_dir)
    val_dataset = CachedDataset(val_dir)

    train_labels = [lbl for _, _, lbl in train_dataset]
    binary_labels = [0 if l == 0 else 1 for l in train_labels]

    class_counts = np.bincount(train_labels)
    weights = 1. / torch.tensor(class_counts, dtype=torch.float)
    global multi_class_weights
    multi_class_weights = weights / weights.sum() * len(class_counts)

    class_sample_count = np.array([binary_labels.count(0), binary_labels.count(1)])
    class_weights = np.array([1. / c if c > 0 else 0.0 for c in class_sample_count])
    samples_weight = np.array([class_weights[t] for t in binary_labels])
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight), replacement=True)

    num_nevus = binary_labels.count(0)
    num_non_nevus = len(binary_labels) - num_nevus
    global binary_class_weight
    binary_class_weight = torch.tensor([num_non_nevus / num_nevus if num_nevus > 0 else 1.0])

    return (
        DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=4, pin_memory=True),
        DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    )

# === Backbone Extractor ===
class BackboneExtractor(nn.Module):
    def __init__(self, backbone, backbone_name):
        super().__init__()
        self.backbone_name = backbone_name
        self.backbone = backbone

    def forward(self, x):
        if self.backbone_name == 'mobilenet':
            x = self.backbone.features(x)
        elif self.backbone_name == 'resnet':
            x = self.backbone.conv1(x)
            x = self.backbone.bn1(x)
            x = self.backbone.relu(x)
            x = self.backbone.maxpool(x)
            x = self.backbone.layer1(x)
            x = self.backbone.layer2(x)
            x = self.backbone.layer3(x)
            x = self.backbone.layer4(x)
        elif self.backbone_name == 'efficientnet':
            x = self.backbone.features(x)
        else:
            raise ValueError("Unsupported backbone")
        x = F.adaptive_avg_pool2d(x, (1, 1))
        return x.view(x.size(0), -1)

# === Classifiers ===
class BinaryClassifier(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)


class MultiClassifier(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 9)  # 9 classes for multi-class classification
        )

    def forward(self, x):
        return self.net(x)

# === Training Function ===
def train_model(backbone, classifier, train_loader, val_loader, mode='binary', num_epochs=50, patience=7, backbone_name='model'):
    backbone = backbone.to(device)
    classifier = classifier.to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=binary_class_weight.to(device)) if mode == 'binary' else nn.CrossEntropyLoss(weight=multi_class_weights.to(device))
    optimizer = torch.optim.Adam(list(classifier.parameters()) + list(backbone.parameters())[-10:], lr=0.0001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=patience, factor=0.5, verbose=True)

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs} [{mode}]")
        classifier.train()
        backbone.train()
        train_loss = 0
        train_preds, train_labels = [], []

        for inputs, binary_targets, multi_targets in tqdm(train_loader, desc="Training"):
            inputs = inputs.to(device)
            targets = binary_targets.to(device).float() if mode == 'binary' else multi_targets.to(device)

            optimizer.zero_grad()
            features = backbone(inputs)
            outputs = classifier(features)

            loss = criterion(outputs.squeeze() if mode == 'binary' else outputs, targets)
            if torch.isnan(loss): return print("NaN detected.")

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            pred = (torch.sigmoid(outputs).squeeze() > 0.5).int() if mode == 'binary' else torch.argmax(outputs, dim=1)
            train_preds.extend(pred.cpu().numpy())
            train_labels.extend(targets.cpu().numpy())

        avg_train_loss = train_loss / len(train_loader)

        # --- Validation ---
        classifier.eval()
        backbone.eval()
        val_loss = 0
        val_preds, val_labels = [], []
        val_probs = []  # For multi-class probabilities

        with torch.no_grad():
            for inputs, binary_targets, multi_targets in tqdm(val_loader, desc="Validation"):
                inputs = inputs.to(device)
                targets = binary_targets.to(device).float() if mode == 'binary' else multi_targets.to(device)
                features = backbone(inputs)
                outputs = classifier(features)
                loss = criterion(outputs.squeeze() if mode == 'binary' else outputs, targets)
                val_loss += loss.item()
                if mode == 'binary':
                    pred = (torch.sigmoid(outputs).squeeze() > 0.5).int()
                    val_preds.extend(pred.cpu().numpy())
                    val_labels.extend(targets.cpu().numpy())
                else:
                    probs = F.softmax(outputs, dim=1)
                    pred = torch.argmax(probs, dim=1)
                    val_preds.extend(pred.cpu().numpy())
                    val_labels.extend(targets.cpu().numpy())
                    val_probs.extend(probs.cpu().numpy())  # ✅ This is what was missing

        avg_val_loss = val_loss / len(val_loader)
        scheduler.step(avg_val_loss)

        average = 'binary' if mode == 'binary' else 'macro'
        acc = accuracy_score(val_labels, val_preds)
        prec = precision_score(val_labels, val_preds, average=average, zero_division=0)
        rec = recall_score(val_labels, val_preds, average=average, zero_division=0)
        f1 = f1_score(val_labels, val_preds, average=average, zero_division=0)
        if mode == 'multi':
            y_true = np.array(val_labels)
            y_probs = np.vstack(val_probs)  # Stack list of softmax outputs into array
            auc = roc_auc_score(y_true, y_probs, multi_class='ovr', average='macro')
        else:
            auc = roc_auc_score(val_labels, val_preds)

        

        print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        print(f"Val Acc: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(classifier.state_dict(), f'{backbone_name}_{mode}_best_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")

                # 🧩 Plot confusion matrix before exiting
                cm = confusion_matrix(val_labels, val_preds)
                disp = ConfusionMatrixDisplay(confusion_matrix=cm)
                disp.plot(cmap='Blues')
                plt.title(f'{backbone_name.upper()} {mode.upper()} Confusion Matrix (Early Stop)')
                plt.show()
                break


        torch.cuda.empty_cache()
            # 🧩 Final confusion matrix if training completed full epochs
    cm = confusion_matrix(val_labels, val_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap='Blues')
    plt.title(f'{backbone_name.upper()} {mode.upper()} Confusion Matrix (Final Epoch)')
    plt.show()

    return classifier

# === Evaluate Combined Model ===
def evaluate_combined_model(backbone, binary_model, multi_model, val_loader, backbone_name='model'):
    print(f"\n🔍 Evaluating Combined Model...")

    binary_model.eval()
    multi_model.eval()
    backbone.eval()

    all_labels = []
    binary_probs = []
    multi_probs = []

    with torch.no_grad():
        for inputs, _, multi_targets in tqdm(val_loader, desc="Extracting Features"):
            inputs = inputs.to(device)
            features = backbone(inputs)

            bin_logits = binary_model(features)
            bin_prob = torch.sigmoid(bin_logits).squeeze().cpu().numpy()
            binary_probs.extend(bin_prob if isinstance(bin_prob, np.ndarray) else [bin_prob])

            multi_out = multi_model(features)
            softmax_probs = F.softmax(multi_out, dim=1).cpu().numpy()
            multi_probs.extend(softmax_probs)

            all_labels.extend(multi_targets.numpy())

    binary_probs = np.array(binary_probs)
    multi_probs = np.array(multi_probs)
    all_labels = np.array(all_labels)

    # === Threshold Sweep for 100% recall on "non-nevus" (positive class = 1) ===
    best_thresh = 0.5
    for thresh in np.linspace(0.0, 1.0, 101):
        bin_preds = (binary_probs > thresh).astype(int)
        recall = recall_score((all_labels != 0).astype(int), bin_preds, zero_division=0)
        if recall == 1.0:
            best_thresh = thresh
            break

    print(f"✅ Using binary threshold = {best_thresh:.2f} (100% recall)")

    # === Final Predictions Using Best Threshold ===
    bin_final = (binary_probs > best_thresh).astype(int)
    multi_preds = np.argmax(multi_probs, axis=1)

    final_preds = []
    for b, m in zip(bin_final, multi_preds):
        final_preds.append(0 if b == 0 else m + 1)  # nevus = 0, others shifted +1

    # === Metrics ===
    acc = accuracy_score(all_labels, final_preds)
    prec = precision_score(all_labels, final_preds, average='macro', zero_division=0)
    rec = recall_score(all_labels, final_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, final_preds, average='macro', zero_division=0)

    # One-hot encode ground truth for multiclass AUC
    y_true_onehot = np.zeros((len(all_labels), 9))
    for idx, label in enumerate(all_labels):
        y_true_onehot[idx, label] = 1

    auc = roc_auc_score(y_true_onehot, np.insert(multi_probs, 0, binary_probs, axis=1), average='macro', multi_class='ovr')

    print(f"Combined Acc: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}")

    cm = confusion_matrix(all_labels, final_preds)
    ConfusionMatrixDisplay(confusion_matrix=cm).plot(cmap='Blues')
    plt.title(f'{backbone_name.upper()} Combined Confusion Matrix')
    plt.show()


# === Main Execution ===
if __name__ == "__main__":
    # --- Build separate caches ---
    #print("📦 Building binary cache (no augmentation)...")
    #build_cache(split_dir, os.path.join(binary_cache_root, 'binary'), transform=base_transform, is_binary=True)

    #print("📦 Building multi-class cache (with augmentation)...")
    #build_cache(split_dir, os.path.join(multi_cache_root, 'multi'), transform=base_transform, is_binary=False)

    # --- Loaders ---
    binary_train_loader, binary_val_loader = get_loaders(
        os.path.join(binary_cache_root, 'binary', 'train'),
        os.path.join(binary_cache_root, 'binary', 'val')
    )

    multi_train_loader, multi_val_loader = get_loaders(
        os.path.join(multi_cache_root, 'multi', 'train'),
        os.path.join(multi_cache_root, 'multi', 'val')
    )

    # --- Train each backbone separately ---
    for backbone_name in ['resnet', 'mobilenet','efficientnet']:
        print(f"\n🚀 Training backbone: {backbone_name}")
        if backbone_name == 'mobilenet':
            model = models.mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
            feature_dim = model.last_channel
        elif backbone_name == 'resnet':
            model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
            feature_dim = model.fc.in_features
        elif backbone_name == 'efficientnet':
            model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
            feature_dim = model.classifier[1].in_features
        else:
            continue

        # Fine-tune last few layers
        for param in list(model.parameters())[:-10]:
            param.requires_grad = False

        backbone = BackboneExtractor(model, backbone_name)
        binary_model = BinaryClassifier(feature_dim)
        multi_model = MultiClassifier(feature_dim)
        #multi_model.load_state_dict(torch.load(f'{backbone_name}_multi_best_model.pth'))


        # --- Train multi ---
        trained_multi = train_model(backbone, multi_model, multi_train_loader, multi_val_loader,
                                    mode='multi', num_epochs=100, patience=10, backbone_name=backbone_name)
        
        # --- Train binary ---
        trained_binary = train_model(backbone, binary_model, binary_train_loader, binary_val_loader,
                                     mode='binary', num_epochs=100, patience=10, backbone_name=backbone_name)

        # --- Load best weights ---
        #trained_binary.load_state_dict(torch.load(f'{backbone_name}_binary_best_model.pth'))
        #trained_multi.load_state_dict(torch.load(f'{backbone_name}_multi_best_model.pth'))

        # --- Evaluate combined model ---
        #evaluate_combined_model(backbone, trained_binary, trained_multi, binary_val_loader, backbone_name)

