import torch
import torch.nn.functional as F
from torchvision import models
from torchvision.models import MobileNet_V2_Weights, ResNet50_Weights, EfficientNet_B0_Weights
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

# --- Load Cached Dataset Class ---
class CachedDataset(torch.utils.data.Dataset):
    def __init__(self, cached_dir):
        from glob import glob
        self.files = glob(os.path.join(cached_dir, '*', '*.pt'))
        self.files.sort()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = torch.load(self.files[idx], weights_only=False)  # Future-proof and warning-safe
        image = data['image']
        label = data['label']
        return image, label

# --- t-SNE Plot Function ---
def plot_tsne(embeddings, labels, title):
    tsne = TSNE(n_components=2, random_state=42)
    reduced = tsne.fit_transform(embeddings)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(reduced[:, 0], reduced[:, 1], c=labels, cmap='tab10', alpha=0.7)
    plt.legend(*scatter.legend_elements(), title="Classes")
    plt.title(title)
    plt.show()

# --- Embedding Extraction Function ---
def extract_embeddings(model, dataloader, model_type='mobilenet'):
    model.eval()
    embeddings = []
    labels = []

    with torch.no_grad():
        for inputs, targets in tqdm(dataloader, desc=f"Extracting embeddings ({model_type})"):
            inputs = inputs.to(device)

            if model_type == 'mobilenet':
                features = model.features(inputs)
            elif model_type == 'resnet':
                x = model.conv1(inputs)
                x = model.bn1(x)
                x = model.relu(x)
                x = model.maxpool(x)
                x = model.layer1(x)
                x = model.layer2(x)
                x = model.layer3(x)
                x = model.layer4(x)
                features = x
            elif model_type == 'efficientnet':
                features = model.features(inputs)
            elif model_type == 'densenet121' or model_type == 'densenet169':
                features = model.features(inputs)


            pooled = F.adaptive_avg_pool2d(features, (1, 1))
            embedding = pooled.view(pooled.size(0), -1)

            embeddings.append(embedding.cpu())
            labels.append(targets)

    embeddings = torch.cat(embeddings, dim=0).numpy()
    labels = torch.cat(labels, dim=0).numpy()

    return embeddings, labels

# --- Model Setup and t-SNE Pipeline ---
def process_pretrained_model(model_type, title, dataloader):
    if model_type == 'mobilenet':
        model = models.mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
    elif model_type == 'resnet':
        model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    elif model_type == 'efficientnet':
        model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
    elif model_type == 'densenet121':
        model = models.densenet121(pretrained=True)
    elif model_type == 'densenet169':
        emodel = models.densenet169(pretrained=True)

    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model = model.to(device)

    embeddings, labels = extract_embeddings(model, dataloader, model_type=model_type)
    plot_tsne(embeddings, labels, title)

# --- Safe Entry Point for Windows ---
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 32
    cache_root = r'C:\Users\shore\Desktop\APS360\Datasets\Cache'

    val_dataset = CachedDataset(os.path.join(cache_root, 'val'))

    # ✅ Use num_workers=0 for full Windows stability
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    process_pretrained_model('densenet121', 't-SNE: DenseNet121 (Pretrained)', val_loader)
    process_pretrained_model('densenet169', 't-SNE: DenseNet169 (Pretrained)', val_loader)
    process_pretrained_model('mobilenet', 't-SNE: MobileNetV2 (Pretrained)', val_loader)
    process_pretrained_model('resnet', 't-SNE: ResNet50 (Pretrained)', val_loader)
    process_pretrained_model('efficientnet', 't-SNE: EfficientNet-B0 (Pretrained)', val_loader)





