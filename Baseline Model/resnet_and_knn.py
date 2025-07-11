# generic imports for training things. add more as needed

# for the CNN part of this
import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.models import resnet18, ResNet18_Weights

# for the KNN part of this
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import numpy

# using gpu if possible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# getting resnet18 model
resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
# getting rid of fully connected layers - https://discuss.pytorch.org/t/use-resnet18-as-feature-extractor/8267 
feature_extraction = nn.Sequential(*list(resnet.children())[:-1])
feature_extraction.to(device)
feature_extraction.eval() 
print("resnet18 successfully extracted")

# the resnet input size is 224x224. our images are 512x512, so we'll need to resize them
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# getting the datasets and putting them into a dataloader. change paths for your own
# ~ ImageFolder(root = 'dataset_path', transform=transform)
train_dataset = ImageFolder(root = 'C:/Users/lawre/OneDrive - University of Toronto/2025 Summer/APS360/Project/DataSplit/test', transform=transform)
val_dataset = ImageFolder(root = 'C:/Users/lawre/OneDrive - University of Toronto/2025 Summer/APS360/Project/DataSplit/train', transform=transform)
test_dataset = ImageFolder(root = 'C:/Users/lawre/OneDrive - University of Toronto/2025 Summer/APS360/Project/DataSplit/val', transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
print("dataloaders ready")
# function for extracting features from dataset
def extract_features(loader):
    all_features = []
    all_labels = []
    with torch.no_grad():
        for imgs, lbls in loader:
            # move to gpu for extraction
            # output (batch, 512, 1, 1)
            # flatten to (batch, 512)
            imgs = imgs.to(device)
            feats = feature_extraction(imgs) 
            feats = feats.view(feats.size(0), -1)  
            feats = feats.cpu()  # move back to CPU for numpy
            # adding to the list
            all_features.append(feats)
            all_labels.append(lbls)
    # turns the thing into one big tensor before converting into numpy
    # scikit-learn stuff (the KNN) uses numpy arrays and not tensors
    # please use CPU because GPU tensors don't work with this
    features = torch.cat(all_features).numpy()
    labels = torch.cat(all_labels).numpy()
    return features, labels

# putting the 3 loaders into the feature extractor
print("Extracting features")
train_features, train_labels = extract_features(train_loader)
val_features, val_labels = extract_features(val_loader)
test_features, test_labels = extract_features(test_loader)
print("done extracting features")

# scaling 
scale = StandardScaler()
train_features_scaled = scale.fit_transform(train_features)
val_features_scaled = scale.transform(val_features)
test_features_scaled = scale.transform(test_features)
print("features successfully scaled")

# training the KNN
knn = KNeighborsClassifier(n_neighbors=8)
knn.fit(train_features_scaled, train_labels)
print("KNN finished fitting")

# validating based on val set (probably wasn't necessary to split into validation set and test set)
prediction_val = knn.predict(val_features_scaled)
accuracy_val = accuracy_score(val_labels, prediction_val)
print(f"Validation Accuracy: {accuracy_val*100:.2f}%")

# final prediction based on test set
prediction_test = knn.predict(test_features_scaled)
accuracy_test = accuracy_score(test_labels, prediction_test)
print(f"Test Accuracy: {accuracy_test*100:.2f}%")

print(classification_report(test_labels, prediction_test, target_names=test_dataset.classes))