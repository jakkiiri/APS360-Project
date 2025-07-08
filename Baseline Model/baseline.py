# KNN and SVM is too simple. Try using a shallow CNN

import torch
import torch.nn as nn
import torch.nn.functional as F

class BaselineCNN(nn.Module):
    # we want the model to first use a binary classifier to check if the skin is healthy
    # then, if the skin is not healthy, we want the model to identify which disease is present
    def __init__(self, num_classes=1):
        """
        takes in num_classes as parameter for number of classes
        variable for flexibility in case we modify the dataset in the future
        """
        super(BaselineCNN, self).__init__()

        # convolutional layers
        self.pool = nn.MaxPool2d(2, 2)

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)

        # global average pooling to replace x.view
        self.global_pool = nn.AdaptiveAvgPool2d(1)  # outputs (B, 64, 1, 1)

        # fully connected layers
        self.fc1 = nn.Linear(64, 64)
        self.dropout = nn.Dropout(0.5)

        # binary classifier
        self.fc_binary = nn.Linear(64, 1)
        self.fc_multiclass = nn.Linear(64, num_classes)

    def forward(self, x):
        # CNN feature extraction
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)

        # simple fully connect layers
        x = self.global_pool(x)  # shape (B, 64, 1, 1)
        x = x.view(x.size(0), -1)  # flatten to (B, 64)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)

        out_binary = torch.sigmoid(self.fc_binary(x))  # shape (B,1)
        out_multiclass = self.fc_multiclass(x)         # raw logits for CrossEntropyLoss

        return out_binary, out_multiclass