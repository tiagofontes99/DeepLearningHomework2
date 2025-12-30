# -*- coding: utf-8 -*-


#https://github.com/MedMNIST/MedMNIST


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from torchvision import transforms

from medmnist import BloodMNIST, INFO

import argparse
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score


device = "cuda" if torch.cuda.is_available() else "cpu"


# Data Loading

data_flag = 'bloodmnist'
print(data_flag)
info = INFO[data_flag]
print(len(info['label']))
n_classes = len(info['label'])

# Transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[.5], std=[.5])
])

import time

# --------- Before Training ----------
total_start = time.time()

#Training Function

def train_epoch(loader, model, criterion, optimizer):
    
    ### YOUR CODE HERE ###

    return total_loss / len(loader)

#Evaluation Function

def evaluate(loader, model):
    model.eval()
    preds, targets = [], []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            labels = labels.squeeze().long()

            outputs = model(imgs)
            preds += outputs.argmax(dim=1).cpu().tolist()
            targets += labels.tolist()

    return accuracy_score(targets, preds)


def plot(epochs, plottable, ylabel='', name=''):
    plt.clf()
    plt.xlabel('Epoch')
    plt.ylabel(ylabel)
    plt.plot(epochs, plottable)
    plt.savefig('%s.pdf' % (name), bbox_inches='tight')

train_dataset = BloodMNIST(split='train', transform=transform, download=True, size=28)
val_dataset   = BloodMNIST(split='val',   transform=transform, download=True, size=28)
test_dataset  = BloodMNIST(split='test',  transform=transform, download=True, size=28)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# initialize the model
# get an optimizer
# get a loss criterion

### YOUR CODE HERE ###

# training loop
### you can use the code below or implement your own loop ###
train_losses = []
val_accs = []
test_accs = []
for epoch in range(epochs):

    epoch_start = time.time()

    train_loss = train_epoch(train_loader, model, criterion, optimizer)
    val_acc = evaluate(val_loader, model)
    print(f"Epoch {epoch+1}/{epochs} | Loss: {train_loss:.4f} | Val Acc: {val_acc:.4f}")

    train_losses.append(train_loss)
    val_accs.append(val_acc)

    epoch_end = time.time()
    epoch_time = epoch_end - epoch_start

    print(f"Epoch {epoch+1}/{epochs} | "
          f"Loss: {train_loss:.4f} | Val Acc: {val_acc:.4f} | "
          f"Time: {epoch_time:.2f} sec")

#Test Accuracy
test_acc = evaluate(test_loader, model)
print("Test Accuracy:", test_acc)
test_accs.append(test_acc)


#Save the model
#torch.save(model.state_dict(), "bloodmnist_cnn.pth")
#print("Model saved as bloodmnist_cnn.pth")


# --------- After Training ----------
total_end = time.time()
total_time = total_end - total_start

print(f"\nTotal training time: {total_time/60:.2f} minutes "
      f"({total_time:.2f} seconds)")

#print('Final Test acc: %.4f' % (evaluate(model, test_X, test_y)))

#config = "{}-{}-{}-{}-{}".format(opt.learning_rate, opt.optimizer, opt.no_maxpool, opt.no_softmax,)
config = "{}".format(str(0.1))

plot(epochs, train_losses, ylabel='Loss', name='CNN-training-loss-{}'.format(config))
plot(epochs, val_accs, ylabel='Accuracy', name='CNN-validation-accuracy-{}'.format(config))
plot(epochs, test_accs, ylabel='Accuracy', name='CNN-test-accuracy-{}'.format(config))