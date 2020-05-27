from torch.utils.data import DataLoader

from preprocessing.methods import parse_datasets, prepare_classification_data, confusion
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import numpy as np

labels, peptide_dict = parse_datasets()
X_train, X_test, y_train, y_test = prepare_classification_data(labels, peptide_dict)

seqs = X_train.view(len(X_train), 21, 70)
#print(seqs.shape)
labels = y_train
writer = SummaryWriter()

net = nn.Sequential(
    nn.Conv1d(21, 10, 1),
    nn.ReLU(),
    nn.BatchNorm1d(10),

    nn.Conv1d(10, 3, 1),
    nn.ReLU(),
    nn.BatchNorm1d(3),

    nn.Conv1d(3, 30, 3),
    nn.MaxPool1d(2),

    nn.Conv1d(30, 2, 34),
    nn.Softmax()

)
from sklearn import svm, datasets
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer

epochs = 30
import matplotlib.pyplot as plt
fpr_list = []
tpr_list = []
losses = []
optimizer = torch.optim.Adam(net.parameters())
for epoch in range(epochs):
    print("Epoch " + str(epoch))
    epoch_loss = 0
    correct = 0
    shuffle = torch.randperm(len(seqs))
    seqs = seqs[shuffle]
    labels = labels[shuffle]
    predicted_list = []
    for minibatchindex in range(0, len(seqs), 32):
        mb_data = seqs[minibatchindex:minibatchindex + 32]
        mb_labels = labels[minibatchindex:minibatchindex + 32]

        prediction = net(mb_data)

        predicted_list.append(prediction[0][0][0].detach().numpy())
        #print(prediction.shape)
        #loss = ((prediction-mb_labels)**2).sum()
        loss = F.binary_cross_entropy(prediction, mb_labels)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        #print((prediction == mb_labels).sum())
        #print((prediction != mb_labels).sum())

        correct += (prediction.argmax() == mb_labels).float().sum()
        accuracy = 100 * correct / len(mb_data)
        epoch_loss += float(loss.detach())

    losses.append(epoch_loss)
    print(' Epoch loss: %.20f, Accuracy %.20f' % (epoch_loss / 100, accuracy))

net.eval()
output = net(seqs).view(-1, 2)
#print(output)
# for i, n in enumerate(output):
#     print(n, labels[i])

plt.plot(range(epochs), losses)
plt.ylabel('Loss')
plt.xlabel('Epoch')
# plt.plot(fpr_list, tpr_list)
#
plt.show()
