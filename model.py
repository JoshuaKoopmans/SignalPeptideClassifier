import torch.nn as nn


class ClassifierModel(nn.Module):
    def __init__(self, input_dim):
        super(ClassifierModel, self).__init__()
        self.seq = nn.Sequential(
            nn.Conv1d(input_dim, 10, 1),
            nn.ReLU(),
            nn.BatchNorm1d(10),

            nn.Conv1d(10, 3, 1),
            nn.ReLU(),
            nn.BatchNorm1d(3),

            nn.Conv1d(3, 30, 3),
            nn.MaxPool1d(2),

            nn.Conv1d(30, 1, 34),
            nn.Softmax(),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.seq(x)
