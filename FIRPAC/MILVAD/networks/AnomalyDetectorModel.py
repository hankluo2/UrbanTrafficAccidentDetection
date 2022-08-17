"""
Code from https://github.com/ekosman/AnomalyDetectionCVPR2018-Pytorch
Modified to adapt the customed method.
"""

import torch
from torch import nn


class AnomalyDetector(nn.Module):
    def __init__(self, input_dim):
        super(AnomalyDetector, self).__init__()

        # customly added layer
        # self.fc0 = nn.Linear(input_dim, 4096)
        # self.relu0 = nn.ReLU()
        # self.dropout0 = nn.Dropout(0.6)

        self.fc1 = nn.Linear(input_dim, 512)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.6)

        self.fc2 = nn.Linear(512, 32)
        self.dropout2 = nn.Dropout(0.6)

        self.fc3 = nn.Linear(32, 1)
        self.sig = nn.Sigmoid()

        # In the original keras code they use "glorot_normal"
        # As I understand, this is the same as xavier normal in Pytorch
        # nn.init.xavier_normal_(self.fc0.weight)

        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.xavier_normal_(self.fc2.weight)
        nn.init.xavier_normal_(self.fc3.weight)

    def forward(self, x):
        # x = self.dropout0(self.relu0(self.fc0(x)))  # custom

        x = self.dropout1(self.relu1(self.fc1(x)))
        # x = self.relu1(self.fc1(x))
        x = self.dropout2(self.fc2(x))
        # x = self.fc2(x)
        x = self.sig(self.fc3(x))
        return x


def custom_objective(y_pred, y_true):
    # y_pred (batch_size, 32, 1)
    # y_true (batch_size)
    lambdas = 8e-5

    normal_vids_indices = torch.where(y_true == 0)
    anomal_vids_indices = torch.where(y_true == 1)

    normal_segments_scores = y_pred[normal_vids_indices].squeeze(-1)  # (batch/2, 32, 1)
    anomal_segments_scores = y_pred[anomal_vids_indices].squeeze(-1)  # (batch/2, 32, 1)

    # get the max score for each video
    normal_segments_scores_maxes = normal_segments_scores.max(dim=-1)[0]
    anomal_segments_scores_maxes = anomal_segments_scores.max(dim=-1)[0]

    hinge_loss = 1 - anomal_segments_scores_maxes + normal_segments_scores_maxes
    hinge_loss = torch.max(hinge_loss, torch.zeros_like(hinge_loss))
    """
    Smoothness of anomalous video
    NOTE: commented for customization: elimination of smooth loss
    """
    # smoothed_scores = anomal_segments_scores[:, 1:] - anomal_segments_scores[:, :-1]
    # smoothed_scores_sum_squared = smoothed_scores.pow(2).sum(dim=-1)
    """
    Sparsity of anomalous video
    """
    sparsity_loss = anomal_segments_scores.sum(dim=-1)

    # final_loss = (hinge_loss + lambdas * smoothed_scores_sum_squared + lambdas * sparsity_loss).mean()
    # modified: eliminated smooth loss
    final_loss = (hinge_loss + lambdas * sparsity_loss).mean()
    return final_loss


class RegularizedLoss(torch.nn.Module):
    def __init__(self, model, original_objective, lambdas=0.001):
        super(RegularizedLoss, self).__init__()
        self.lambdas = lambdas
        self.model = model
        self.objective = original_objective

    def forward(self, y_pred, y_true):
        # loss
        # Our loss is defined with respect to l2 regularization, as used in the original keras code
        # fc0_params = torch.cat(tuple([x.view(-1) for x in self.model.fc0.parameters()]))

        fc1_params = torch.cat(tuple([x.view(-1) for x in self.model.fc1.parameters()]))
        fc2_params = torch.cat(tuple([x.view(-1) for x in self.model.fc2.parameters()]))
        fc3_params = torch.cat(tuple([x.view(-1) for x in self.model.fc3.parameters()]))

        # l0_regularization = self.lambdas * torch.norm(fc0_params, p=2)

        l1_regularization = self.lambdas * torch.norm(fc1_params, p=2)
        l2_regularization = self.lambdas * torch.norm(fc2_params, p=2)
        l3_regularization = self.lambdas * torch.norm(fc3_params, p=2)

        return self.objective(y_pred, y_true) + l1_regularization + l2_regularization + l3_regularization
