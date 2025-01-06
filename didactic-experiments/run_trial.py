import os
import IPython.display
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import open_clip
import pandas as pd
from collections import OrderedDict
import torch
import tensorflow as tf
from torch.nn import functional as F
import torch.nn as nn
import pickle
from scipy.stats import multivariate_normal
from scipy.spatial import distance
from typing import Any, Dict, Optional, Tuple, Union

import torch
from torch.utils.data import DataLoader
from torch import nn
from torch.optim import Adam
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import StepLR

import pickle
import pandas as pd
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BJointGaussianDataset:
    """
    generates B data from a gaussian, takes in two transforms to go from
    B -> A
    B -> C
    for multi-modal contrastive training
    """
    def __init__(self, mu, sigma, transform1, transform2, pair, dims=(2, 2, 2), num_points=1000, noise=(0.01, 0.01, 0.1)):
        assert transform1.shape == (dims[0], dims[1]), "transform1 must be of shape (A_dim, B_dim)"
        assert transform2.shape == (dims[2], dims[1]), "transform2 must be of shape (C_dim, B_dim)"

        A_dim, B_dim, C_dim = dims
        eps1 = np.random.randn(num_points, A_dim) * noise[0]  # Adjusted shape for noise addition
        eps2 = np.random.randn(num_points, C_dim) * noise[1]  # Adjusted for transform2's output dimension
        corr_eps = np.random.randn(num_points, C_dim) * noise[2]  # correlated noise

        self.B = np.random.multivariate_normal(mu, sigma, num_points)
        self.A = np.dot(self.B, transform1.T) + eps1 + corr_eps
        self.C = np.dot(self.B, transform2.T) + eps2 + corr_eps

        self.points = np.hstack([self.A, self.B, self.C])
        
        self.pair = pair
        self.dims = dims
    
    def __len__(self):
        return self.points.shape[0]
    
    def __getitem__(self, idx):
        vec = self.points[idx]

        A_end = self.dims[0]
        B_end = self.dims[0] + self.dims[1]

        A = vec[:A_end]
        B = vec[A_end: B_end]
        C = vec[B_end:]
        quads = [A, B, C]

        return quads[self.pair[0]], quads[self.pair[1]]
    
    
def calculate_l2_penalty(features):
    """
    penalizes embeddings from deviating too far from the hypersphere
    """
    norms = torch.norm(features, p=2, dim=1)  # Compute L2 norms along the appropriate dimension
    penalty = torch.mean(norms)  # Calculate mean squared deviation from 1
    return penalty

class ClipLoss(nn.Module):
    """
    Traditional CLIP loss in the 2-dimensional setting
    """
    def __init__(
            self,
            penalty=1e-4,
            dist_loss=True
    ):
        super().__init__()
        self.penalty = penalty
        self.dist_loss = dist_loss

        # cache state
        self.prev_num_logits = 0
        self.labels = {}

    def get_ground_truth(self, device, num_logits) -> torch.Tensor:
        # calculated ground-truth and cache if enabled
        if self.prev_num_logits != num_logits or device not in self.labels:
            labels = torch.arange(num_logits, device=device, dtype=torch.long)
            self.labels[device] = labels
            self.prev_num_logits = num_logits
        else:
            labels = self.labels[device]
        return labels

    def get_logits(self, A_features, B_features, logit_scale):
        A_logits = logit_scale * A_features @ B_features.T
        B_logits = logit_scale * B_features @ A_features.T
        if self.dist_loss:
            A_logits = -logit_scale * torch.cdist(A_features, B_features, p=2)
            B_logits = -logit_scale * torch.cdist(B_features, A_features, p=2)

        return A_logits, B_logits

    def forward(self, A_features, B_features, logit_scale, output_dict=False):
        device = A_features.device
        A_logits, B_logits = self.get_logits(A_features, B_features, logit_scale)

        labels = self.get_ground_truth(device, A_logits.shape[0])

        contrastive_loss = (
            F.cross_entropy(A_logits, labels) +
            F.cross_entropy(B_logits, labels)
        ) / 2

        # Calculate L2 norm penalties for image and text features
        l2_penalty  = calculate_l2_penalty(A_features)
        l2_penalty += calculate_l2_penalty(B_features)

        # Combine the penalties (adjust the regularization strength as needed)
        total_loss = contrastive_loss + self.penalty * l2_penalty

        return {"contrastive_loss": total_loss} if output_dict else total_loss


class UnconsciousLoss(nn.Module):
    """
    CLIP loss in the three-modality setting
    """
    def __init__(self, penalty=1e-4, dist_loss=True):
        super(UnconsciousLoss, self).__init__()
        self.penalty = penalty
        self.dist_loss = dist_loss

        # Cache state
        self.prev_num_logits = 0
        self.labels = {}

    def get_ground_truth(self, device, num_logits) -> torch.Tensor:
        # Calculated ground-truth and cache if enabled
        if self.prev_num_logits != num_logits or device not in self.labels:
            labels = torch.arange(num_logits, device=device, dtype=torch.long)
            self.labels[device] = labels
            self.prev_num_logits = num_logits
        else:
            labels = self.labels[device]
        return labels

    def get_logits(self, A_features, B_features, logit_scale):
        A_logits = logit_scale * A_features @ B_features.T
        B_logits = logit_scale * B_features @ A_features.T
        if self.dist_loss:
            A_logits = -logit_scale * torch.cdist(A_features, B_features, p=2)
            B_logits = -logit_scale * torch.cdist(B_features, A_features, p=2)

        return A_logits, B_logits
    def forward(self, A1_features, B1_features, B2_features, C2_features, logit_scale, output_dict=False):
        device = A1_features.device

        # Compute Contrastive Loss for A-B and B-C
        A1_logits, B1_logits = self.get_logits(A1_features, B1_features, logit_scale)
        B2_logits, C2_logits = self.get_logits(B2_features, C2_features, logit_scale)

        AB_labels = self.get_ground_truth(device, A1_logits.shape[0])
        BC_labels = self.get_ground_truth(device, B2_logits.shape[0])
        
        AB_loss = (F.cross_entropy(A1_logits, AB_labels) + F.cross_entropy(B1_logits, AB_labels)) / 2
        BC_loss = (F.cross_entropy(B2_logits, BC_labels) + F.cross_entropy(C2_logits, BC_labels)) / 2
        contrastive_loss = (AB_loss + BC_loss) / 2

        # Compute L2 norm penalties for all features
        l2_penalty = calculate_l2_penalty(A1_features) + calculate_l2_penalty(B1_features) + \
                     calculate_l2_penalty(B2_features) + calculate_l2_penalty(C2_features)

        # Combine the penalties
        total_loss = contrastive_loss + (self.penalty * l2_penalty)

        return {"contrastive_loss": total_loss} if output_dict else total_loss


class SimpleMLP(nn.Module):
    """
    Simple MLP architecture for experiments
    """
    def __init__(self, input_dim, embed_dim):
        super(SimpleMLP, self).__init__()
        # Adjust the hidden_size or add more layers as needed
        hidden_size = input_dim // 2  # hyperparameter, could be adjusted for optimization
        self.fc1 = nn.Linear(input_dim, input_dim)  # Input layer
        self.fc2 = nn.Linear(input_dim, hidden_size)  # Hidden layer 2, added for more complexity
        self.fc3 = nn.Linear(hidden_size, embed_dim)  # Output layer
        
        # Adding batch normalization for stabilization and speed
        self.bn1 = nn.BatchNorm1d(input_dim)
        self.bn2 = nn.BatchNorm1d(hidden_size)

    def forward(self, x):
        x = x.float()
        x = F.relu(self.bn1(self.fc1(x)))  # Activation function after first layer + BatchNorm
        x = F.relu(self.fc2(x))  # Activation function after second layer + BatchNorm
        x = self.fc3(x)  # No activation function after output layer, linear output
        return x

class CLIP(nn.Module):
    """
    Model for 2-modality CLIP.
    """
    def __init__(
            self,
            input_dims: Tuple[int, int],
            embed_dim: int,
            init_logit_scale: float = np.log(1 / 0.07),
            init_logit_bias: Optional[float] = None,
            output_dict: bool = False,
            normalize = False
    ):
        super(CLIP, self).__init__()
        self.output_dict = output_dict
        self.normalize = normalize

        # Define encoders for modality A and B
        self.encoderA = SimpleMLP(input_dims[0], embed_dim)
        self.encoderC = SimpleMLP(input_dims[1], embed_dim)

        self.logit_scale = nn.Parameter(torch.tensor(init_logit_scale))
        self.logit_bias = nn.Parameter(torch.tensor(init_logit_bias)) if init_logit_bias is not None else None

    def encode(self, inputs, encoder):
        features = encoder(inputs)
        return F.normalize(features, dim=-1) if self.normalize else features

    def forward(
            self,
            A: Optional[torch.Tensor] = None,
            C: Optional[torch.Tensor] = None,
    ):
        A_features = self.encode(A, self.encoderA) if A is not None else None
        C_features = self.encode(C, self.encoderC) if C is not None else None

        if self.output_dict:
            out_dict = {
                "A_features": A_features,
                "C_features": C_features,
                "logit_scale": self.logit_scale.exp()
            }
            if self.logit_bias is not None:
                out_dict['logit_bias'] = self.logit_bias
            return out_dict

        results = [A_features, C_features, self.logit_scale.exp()]
        if self.logit_bias is not None:
            results.append(self.logit_bias)
        return results

class UnconsciousCLIP(nn.Module):
    """
    Model for 3-modality CLIP
    """
    def __init__(
            self,
            input_dims: Tuple[int, int, int],
            embed_dim: int,
            init_logit_scale: float = np.log(1 / 0.07),
            init_logit_bias: Optional[float] = None,
            output_dict: bool = False,
            normalize = False
    ):
        super(UnconsciousCLIP, self).__init__()
        self.output_dict = output_dict
        self.normalize = normalize
        
        # Define three encoders for different modalities
        self.encoderA = SimpleMLP(input_dims[0], embed_dim)
        self.encoderB = SimpleMLP(input_dims[1], embed_dim)
        self.encoderC = SimpleMLP(input_dims[2], embed_dim)

        self.logit_scale = nn.Parameter(torch.tensor(init_logit_scale))
        self.logit_bias = nn.Parameter(torch.tensor(init_logit_bias)) if init_logit_bias is not None else None
        
    def encode(self, features, encoder):
        encoded_features = encoder(features)
        return F.normalize(encoded_features, dim=-1) if self.normalize else encoded_features
    
    def forward(
            self,
            A1: Optional[torch.Tensor] = None,
            B1: Optional[torch.Tensor] = None,
            B2: Optional[torch.Tensor] = None,
            C2: Optional[torch.Tensor] = None,
    ):
        A1_features = self.encode(A1, self.encoderA) if A1 is not None else None
        B1_features = self.encode(B1, self.encoderB) if B1 is not None else None
        B2_features = self.encode(B2, self.encoderB) if B2 is not None else None
        C2_features = self.encode(C2, self.encoderC) if C2 is not None else None

        if self.output_dict:
            out_dict = {
                "A1_features": A1_features,
                "B1_features": B1_features,
                "B2_features": B2_features,
                "C2_features": C2_features,
                "logit_scale": self.logit_scale.exp()
            }
            if self.logit_bias is not None:
                out_dict['logit_bias'] = self.logit_bias
            return out_dict

        return_vals = [A1_features, B1_features, B2_features, C2_features, self.logit_scale.exp()]
        if self.logit_bias is not None:
            return_vals.append(self.logit_bias)
        return return_vals


class DisparateCLIP(nn.Module):
    """
    Model for 3-modality CLIP
    """
    def __init__(
            self,
            input_dims: Tuple[int, int, int],
            embed_dim: int,
            init_logit_scale: float = np.log(1 / 0.07),
            init_logit_bias: Optional[float] = None,
            output_dict: bool = False,
            normalize = False
    ):
        super(DisparateCLIP, self).__init__()
        self.output_dict = output_dict
        self.normalize = normalize
        
        # Define three encoders for different modalities
        self.encoderA = SimpleMLP(input_dims[0], embed_dim)
        self.encoderB1 = SimpleMLP(input_dims[1], embed_dim)
        self.encoderB2 = SimpleMLP(input_dims[1], embed_dim)
        self.encoderC = SimpleMLP(input_dims[2], embed_dim)

        self.logit_scale = nn.Parameter(torch.tensor(init_logit_scale))
        self.logit_bias = nn.Parameter(torch.tensor(init_logit_bias)) if init_logit_bias is not None else None
        
    def encode(self, features, encoder):
        encoded_features = encoder(features)
        return F.normalize(encoded_features, dim=-1) if self.normalize else encoded_features
    
    def forward(
            self,
            A1: Optional[torch.Tensor] = None,
            B1: Optional[torch.Tensor] = None,
            B2: Optional[torch.Tensor] = None,
            C2: Optional[torch.Tensor] = None,
    ):
        A1_features = self.encode(A1, self.encoderA) if A1 is not None else None
        B1_features = self.encode(B1, self.encoderB1) if B1 is not None else None
        B2_features = self.encode(B2, self.encoderB2) if B2 is not None else None
        C2_features = self.encode(C2, self.encoderC) if C2 is not None else None

        if self.output_dict:
            out_dict = {
                "A1_features": A1_features,
                "B1_features": B1_features,
                "B2_features": B2_features,
                "C2_features": C2_features,
                "logit_scale": self.logit_scale.exp()
            }
            if self.logit_bias is not None:
                out_dict['logit_bias'] = self.logit_bias
            return out_dict

        return_vals = [A1_features, B1_features, B2_features, C2_features, self.logit_scale.exp()]
        if self.logit_bias is not None:
            return_vals.append(self.logit_bias)
        return return_vals

def computeLSEGridDist(featureA, featureC, PHI_B):
    ADist = -torch.cdist(featureA, PHI_B, p=2).unsqueeze(1)
    CDist = -torch.cdist(featureC, PHI_B, p=2).unsqueeze(0)
    aggregate = ADist + CDist
    featureLSE = torch.logsumexp(aggregate, -1).detach().numpy()
    return featureLSE

def computeLSEGridDot(featureA, featureC, PHI_B):
    AplusC_grid = featureA.unsqueeze(1) + featureC.unsqueeze(0)        
    aggregate = AplusC_grid @ PHI_B.T
    featureLSE = torch.logsumexp(aggregate, -1).detach().numpy()
    return featureLSE

def evaluate(model, A, B, C, PHI_B, dist_loss=True):
    """
    Evaluate accuracy. Is the cosine similarity of the positive pair the highest among random pairs?
    PHI_B is a precomputed vector of feature_B encodings used for LSE.
    """
    def computeAcc(grid):
        max_indices = np.argmax(grid, axis=1)
        count = 0
        for idx, i in enumerate(max_indices):
            count += (idx == i)
        
        acc = count / len(max_indices)
        return acc
    
    # Features
    featureA = model.encode(A, model.encoderA)
    featureC = model.encode(C, model.encoderC)
    
    if dist_loss:
        grid = -torch.cdist(featureA, featureC, p=2).detach().numpy()
    else:
        grid = (featureA @ featureC.T).detach().numpy()
    direct_acc = computeAcc(grid)

    # Compute LSE Accuracy
    LSE_acc = None
    if PHI_B is not None:
        featureLSE = None
        if dist_loss:
            featureLSE = computeLSEGridDist(featureA, featureC, PHI_B)
        else:
            featureLSE = computeLSEGridDot(featureA, featureC, PHI_B)
        LSE_acc = computeAcc(featureLSE)
    
    print(f'accuracy: {direct_acc}, {LSE_acc}')
    return direct_acc, LSE_acc


def computeLSEGridDistDisparate(featureA, featureC, PHI_B1, PHI_B2):
    ADist = -torch.cdist(featureA, PHI_B1, p=2).unsqueeze(1)
    CDist = -torch.cdist(featureC, PHI_B2, p=2).unsqueeze(0)
    aggregate = ADist + CDist
    featureLSE = torch.logsumexp(aggregate, -1).detach().numpy()
    return featureLSE

def computeLSEGridDotDisparate(featureA, featureC, PHI_B1, PHI_B2):
    ADist = (featureA @ PHI_B1.T).unsqueeze(1)
    CDist = (featureC @ PHI_B2.T).unsqueeze(0)
    aggregate = ADist + CDist
    featureLSE = torch.logsumexp(aggregate, -1).detach().numpy()
    return featureLSE

def evaluateDisparate(model, A, B, C, B_ARR, dist_loss=True):
    """
    Evaluate accuracy. Is the cosine similarity of the positive pair the highest among random pairs?
    PHI_B is a precomputed vector of feature_B encodings used for LSE.
    """
    def computeAcc(grid):
        max_indices = np.argmax(grid, axis=1)
        count = 0
        for idx, i in enumerate(max_indices):
            count += (idx == i)
        
        acc = count / len(max_indices)
        return acc
    
    # Features
    featureA = model.encode(A, model.encoderA)
    featureC = model.encode(C, model.encoderC)
    
    if dist_loss:
        grid = -torch.cdist(featureA, featureC, p=2).detach().numpy()
    else:
        grid = (featureA @ featureC.T).detach().numpy()
    direct_acc = computeAcc(grid)

    # Compute LSE Accuracy
    LSE_acc = None
    if B_ARR is not None:
        PHI_B1 = model.encode(B_ARR, model.encoderB1)
        PHI_B2 = model.encode(B_ARR, model.encoderB2)
        featureLSE = None
        if dist_loss:
            featureLSE = computeLSEGridDistDisparate(featureA, featureC, PHI_B1, PHI_B2)
        else:
            featureLSE = computeLSEGridDotDisparate(featureA, featureC, PHI_B1, PHI_B2)
        LSE_acc = computeAcc(featureLSE)
    
    return direct_acc, LSE_acc

def computeAccuracy(model, dataset, dims=(2, 2, 2), unconscious=True, points=None, dist_loss=True, disparate=False):
    """
    Compute accuracy for a dataset given a model.

    Args:
    - model: The model to evaluate.
    - dataset: Dataset containing the points.
    - dims: Tuple containing the dimensions for slicing the dataset points.
            Assumes that points are concatenated in the order they appear in dims.
    - unconscious: Boolean indicating whether to compute unconscious accuracy.

    Returns:
    - Tuple containing direct accuracy and LSE accuracy.
    """
    if points is None:
        points = torch.tensor(dataset.points)
    
    # Calculate cumulative indices based on dims
    cum_dims = [sum(dims[:i+1]) for i in range(len(dims))]

    PHI_B = None
    if unconscious and not disparate:
        # Adjust to dynamically use dims
        PHI_B = model.encode(points[:, cum_dims[0]:cum_dims[1]], model.encoderB)
        
    B_raw = None
    if disparate:
        B_raw = points[:, cum_dims[0]:cum_dims[1]]

    batch_size = 32
    direct_acc = 0
    LSE_acc = 0
    for i in range(len(dataset) // batch_size):
        # Dynamically slice points based on dims
        A_slice = points[batch_size*i:batch_size*(i+1), 0:cum_dims[0]]
        B_slice = points[batch_size*i:batch_size*(i+1), cum_dims[0]:cum_dims[1]]
        C_slice = points[batch_size*i:batch_size*(i+1), cum_dims[1]:cum_dims[2]]

        direct, LSE = evaluate(model, A_slice, B_slice, C_slice, PHI_B, dist_loss=dist_loss)
        if disparate:
            direct, LSE = evaluateDisparate(model, A_slice, B_slice, C_slice, B_raw, dist_loss=dist_loss)
        
        direct_acc += direct
        if unconscious:
            LSE_acc += LSE
        
    direct_acc /= (len(dataset) // batch_size)
    
    if unconscious:
        LSE_acc /= (len(dataset) // batch_size)
    
    return direct_acc, LSE_acc




def train_conscious(conscious_model, conscious_loss, optimizer, scaler, scheduler, AC_dataset, val_AC, val_general, dims, batch_size=64, num_epochs=32, dist_loss=True):
    losses = []
    val_losses = []
    conscious_accs = []

    AC_dataloader = DataLoader(AC_dataset, batch_size=batch_size, shuffle=True)
    val_AC_dl = DataLoader(val_AC, batch_size=batch_size, shuffle=True)

    for epoch in range(num_epochs):
        acc = computeAccuracy(conscious_model, val_AC, dims=dims, unconscious=False, dist_loss=dist_loss)
        conscious_accs.append(acc[0])  

        conscious_model.train()

        running_loss = 0.0
        running_val_loss = 0.0

        # train conscious
        for AC_batch in AC_dataloader:
            A, C = AC_batch
            A, C = A.to(device), C.to(device)

            optimizer.zero_grad()

            model_out = conscious_model(A=A, C=C)
            total_loss = conscious_loss(*model_out)

            # Backward pass and optimize
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += total_loss.item() * A.size(0)

        # Learning rate scheduling
        scheduler.step()

        # VALIDATION CODE
        conscious_model.eval()

        for AC_batch in val_AC_dl:
            A, C = AC_batch
            A, C = A.to(device), C.to(device)

            optimizer.zero_grad()

            model_out = conscious_model(A=A, C=C)
            val_loss = conscious_loss(*model_out)
            running_val_loss += val_loss.item() * A.size(0)

        val_epoch_loss = running_val_loss / len(val_AC_dl.dataset)
        val_losses.append(val_epoch_loss)
        # Calculate average loss over the epoch
        epoch_loss = running_loss / len(AC_dataloader.dataset)
        losses.append(epoch_loss)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Val Loss: {val_epoch_loss:.4f}')
    print('Training complete.')
    return losses, val_losses, conscious_accs

def train_unconscious(model, loss_function, 
                      optimizer, scaler, scheduler, 
                      AB_dataset, BC_dataset,
                      val_AB, val_BC, 
                      dims,
                      batch_size=64, 
                      num_epochs=32,
                      dist_loss=True,
                      disparate=False
                     ):
    losses = []
    val_losses = []
    val_accs = []
    
    # dataloaders for train
    AB_dataloader = DataLoader(AB_dataset, batch_size=batch_size, shuffle=True)
    BC_dataloader = DataLoader(BC_dataset, batch_size=batch_size, shuffle=True)

    # dataloaders for val
    val_AB_dl = DataLoader(val_AB, batch_size=batch_size, shuffle=True)
    val_BC_dl = DataLoader(val_BC, batch_size=batch_size, shuffle=True)
    
    for epoch in range(num_epochs):
        acc = computeAccuracy(model, val_AB, dims=dims, unconscious=True, points=torch.tensor(AB_dataset.points), dist_loss=dist_loss, disparate=disparate)
        val_accs.append(acc)  

        model.train()  # Set model to training mode

        running_loss = 0.0
        running_val_loss = 0.0

        # train unconscious
        BC_dataloader_iterator = iter(BC_dataloader)
        for AB_batch in AB_dataloader:
            try:
                BC_batch = next(BC_dataloader_iterator)
            except StopIteration:
                BC_dataloader_iterator = iter(BC_dataloader)
                BC_batch = next(BC_dataloader_iterator)

            # Unpack the batches
            A1, B1 = AB_batch
            B2, C2 = BC_batch

            # Move tensors to the correct device
            A1, B1 = A1.to(device), B1.to(device)
            B2, C2 = B2.to(device), C2.to(device)

            optimizer.zero_grad()

            model_out = model(A1=A1, B1=B1, B2=B2, C2=C2)
            total_loss = loss_function(*model_out)

            # Backward pass and optimize
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += total_loss.item() * A1.size(0)

        # Learning rate scheduling
        scheduler.step()

        # VALIDATION CODE
        model.eval()

        val_BC_iter = iter(val_BC_dl)
        for AB_batch in val_AB_dl:
            try:
                BC_batch = next(val_BC_iter)
            except StopIteration:
                val_BC_iter = iter(val_BC_dl)
                BC_batch = next(val_BC_iter)

            # Unpack the batches
            A1, B1 = AB_batch
            B2, C2 = BC_batch

            # Move tensors to the correct device
            A1, B1 = A1.to(device), B1.to(device)
            B2, C2 = B2.to(device), C2.to(device)
            # Forward pass
            model_out = model(A1=A1, B1=B1, B2=B2, C2=C2)
            # Assume the loss function is defined to handle the outputs from the model
            val_loss = loss_function(*model_out)  # You need to define or adjust this

            running_val_loss += val_loss.item() * A1.size(0)

        val_epoch_loss = running_val_loss / len(val_AB_dl.dataset)
        val_losses.append(val_epoch_loss)
        # Calculate average loss over the epoch
        epoch_loss = running_loss / len(AB_dataloader.dataset)
        losses.append(epoch_loss)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Val Loss: {val_epoch_loss:.4f}')
    print('Training complete.')
    return losses, val_losses, val_accs

def runExperiment(input_dims, 
                  dataDistribution,
                  l2_penalty, 
                  feature_dim, 
                  temperature, 
                  num_epochs,
                  lr,
                  TRAIN_SIZE, 
                  VAL_SIZE, 
                  BATCH_SIZE,
                  NOISE
                 ):

    ### DEFINE DATA ####
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    DIMS = input_dims
    
    mu = dataDistribution['mu']
    Sigma = dataDistribution['Sigma']
    transform1 = dataDistribution['transform1']
    transform2 = dataDistribution['transform2']

    # define datasets
    AB_dataset = BJointGaussianDataset(mu, Sigma, transform1=transform1, transform2=transform2, pair=[0, 1], dims=DIMS, num_points=TRAIN_SIZE, noise=NOISE)
    BC_dataset = BJointGaussianDataset(mu, Sigma, transform1=transform1, transform2=transform2, pair=[1, 2], dims=DIMS, num_points=TRAIN_SIZE, noise=NOISE)
    AC_dataset = BJointGaussianDataset(mu, Sigma, transform1=transform1, transform2=transform2, pair=[0, 2], dims=DIMS, num_points=TRAIN_SIZE, noise=NOISE)

    # validation data
    val_AB = BJointGaussianDataset(mu, Sigma, transform1=transform1, transform2=transform2, pair=[0, 1], dims=DIMS, num_points=VAL_SIZE, noise=NOISE)
    val_BC = BJointGaussianDataset(mu, Sigma, transform1=transform1, transform2=transform2, pair=[1, 2], dims=DIMS, num_points=VAL_SIZE, noise=NOISE)
    val_AC = BJointGaussianDataset(mu, Sigma, transform1=transform1, transform2=transform2, pair=[0, 2], dims=DIMS, num_points=VAL_SIZE, noise=NOISE)
    ##########################
    
    trained_models = {}
    accuracy_curves = {}
    perf_arr = []

    ### NORMALIZED MODEL ###
    normed_model = UnconsciousCLIP(DIMS, feature_dim, normalize=True, init_logit_scale=temperature)
    loss_function = UnconsciousLoss(penalty=l2_penalty, dist_loss=False)
    optimizer = Adam(normed_model.parameters(), lr=lr)
    scaler = GradScaler()
    scheduler = StepLR(optimizer, step_size=30, gamma=0.1)

    _, _, UC_acc_normalized = train_unconscious(normed_model, loss_function, optimizer, scaler, scheduler, 
                                     AB_dataset, BC_dataset, val_AB, val_BC, DIMS, batch_size=BATCH_SIZE, num_epochs=num_epochs,
                                        dist_loss=False)
    #########################

    #### L2 LOSS MODEL ######
    dist_model = UnconsciousCLIP(DIMS, feature_dim, normalize=False, init_logit_scale=temperature)
    loss_function = UnconsciousLoss(penalty=l2_penalty, dist_loss=True)
    optimizer = Adam(dist_model.parameters(), lr=lr)
    scaler = GradScaler()
    scheduler = StepLR(optimizer, step_size=30, gamma=0.1)

    _, _, UC_acc_dist = train_unconscious(dist_model, loss_function, optimizer, scaler, scheduler, 
                                     AB_dataset, BC_dataset, val_AB, val_BC, DIMS, batch_size=BATCH_SIZE, num_epochs=num_epochs,
                                    dist_loss=True)
    ##########################

    #### INNER PROD MODEL ####
    dot_model = UnconsciousCLIP(DIMS, feature_dim, normalize=False, init_logit_scale=temperature)
    loss_function = UnconsciousLoss(penalty=l2_penalty, dist_loss=False)
    optimizer = Adam(dot_model.parameters(), lr=lr)
    scaler = GradScaler()
    scheduler = StepLR(optimizer, step_size=30, gamma=0.1)

    _, _, UC_acc_dot = train_unconscious(dot_model, loss_function, optimizer, scaler, scheduler, 
                                     AB_dataset, BC_dataset, val_AB, val_BC, DIMS, batch_size=BATCH_SIZE, num_epochs=num_epochs,
                                        dist_loss=False)
    ##########################

    ##DISPARATE MODEL#########
    disparate_model = DisparateCLIP(DIMS, feature_dim, normalize=False, init_logit_scale=temperature)
    loss_function = UnconsciousLoss(penalty=l2_penalty, dist_loss=True)
    optimizer = Adam(disparate_model.parameters(), lr=lr)
    scaler = GradScaler()
    scheduler = StepLR(optimizer, step_size=30, gamma=0.1)

    _, _, UC_acc_disp = train_unconscious(disparate_model, loss_function, optimizer, scaler, scheduler, 
                                     AB_dataset, BC_dataset, val_AB, val_BC, DIMS, batch_size=64, num_epochs=num_epochs,
                                    dist_loss=True, disparate=True)

    ##########################
    
    # Train unconscious model
    

    ### CONSCIOUS MODEL ######
    conscious_model = CLIP((DIMS[0], DIMS[2]), feature_dim, normalize=False, init_logit_scale=temperature)
    conscious_loss = ClipLoss(penalty=l2_penalty)
    optimizer = Adam(conscious_model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=30, gamma=0.1)

    _, _, C_acc = train_conscious(conscious_model, conscious_loss, optimizer, scaler, scheduler, 
                                  AC_dataset, val_AC, val_AB, DIMS, batch_size=BATCH_SIZE, num_epochs=num_epochs)
    ##########################

    # Store trained models
    trained_models = {
        'norm_model': normed_model,
        'dist_model': dist_model,
        'dot_model': dot_model,
        'unconnected_mdoel': disparate_model,
        'conscious_model': conscious_model
    }

    accuracy_curves = {
        'norm_model': UC_acc_normalized,
        'dist_model': UC_acc_dist,
        'dot_model': UC_acc_dot,
        'unconnected_model': UC_acc_disp,
        'conscious_model': C_acc
    }
    return dataDistribution, trained_models, accuracy_curves

import argparse

def main():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--trial', type=int, required=True, help='Trial number')
    args = parser.parse_args()
    
    # Extract the 'trial' argument value
    trial = args.trial

    print(f'starting script')
    experiments_base_dir = 'corr_try2'

    # Define model parameters and data transformation
    num_epochs = 32
    lr = 0.01
    input_dims = [32, 16, 32]
    TRAIN_SIZE = 5000
    VAL_SIZE = 1000
    BATCH_SIZE = 64
    

    # Hyperparameters
    l2_penalties = [0.0]# [0.0, 0.001, 0.01, 0.1, 1.]
    feature_dims = [8]
    temperatures =  [0.01] #[0.001, 0.01, 0.1, 1.]
    correlated_noises = [0., 1., 2., 4., 8., 16., 32.]

    # Prepare to log experiment details
    df_records = []
    os.makedirs(experiments_base_dir, exist_ok=True)

    mu = np.random.uniform(-1, 1, (input_dims[1]))
    A = np.random.uniform(-1, 1, (input_dims[1], input_dims[1]))
    Sigma = A.T @ A

    transform1 = np.random.uniform(-1, 1, (input_dims[0], input_dims[1]))
    transform2 = np.random.uniform(-1, 1, (input_dims[2], input_dims[1]))

    dataDistribution = { }

    for corr_noise in correlated_noises:
        NOISE = (2., 2., corr_noise)

        dataDistribution = {
            'mu': mu,
            'Sigma': Sigma,
            'transform1': transform1,
            'transform2': transform2,
            'noise': NOISE
        }
        for l2_penalty in l2_penalties:
            for feature_dim in feature_dims:
                for temperature in temperatures:
                    # Construct a unique directory for each experiment configuration
                    exp_dir = f"trial_{trial}_l2_{l2_penalty}_fd_{feature_dim}_temp_{temperature}_noise_{corr_noise}"
                    exp_path = os.path.join(experiments_base_dir, exp_dir)
                    os.makedirs(exp_path, exist_ok=True)
                    
                    # Placeholder: Replace with actual experiment function
                    distribution, models, accs = runExperiment(
                        input_dims=input_dims, 
                        dataDistribution=dataDistribution,
                        l2_penalty=l2_penalty, 
                        feature_dim=feature_dim, 
                        temperature=temperature, 
                        num_epochs=num_epochs,
                        lr=lr,
                        TRAIN_SIZE=TRAIN_SIZE, 
                        VAL_SIZE=VAL_SIZE, 
                        BATCH_SIZE=BATCH_SIZE,
                        NOISE=NOISE
                    )
                    
                    experimentInfo = (distribution, models, accs)
                    
                    # Serialize experimentInfo
                    info_file_path = os.path.join(exp_path, 'experimentInfo.pkl')
                    with open(info_file_path, 'wb') as f:
                        pickle.dump(experimentInfo, f)
                    
                    # Log the experiment details for the DataFrame
                    df_records.append({
                        'trial': trial,
                        'corr_noise': corr_noise,
                        'l2_penalty': l2_penalty,
                        'feature_dim': feature_dim,
                        'temperature': temperature,
                        'experimentInfo_path': info_file_path
                    })

    # Create DataFrame and save to CSV for easy querying
    df = pd.DataFrame(df_records)
    df_csv_path = os.path.join(experiments_base_dir, f'experiment_details{trial}.csv')
    df.to_csv(df_csv_path, index=False)

    print(f"Experiment details saved to {df_csv_path}")

if __name__ == '__main__':
    main()