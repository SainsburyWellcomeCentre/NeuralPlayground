import json
import os
import random
import sys
from datetime import datetime
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import matthews_corrcoef, roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F


class Evaluator:
    """
    Class for evaluating the model
    """

    def __init__(self, model, args, dtype=torch.float64):
        """
        Args:
        :param model:   model to be evaluated
        :param args:    arguments
        :param dtype:   data type
        """

        self.model = model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

        self.pytorch_total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print("Total number of parameters: ", self.pytorch_total_params)
        self.dtype = dtype
        self.model = self.model.to(self.device)




