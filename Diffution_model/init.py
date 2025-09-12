import torch
import torch.nn as nn
import torchvision
from torchvision.transforms import v2
from torch.utils.data import DataLoader
from tqdm import tqdm
from matplotlib.pyplot import plt
import math

DEVICE = "cude" if torch.cuda.is_available() else "cpu"

print(f'Using device {DEVICE}')