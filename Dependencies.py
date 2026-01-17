!pip install geopandas rasterio torch torchvision scikit-learn matplotlib tqdm
import os
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
from tqdm import tqdm
