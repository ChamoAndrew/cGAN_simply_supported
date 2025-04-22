import cv2
import pandas as pd
import numpy as np
import torch
from torch.utils.data  import Dataset
from tqdm import tqdm

class CustomDataset(Dataset):
    def __init__(self, X, Y):
        super(CustomDataset).__init__()
        self.X = X.float()
        self.Y = Y.float()

    def __len__(self)-> int:
        return len(self.X)

    def __getitem__(self, idx)-> tuple:
        x_sample = self.X[idx]
        y_sample = self.Y[idx]
        return x_sample, y_sample
    
def dataset_creation(csv_path: str, target_folder: str)-> CustomDataset:

    num = []    
    im_target = []
    df = pd.read_csv(csv_path, index_col=0)

    for index, row in tqdm(df.iterrows(), desc='Pre-processing Data ...'):
        target_image_path = target_folder + "/" +str(index) + '.jpg'
        img_target = cv2.imread(target_image_path, cv2.IMREAD_GRAYSCALE)
        img_target = ((img_target / 255.0) * 2) - 1  # Normalize to [-1, 1]
        im_target.append(img_target)
        num.append([float(row['input1']), float(row['input2']), float(row['input3'])])

    num = torch.tensor(np.array(num)).float()
    im_target = torch.tensor(np.array(im_target)).view(-1, 1, 512, 512).float()
    Data_tg = CustomDataset(num, im_target)
    return Data_tg

