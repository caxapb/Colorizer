from torch.utils.data import DataLoader
from data.functions import get_urls
from data.Dataset import PictureDataset
from data.mlFunctions import *
from sklearn.model_selection import train_test_split
from model import PictureColorizer
import torch
import torch.optim as optim
import pickle
import os

if __name__ == "__main__":
    path = os.path.dirname(__file__) + '/loaders/'
    
    with open(path + 'validation_loader.pkl', 'rb') as file:
        val_dataloader = pickle.load(file)
    with open(path + 'train_loader.pkl', 'rb') as file:
        train_dataloader = pickle.load(file)

    model = PictureColorizer()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = torch.nn.MSELoss()
    epochs = 10
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    print("start train")
    train(model, train_dataloader, val_dataloader, optimizer, loss_fn, epochs, device=device, plotting=True)
