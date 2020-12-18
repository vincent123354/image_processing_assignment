import argparse
import os
import time
import pandas as pd
import numpy as np
import torch
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
from Dataset import Dataset
from Model import Net

from PIL import Image
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

parser = argparse.ArgumentParser()

parser.add_argument('--train-data-path', type=str, default='../../COVID-19 Radiography Database')
parser.add_argument('--model-path', type=str, default='./model')
parser.add_argument('--batch-size', type=int, default=16)
parser.add_argument('--epochs', type=int, default=1)

args = parser.parse_args()

DATA_PATH = args.train_data_path
BATCH_SIZE = args.batch_size
N_EPOCHS = args.epochs
MODEL_PATH = args.model_path

if not os.path.isdir(MODEL_PATH): # create model folder to save model if it doesn't exist
    os.mkdir(MODEL_PATH)

def train(net, dataloader, criterion, optimizer, device):
    """
    Method to run one epoch of training

    Attributes
    ----------
    net : nn.Module
        Model architecture
    dataloader : torch.utils.data.Dataloader
        Data iterator for the dataset
    criterion : 
        loss function for training
    optimizer : nn.optim
        optimizer for training
    device : str
        'cuda' or 'cpu'

    Returns
    -------
    Float
        Average training loss for this epoch
    """
    net.train()
    running_loss = 0
    for idx, (x, y) in enumerate(dataloader):
        print('\r{:.2f}%'.format((idx+1) / len(dataloader) * 100), end='')
        x = x.to(device)
        y = y.to(device).long()
        
        optimizer.zero_grad()
        
        predict = net(x)
        loss = criterion(predict.squeeze(), y)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    return running_loss / len(dataloader)

def evaluate(net, dataloader, device):
    """
    Method to run one epoch of evaluation

    Attributes
    ----------
    net : nn.Module
        Model architecture
    dataloader : torch.utils.data.Dataloader
        Data iterator for the dataset
    device : str
        'cuda' or 'cpu'

    Returns
    -------
    List, List
        list that contain prediction and true label respectively
    """

    net.eval()
    predicts = []
    labels = []
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device).float()

            predict = net(x)
            predict = predict.argmax(1)
            predicts.append(predict.cpu().numpy())
            labels.append(y.cpu().numpy())
    return np.concatenate(predicts), np.concatenate(labels)

def epoch_time(start, end):
    """
    Method to compute time for one epoch

    Attributes
    ----------
    start : Long / int
        start time
    end : Long / int
        end time

    Returns
    -------
    int, int
        time in minute and second format
    """
    diff = end - start
    m = int(diff / 60)
    s = int(diff - m * 60)
    return m, s

def main():

    # obtain all training image in DATA_PATH
    covid_imgs = [os.path.join(DATA_PATH, 'COVID-19', i) for i in os.listdir(os.path.join(DATA_PATH, 'COVID-19'))]
    normal_imgs = [os.path.join(DATA_PATH, 'NORMAL', i) for i in os.listdir(os.path.join(DATA_PATH, 'NORMAL'))]
    pneu_imgs = [os.path.join(DATA_PATH, 'Viral Pneumonia', i) for i in os.listdir(os.path.join(DATA_PATH, 'Viral Pneumonia'))]
    imgs = covid_imgs + normal_imgs + pneu_imgs

    # define label
    labels = np.r_[np.zeros((len(covid_imgs), )), np.ones((len(normal_imgs), )), np.ones((len(pneu_imgs), )) * 2]

    # split dataset
    df = pd.DataFrame({'imgs':imgs, 'labels':labels})
    df_train, df_test = train_test_split(df, test_size=0.1)

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = Dataset(df_train, transform)
    test_dataset = Dataset(df_test, transform)

    train_iterator = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_iterator = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # use resnet for transfer learning
    resnet = models.resnet18(pretrained=True)
    resnet = nn.Sequential(*list(resnet.children())[:-1])

    net = Net(resnet).to(device)

    # define weight for loss function
    weight = df_train.labels.value_counts().sort_values().values
    weight = torch.FloatTensor(weight.max() / weight).to(device)

    criterion = nn.CrossEntropyLoss(weight=weight)
    optimizer = optim.Adam(net.parameters())

    for epoch in range(N_EPOCHS):
        start = time.time()
        train_loss = train(net, train_iterator, criterion, optimizer, device)
        end = time.time()
        
        m, s = epoch_time(start, end)
        
        print(f'Epoch {epoch+1} | Time : {m}m {s}s')
        print(f'Train Loss : {train_loss:.4f}')
        print()

    torch.save(net.state_dict(), os.path.join(MODEL_PATH, 'net.pth'))

    # evaluate model with various metrics
    pre, label = evaluate(net, test_iterator, device)
    accuracy = (pre == label).mean() * 100
    conf_mat = confusion_matrix(label, pre)
    f1 = f1_score(label, pre, average='macro')
    precision = precision_score(label, pre, average='macro')
    recall = recall_score(label, pre, average='macro')

    print("Evaluation Result")
    print(f"Accuracy : {accuracy:.2f}")
    print(f"F1 score : {f1:.2f}")
    print(f"Precision : {precision:.2f}")
    print(f"Recall : {recall:.2f}")
    print(f"Confusion Matrix")
    print(conf_mat)

if __name__ == "__main__":
    main()