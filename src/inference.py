import argparse
import os
import torch
import shutil
import numpy as np
import torchvision.models as models
import torch.nn as nn
import matplotlib.pyplot as plt

from Dataset import Dataset
from Model import Net
from torchvision import transforms
from PIL import Image

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser()

parser.add_argument('--image-path', type=str, default='./test')
parser.add_argument('--model-path', type=str, default='./model/net.pth')
parser.add_argument('--save-result', type=str2bool, default=False)

args = parser.parse_args()

DATA_PATH = args.image_path
MODEL_PATH = args.model_path
SAVE_RESULT = args.save_result

# if result folder not exist and save_result is set to true, create the result folder
if SAVE_RESULT and not os.path.isdir(os.path.join(DATA_PATH, 'result')):
    os.mkdir(os.path.join(DATA_PATH, 'result'))

# if result folder exist and save_result is set to true, delete and create new result folder
if SAVE_RESULT and os.path.isdir(os.path.join(DATA_PATH, 'result')):
    shutil.rmtree(os.path.join(DATA_PATH, 'result'))
    os.mkdir(os.path.join(DATA_PATH, 'result'))

# map prediction integer to class
class_map = {
	0: 'Covid',
	1: 'Normal',
	2: 'Viral Pneumonia'
}

def inference(net, image_path, transform, device):
    """
    Method perform inference on one image
    
    Attributes
    ----------
    net : Net class
        Trained model
    image_path : str
        Path to an image
    transform :
        Preprocessing step
    device : str
        'cuda' or 'cpu'

    Returns
    -------
    Tensor
        tensor that contain predicted probability for each class
    """
    net.eval()
    image = np.array(Image.open(image_path).convert('RGB'))
    image = transform(image).unsqueeze(0) # add one batch dimension to fit into model
    with torch.no_grad():
        image = image.to(device)
        predict = net(image)
    return predict[0]

def save_result(image_path, class_predictions):
    """
    Method to save image with its label into new image

    Attributes
    ----------
    image_path : str
        Path to an image
    class_predictions : Tensor
        Tensor that contain predicted probability for each class
    """
    image_name = os.path.split(image_path)[-1]

    label = class_predictions.argmax().item() # get index to max probability
    image = np.array(Image.open(image_path).convert('RGB'))
    plt.imshow(image)
    plt.title(str(class_map[label]) + ' : ' + str(class_predictions[label].item()))
    plt.axis('off')
    plt.savefig(os.path.join(DATA_PATH, 'result', image_name))
    plt.close()

def main():
    """
    Returns
    -------
    Dict
        dictionary that contain image path as key and its predicted class as value
    """
    predicts = []
    imgs = [os.path.join(DATA_PATH, i) for i in os.listdir(DATA_PATH) if i.endswith('.png')] # get all image path in DATA_PATH folder

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    resnet = models.resnet18()
    resnet = nn.Sequential(*list(resnet.children())[:-1])
    net = Net(resnet).to(device)
    net.load_state_dict(torch.load(MODEL_PATH))

    for img in imgs:
        predict = inference(net, img, transform, device)
        if SAVE_RESULT:
            save_result(img, predict)
        predicts.append(predict.argmax().item())

    return dict(zip(imgs, predicts))

if __name__=='__main__':
    main()