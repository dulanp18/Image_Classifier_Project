import argparse
import json
import PIL
import torch
import numpy as np
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
import seaborn as sns



def argum_parser():
    # Define a parser
    parser = argparse.ArgumentParser()

    # Point towards image for prediction
    parser.add_argument('--image_path', 
                        type=str, 
                        help='Point to impage file for prediction.',
                        required=True)

    # Load checkpoint created by train.py
    parser.add_argument('--checkpoint', 
                        type=str, 
                        help='Point to checkpoint file as str.')
    
    # Specify top-k
    parser.add_argument('--top_k', 
                        type=int, 
                        help='Choose top K matches as int.')
    
    # Import category names
    parser.add_argument('--category_names', 
                        type=str, 
                        help='Mapping from categories to real names.')

    # Add GPU Option to parser
    parser.add_argument('--gpu', 
                        action="store_true", 
                        help='Use GPU + Cuda for calculations')

    # Parse args
    args = parser.parse_args()
    
    return args

def initiate_model(arch='vgg16'):
    
    model =  getattr(models,arch)(pretrained=True)
    #model = models.vgg16(pretrained=True)
    print("Network architecture specified as",arch,".")
    
    for param in model.parameters():
        param.requires_grad = False
        
    return model

def switch_gpu(option):
    if not option:
        return torch.device("cpu")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if device.type == 'cpu':
        print('CUDA was not available, so using CPU')
    else:
        print('CUDA was available, so using CUDA')
    
    return device

def load_checkpoint(check_dir,model,device):
    model.to(device)
    checkpoint = torch.load(check_dir)
        
    model.class_to_idx = checkpoint['class_to_idx']
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict']) 
    
    return model

def process_image(file_path):
    
    
    # TODO: Process a PIL image for use in a PyTorch model
    test_image = PIL.Image.open(file_path)
     
    
    image_transforms = transforms.Compose([transforms.Resize(256),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

    
    processed_image = image_transforms(test_image)
    
    return processed_image

def predict(processed_image, model, device, top_k=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    image = torch.tensor(processed_image).unsqueeze_(0).float()
    image = image.to(device)
    
    model.eval()
    
    for param in model.parameters(): 
        param.requires_grad = False
        
    logps = model.forward(image)
    ps = torch.exp(logps)
    probs, classes = ps.topk(top_k, dim=1)
        
    return probs, classes


def classify_image(processed_image,probs,classes,category_names='cat_to_name.json'):
    
    with open(category_names, 'r') as f:
        cat_to_name = json.load(f)
    
    
    labels = np.array(classes)
    probs = list(np.array(probs)[0])
    flowers = [cat_to_name[lab.astype('str')] for lab in labels[0]]
    
    for result in zip(flowers,probs):
        print(result)
    
    
def main():
    
    check_dir = 'my_checkpoint.pth'
    catergory_names = 'cat_to_name.json'    
    arch='vgg16'
    image_path = 'flowers/test/100/image_07896.jpg'
    top_k = 5
    option = 'cuda'
    
    args = argum_parser()
    
    if args.checkpoint:
        check_dir = args.checkpoint
    if args.image_path:
        image_path = args.image_path
    if args.top_k:
        top_k = args.top_k
    if args.category_names:
        category_names = args.category_names
    if args.gpu:
        option = args.gpu
        
    model = initiate_model(arch='vgg16') 
    
    device = switch_gpu(option)
    
    trained_model = load_checkpoint(check_dir,model,device)
    
    processed_image = process_image(image_path)
    
    probs, classes = predict(processed_image, trained_model, device, top_k=5)
    
    classify_image(processed_image,probs,classes,category_names='cat_to_name.json')
    
main() 