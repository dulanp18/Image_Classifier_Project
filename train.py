import argparse
import torch
from collections import OrderedDict
from os.path import isdir
from torch import nn
import torch.nn.functional as F
from torch import optim
from torchvision import datasets, transforms, models


import matplotlib.pyplot as plt
import seaborn as sns
import PIL
import numpy as np

def argum_parser():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch',action='store',type=str, help='Choose among 3 pretrained networks - vgg16, alexnet, and densenet121')
    parser.add_argument('--hidden_units',action='store',type=int, help='Select number of hidden units for 1st layer')
    parser.add_argument('--learning_rate',action='store',type=float, help='Choose a float number as the learning rate for the model')
    parser.add_argument('--epochs',action='store',type=int, help='Choose the number of epochs you want to perform gradient descent')
    parser.add_argument('--save_dir',action='store', type=str, help='Select name of file to save the trained model')
    parser.add_argument('--gpu',action='store_true',help='Use GPU if available')
    
    args = parser.parse_args()
    return args

def data_prep(train_dir,valid_dir,test_dir):
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])
    
    val_transforms = transforms.Compose([transforms.Resize(255),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406],
                                                              [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])


# TODO: Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    val_data = datasets.ImageFolder(valid_dir, transform=train_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)

# TODO: Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=40, shuffle=True)
    valloader = torch.utils.data.DataLoader(val_data, batch_size=40, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=50)
    
    return trainloader,valloader,testloader,train_data

def switch_gpu(option):
    if not option:
        return torch.device("cpu")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if device.type == 'cpu':
        print('CUDA was not available, so using CPU')
    else:
        print('CUDA was available, so using CUDA')
    
    return device

def initiate_model(arch='vgg16'):
    
    model =  getattr(models,arch)(pretrained=True)
    #model = models.vgg16(pretrained=True)
    print("Network architecture specified as",arch,".")
    
    for param in model.parameters():
        param.requires_grad = False
        
    return model      

def define_classifier(model, hidden_units=2500):
    print("Number of Hidden Layers used",hidden_units,".")
    
    # Find Input Layers
    input_features = model.classifier[0].in_features
    
    classifier = nn.Sequential(nn.Linear(input_features, hidden_units),
                                 nn.ReLU(),
                                 nn.Dropout(0.2),
                                 nn.Linear(hidden_units, 102),
                                 nn.LogSoftmax(dim=1))
    
    return classifier
      
def train_validate_model(model, trainloader, valloader, device, criterion, optimizer, epochs=2):
    
    model.to(device)
  
 
    print("Training process initializing .....\n")
    
    steps = 0
    running_loss = 0
    print_every = 5
    for epoch in range(epochs):
        for inputs, labels in trainloader:
            steps += 1
         # Move input and label tensors to the default device
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
        
            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        
            if steps % print_every == 0:
                val_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in valloader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)
                    
                        val_loss += batch_loss.item()
                    
                        # Calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                    
                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"Val loss: {val_loss/len(valloader):.3f}.. "
                      f"Val accuracy: {accuracy/len(valloader):.3f}")
                running_loss = 0
                model.train()
    
    return model

def test_model(model,testloader,device,criterion):
    print('Testing model for accuracy.......')
    model.eval()
    with torch.no_grad():
        accuracy = 0 
        test_loss = 0 
        
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            logps = model.forward(inputs)
            batch_loss = criterion(logps, labels)
            test_loss += batch_loss.item()
                    
        # Calculate accuracy
            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                    
    print(f"Test accuracy: {accuracy/len(testloader)*100:.3f}")

def checkpoint(model,train_data,save_dir = "my_checkpoint.pth"):
    
    model.class_to_idx = train_data.class_to_idx
    checkpoint = {'class_to_idx': model.class_to_idx,
                  'classifier':model.classifier,
                  'state_dict': model.state_dict()}
    
    torch.save(checkpoint, save_dir)
    
    print('Your model has been successfully saved.')

def main():
    

    arch = 'vgg16'
    hidden_units = 2500
    learning_rate = 0.001
    epochs = 2
    option = 'cuda'
    save_dir= 'my_checkpoint.pth'
    
    args = argum_parser()
    
    if args.arch:
        arch = args.arch
    if args.hidden_units:
        hidden_units = args.hidden_units
    if args.learning_rate:
        learning_rate = args.learning_rate
    if args.epochs:
        epochs = args.epochs
    if args.gpu:        
        option = args.gpu
    if args.save_dir:
        save_dir = args.save_dir

    
    data_dir = 'flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    trainloader,valloader,testloader,train_data = data_prep(train_dir,valid_dir,test_dir)
    
    model = initiate_model(arch)
    
    model.classifier = define_classifier(model, hidden_units)
            
    device = switch_gpu(option)  
    
    
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    
    trained_model = train_validate_model(model, trainloader, valloader, device,criterion, optimizer, epochs)
    
    print("\nTraining process is now complete!!")
        
    test_model(trained_model,testloader,device,criterion)
    
    checkpoint(trained_model, train_data,save_dir)
            
main()