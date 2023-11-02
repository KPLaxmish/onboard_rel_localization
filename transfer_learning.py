from DataReader import DataReader
from Dataset import Dataset
from torch.utils import data
import torch
import torch.nn as nn
import os
from Loss import mse_loss, weighted_mse_loss
import matplotlib.pyplot as plt
from Loss import mse_loss
from model import test
from time import perf_counter
from pathlib import Path
import argparse
from tqdm import trange, tqdm
import yaml
import argparse
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', default='settings.yaml')
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # SETTINGS
    with open(args.file,'r') as f:
        settings = yaml.load(f, Loader=yaml.CSafeLoader)

    repo = settings['repo_folder_path']
    folder = settings['transfer_train_imgs_folder_path']
    trained_model = settings['trained_model_path']
    yaml_path = folder + 'dataset_train.yaml'
    lr = settings['learning_rate']
    TL_model_name = settings['model_name'] + '_TL'
    epochs = settings['num_epochs']
    batch_size = settings['train_batch_size']
    img_size = settings['img_size']
    output_path = repo + 'outputs'
    
    data_params = {'batch_size': batch_size, 'shuffle': True, 'num_workers': 0}     
    train_img_names,train_images, train_labels,val_img_names,val_images, val_labels = DataReader.process_train_data(repo,folder,yaml_path,img_size=img_size,stage='train')
    training_set = Dataset(train_images, train_labels,train_img_names)
    training_loader = data.DataLoader(training_set, **data_params)
    
    # Validation dataset images
    testing_set = Dataset(val_images, val_labels,val_img_names)
    testing_loader = data.DataLoader(testing_set, **data_params)

    net = test()
    net.load_state_dict(torch.load(trained_model,map_location=device))
    
    net.to(device)
    print('Training on {}'.format(device))

    os.makedirs(repo+'outputs', exist_ok = True)
    optimizer = torch.optim.Adam(net.parameters(), lr) # lr=0.0001, weight_decay=0.005
    saved_model_name = os.path.join(repo+'outputs', '{}.pt'.format(TL_model_name))
    criterion_weighted = nn.BCEWithLogitsLoss(reduction='mean', pos_weight=torch.tensor(1599.)) # negative/positive
    
    train_loss, test_loss, train_loss_z, train_loss_conf = [], [], [], []
    start = perf_counter()
    # train_epoch_loss = torch.inf
    for epoch in trange(epochs): 
        net.train() # IMPORTANT STATUS // 
        running_loss, test_running_loss, running_loss_z, running_loss_conf = 0.0, 0.0, 0.0, 0.0

        for i, dataset in enumerate(training_loader):
            _, inputs,labels = dataset
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            # prediction
            outputs = net(inputs) # <8, 2, 40, 40>
            loss_weighted_conf = criterion_weighted(outputs[:,0:1,:,:], labels[:,0:1,:,:])  
            loss_z = weighted_mse_loss(outputs[:,1:2,:,:], labels[:,1:2,:,:], labels[:,0:1,:,:])
            loss_total = loss_weighted_conf + loss_z   
            loss_total.backward()
            optimizer.step()  
            running_loss += loss_total.item()*inputs.shape[0]   
            running_loss_z += loss_z.item()*inputs.shape[0]    
            running_loss_conf += loss_weighted_conf.item()*inputs.shape[0]     
    
        train_epoch_loss =  running_loss/len(training_loader.sampler) 
        train_epoch_loss_z =  running_loss_z/len(training_loader.sampler)
        train_epoch_loss_conf =  running_loss_conf/len(training_loader.sampler)
        
        net.eval()
        #with torch.no_grad():
        for j, test_dataset in enumerate(testing_loader):
            test_image_names, test_inputs, test_labels = test_dataset
            test_inputs = test_inputs.to(device)
            test_labels = test_labels.to(device)
            # prediction
            test_outputs = net(test_inputs)    
            # loss calculation   
            test_loss_weighted_conf = criterion_weighted(test_outputs[:,0:1,:,:], test_labels[:,0:1,:,:])     
            test_loss_z = weighted_mse_loss(test_outputs[:,1:2,:,:], test_labels[:,1:2,:,:], test_labels[:,0:1,:,:]) 
            test_loss_total = test_loss_weighted_conf + test_loss_z 
            test_running_loss += test_loss_total.item()*test_inputs.shape[0] 

        val_epoch_loss = test_running_loss/len(testing_loader.sampler)
        print('Epoch {}, train_loss {}, test_loss {}'.format(epoch, train_epoch_loss, val_epoch_loss))

        train_loss.append(train_epoch_loss)
        test_loss.append(val_epoch_loss)
        train_loss_z.append(train_epoch_loss_z)
        train_loss_conf.append(train_epoch_loss_conf)
    end = perf_counter()
    
    torch.save(net.state_dict(), saved_model_name)
    settings['TL_trained_model_path']=saved_model_name
    # save settings directly to result folder for later use
    path = Path(repo)
    print(path)
    with open(path / 'settings.yaml', 'w') as f:
        yaml.dump(settings, f)
    
    fig1 = plt.figure("Figure 1")
    plt.plot(train_loss, color='green', label='Training loss')
    plt.plot(test_loss, color='black', label='Testing loss')
    plt.legend()
    plt.xlabel('epochs')
    plt.ylabel('loss')
    fig1.savefig(os.path.join(output_path, TL_model_name + '.jpg'))      
    fig2 = plt.figure("Figure 2")
    plt.plot(train_loss_z, color='green', label='Training loss z')
    plt.plot(train_loss_conf, color='black', label='Training loss conf')
    plt.legend()
    plt.xlabel('epochs')
    plt.ylabel('loss')
    fig2.savefig(os.path.join(output_path,TL_model_name + '_individual'+'.jpg'))
    
    print("Time taken for {} epochs is: {}".format(epochs, (end-start)/60.))
    print('Finished Training')

if __name__ == '__main__':
    main()