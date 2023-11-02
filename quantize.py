import torch
import os
import shutil
import numpy as np
import yaml
import matplotlib.pyplot as plt
from model import test
from torch.utils import data
from DataReader import DataReader
from Dataset import Dataset
from pathlib import Path
import argparse
import nemo
from copy import deepcopy
from inference import testing_model, test_single_image
from utils import get_fc_quantum
    
def print_summary(model,dummy_input_net):
    summary = nemo.utils.get_summary(model,tuple(torch.squeeze(dummy_input_net, 0).size()),verbose=True)
    print(summary['prettyprint'])

def network_size(model,dummy_input_net):
    summary = nemo.utils.get_summary(model,tuple(torch.squeeze(dummy_input_net, 0).size()),verbose=True)
    params_size = 0
    for layer_name, layer_info in summary['dict'].items():
        try:
            params_size += abs(layer_info["nb_params"]  * layer_info["W_bits"] / 8. / (1024.))
        except KeyError:
            params_size += abs(layer_info["nb_params"] * 32. / 8. / (1024.))
    return int(params_size)

def quantize_model():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', default='settings.yaml')
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # SETTINGS
    with open(args.file,'r') as f:
        settings = yaml.load(f, Loader=yaml.CSafeLoader)
    
    repo = settings['repo_folder_path']    
    test_folder_path = settings['test_imgs_folder_path'] 
    # trained_model = settings['trained_model_path']
    model_name  = settings['model_name']
    trained_model = settings['TL_trained_model_path']
    batch_size = settings['test_batch_size']
    test_yaml_path = test_folder_path + 'dataset.yaml'
    img_size = settings['img_size']
    channel = settings['input_channel']
    output_path = repo + 'outputs'
    
    input_size = (batch_size,channel,img_size[0],img_size[1])
    data_params = {'batch_size': batch_size, 'shuffle': True, 'num_workers': 0}  
    test_img_names,test_images, test_labels = DataReader.process_train_data(repo,test_folder_path,test_yaml_path,img_size=img_size,stage='test')
    testing_set = Dataset(test_images, test_labels,test_img_names)
    testing_loader = data.DataLoader(testing_set, **data_params)

    #=============================================================
    # Full Precision
    #=============================================================
    
    print("FP")
    
    dummy_input_net = torch.randn(input_size).to(device)
    state_dict = torch.load(trained_model,map_location=device)
    model_fp = test()
    print_summary(model_fp,dummy_input_net)
    model_fp.load_state_dict(state_dict)
    model_fp = model_fp.to(device)
    test_single_image(test_folder_path,model_fp,testing_loader,'cf7_00089.jpg',img_size=img_size)
    img_dict_fp = testing_model(test_folder_path,model_fp,testing_loader,img_size=img_size)
    
    # Store predictions for each image for stagewise comparison
    with open(test_folder_path+'1_fp.yaml', 'w') as fp:
        yaml.dump(img_dict_fp, fp, default_flow_style=False) 
    #=============================================================
    # FakeQuantized (FQ) stage
    #=============================================================  
    
    print("FQ")

    model_q = nemo.transform.quantize_pact(deepcopy(model_fp), dummy_input=dummy_input_net, remove_dropout=True)
    # print(model_q)
    model_q.change_precision(bits=8, scale_weights=False, scale_activations=True)
    model_q.change_precision(bits=7, scale_weights=True, scale_activations=False)
    
    
    # Load calibration dataset created during training. If not present, run train.py for 1 epoch
    calib_images = np.load('outputs/calib_imgs.npy')
    calib_labels = np.load('outputs/calib_labels.npy')
    calib_img_names = np.load('outputs/calib_names.npy')  
    
    calib_set = Dataset(calib_images, calib_labels,calib_img_names)
    calib_loader = data.DataLoader(calib_set, **data_params)
    
    # Recalibration to set alpha from a subset of training set
    with model_q.statistics_act(): 
        _ = testing_model(test_folder_path,model_q,calib_loader,img_size,id_stage='calib')  
    model_q.reset_alpha_act()
    
    img_dict = testing_model(test_folder_path,model_q,testing_loader,img_size=img_size)
    with open(test_folder_path+'2_fq.yaml', 'w') as fp:
        yaml.dump(img_dict, fp, default_flow_style=False) 

    #=============================================================
    # QuantizedDeployable (QD) stage
    #=============================================================
    
    print("QD")
    model_q.qd_stage(eps_in=1)
    img_dict = testing_model(test_folder_path,model_q,testing_loader,img_size=img_size)
    with open(test_folder_path+'3_qd.yaml', 'w') as fp:
        yaml.dump(img_dict, fp, default_flow_style=False) 
        
    #=============================================================
    # IntegerDeployable (ID) stage
    #=============================================================
    
    print("ID")
    model_q.id_stage()
    torch.save(model_q.state_dict(), 'outputs/'+model_name+'_ID_quant_TL.pt') 
    img_dict = testing_model(test_folder_path,model_q,testing_loader,img_size,id_stage=True)
    with open(test_folder_path+'4_id.yaml', 'w') as fp:
        yaml.dump(img_dict, fp, default_flow_style=False) 

    #=============================================================
    # Export ONNX and Activations for DORY checksums
    #=============================================================
    
    checksum_path = output_path+'/checksum'
    if os.path.exists(checksum_path): 
        shutil.rmtree(checksum_path)
    os.makedirs(checksum_path, exist_ok = True)
    
    # Export ONNX
    nemo.utils.export_onnx('outputs/checksum/testnet_ID_quant.onnx', model_q, model_q,input_size[1:])
    
    # Activation buffers
    print("Quantization Done")
    
    # Store final quantum to recreate the real-values from quantized output
    network_output_quantum = get_fc_quantum(model_q)
    settings['NEMO_QUANTUM']= round(network_output_quantum.detach().numpy().item(),4)
    with open(Path(repo) / 'settings.yaml', 'w') as f:
        yaml.dump(settings, f)
    print('network_output_quantum:', network_output_quantum)
    
    
    # Export one image from test folder for checksum
    test_single_image(test_folder_path,model_q,testing_loader,'cf7_00089.jpg',img_size=img_size,id_stage=True)
    buf_in, buf_out, _ = nemo.utils.get_intermediate_activations(model_q, test_single_image, test_folder_path,model_q,testing_loader,'cf7_00089.jpg',img_size,id_stage=True)
    # Save the input
    t = buf_in['conv2'][0][-1].cpu().detach().numpy()
    np.savetxt(os.path.join(checksum_path,'input.txt'), t.flatten(), '%.3f', newline=',\\\n', header = 'input (shape %s)' % str(list(t.shape)))

    # Save the output buffers
    names = ['relu2', 'pool2', 'relu3','pool3','relu4']
    for l in range(len(names)):
        t = np.moveaxis(buf_out[names[l]][-1].cpu().detach().numpy(), 0, -1)
        np.savetxt(os.path.join(checksum_path,'out_layer%d.txt') % l, t.flatten(), '%.3f', newline=',\\\n', header = names[l] + ' (shape %s)' % str(list(t.shape)))

    print('\nExport of golden activations was successful \n')
    

if __name__ == "__main__":
    quantize_model()