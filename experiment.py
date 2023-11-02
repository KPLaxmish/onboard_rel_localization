import cv2
import torch
import nemo
from copy import deepcopy
import os
import numpy as np
import yaml
import matplotlib.pyplot as plt
from model import test
from torch.utils import data
from DataReader import DataReader
from Dataset import Dataset
import argparse
from plot import PlotViolin, PlotWhisker, PlotWhiskerALL, PlotonImage
from inference import testing_model,test_single_image
from utils import get_euclidean_err_3D, resize_images
    



parser = argparse.ArgumentParser()
parser.add_argument('--file', default='settings.yaml')
args = parser.parse_args()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# SETTINGS
with open(args.file,'r') as f:
    settings = yaml.load(f, Loader=yaml.CSafeLoader)
    
NEMO_QUANTUM = settings['NEMO_QUANTUM']
repo = settings['repo_folder_path']  
fp_model = settings['TL_trained_model_path']
test_folder_path = settings['test_imgs_folder_path'] 
batch_size = settings['test_batch_size']
img_size = settings['img_size']
channel = settings['input_channel']
experiments_folder = settings['experiments_folder']
plot_folder = experiments_folder+ 'plots/'
no_of_exp = settings['no_of_experiments']
output_path = repo + 'outputs'
input_size = (batch_size,channel,img_size[0],img_size[1])
data_params = {'batch_size': batch_size, 'shuffle': True, 'num_workers': 0}  

def load_quantized(model,input_size,device):
    dummy_input_net = torch.randn(input_size).to(device)
    model_q = nemo.transform.quantize_pact(deepcopy(model), dummy_input=dummy_input_net, remove_dropout=True)
    # print(model_q)
    model_q.change_precision(bits=8, scale_weights=False, scale_activations=True)
    model_q.change_precision(bits=7, scale_weights=True, scale_activations=False)
    
    calib_images = np.load('outputs/calib_imgs.npy')
    calib_labels = np.load('outputs/calib_labels.npy')
    calib_img_names = np.load('outputs/calib_names.npy')  
    calib_set = Dataset(calib_images, calib_labels,calib_img_names)
    calib_loader = data.DataLoader(calib_set, **data_params)
    
    # Recalibration to set alpha from a subset of training set
    with model_q.statistics_act(): 
        _ = testing_model(test_folder_path,model_q,calib_loader,img_size,id_stage='calib')  
    model_q.reset_alpha_act()
    
    model_q.qd_stage(eps_in=1)
    
    model_q.id_stage()
    
    return model_q


def inference_real_exp(model,image_folder,id_stage=False):
    img_dict = {}
    #Open folder and get list of images , sort the names and make a yaml file with initial labels 0
    names = os.listdir(image_folder)
    
    # Create template yaml file , DONE ONCE
    # for name in sorted(names):
    #     if name.endswith(".png"):
    #         img_dict[name]={'pix': [0,0],'z-pos':0}    
        
    # with open(experiments_folder+'dataset_template.yaml', 'w') as outfile:
    #     yaml.dump(img_dict, outfile, default_flow_style=False)
    
    img_names,images, labels = DataReader.process_train_data(repo,image_folder,experiments_folder+'dataset_template.yaml',img_size,stage='test')
    testing_set = Dataset(images, labels,img_names)
    testing_loader = data.DataLoader(testing_set, **data_params)
    model.eval()
    for i, dataset in enumerate(testing_loader):
        image_names,inputs,labels = dataset
        inputs = inputs.to(device)
        labels = labels.to(device)
        pred = model(inputs)
        # print(pred.shape)
        max_pred = np.argmax(pred[:,0:1,:,:].detach().cpu(), axis=None)
        ind = np.unravel_index(max_pred, pred.detach().cpu().shape)
        curH = int(ind[2]*8) # Height corresponding to y
        curW = int(ind[3]*8) # Width corresponding to x
        z =  (pred[:,1,ind[2],ind[3]]).detach().numpy().item()
        if id_stage:
            z =  z*NEMO_QUANTUM
            
        # print([curW,curH])
        img_dict[str(image_names[0])] ={'pix': [curW,curH],'z-pos':z}   
    return img_dict

def pos3D_from_pix(pix,depth):
    u= pix[0]
    v= pix[1]
    with open(experiments_folder+'calibration.yaml') as f:
        camera_config = yaml.load(f, Loader=yaml.FullLoader)
    camera_intrinsic = np.array(camera_config['camera_matrix'])
    X_c = ((u - camera_intrinsic[0][2]) / camera_intrinsic[0][0])* depth
    Y_c = ((v - camera_intrinsic[1][2]) / camera_intrinsic[1][1])* depth
    Z_c = depth
    return [X_c,Y_c,Z_c]

    

    
def compare_yaml(folder,yaml_files,exp_no):
    
    err_fp_id_x,err_fp_id_y,err_id_g8_x,err_id_g8_y,err_id_g8_dis,err_fp_id_dis,err_fp_g8_x,err_fp_g8_y,err_fp_gap8_dis,err_fp_g8_3Dpos = [],[],[],[],[],[],[],[],[],[]
    # Read two yaml files and compare pixel predictions for each image
    # print(yaml1,yaml2)
    with open(folder+yaml_files[0],'r') as f1:
        yaml1_data = yaml.load(f1, Loader=yaml.CSafeLoader)
    with open(folder+yaml_files[1],'r') as f2:
        yaml2_data = yaml.load(f2, Loader=yaml.CSafeLoader)
    with open(folder+yaml_files[2],'r') as f3:
        yaml3_data = yaml.load(f3, Loader=yaml.CSafeLoader)    
    

        
    for image_name, entry in yaml1_data.items():  
        
        # Compare corresponding distance predictions from yaml1 and yaml2
        err_fp_id_dis.append(entry['z-pos']-yaml2_data[image_name]['z-pos'])
        err_id_g8_dis.append(yaml2_data[image_name]['z-pos']- yaml3_data[image_name]['z-pos'])
        
        # Compare corresponding (x,y) pixels from yaml1 and yaml2
        err_fp_id_x.append(entry['pix'][0]-yaml2_data[image_name]['pix'][0])
        err_fp_id_y.append(entry['pix'][1]-yaml2_data[image_name]['pix'][1])
        
        err_id_g8_x.append(yaml2_data[image_name]['pix'][0]- yaml3_data[image_name]['pix'][0])
        err_id_g8_y.append(yaml2_data[image_name]['pix'][1]- yaml3_data[image_name]['pix'][1])
        
        err_fp_gap8_dis.append(entry['z-pos']-yaml3_data[image_name]['z-pos'])
        err_fp_g8_x.append(yaml1_data[image_name]['pix'][0]- yaml3_data[image_name]['pix'][0])
        err_fp_g8_y.append(yaml1_data[image_name]['pix'][1]- yaml3_data[image_name]['pix'][1])
        
        # err_data = [err_fp_id_x, err_fp_id_y,err_id_g8_x,err_id_g8_y,err_fp_id_dis,err_id_g8_dis]
        err_data = [err_fp_id_x,err_fp_id_y,err_fp_id_dis]
        # Plots 
        img = image_name.split('.png')[0]
        PlotonImage.plot(int(img.split('img_')[1]),experiments_folder+'resized/'+image_name,entry['pix'],yaml2_data[image_name]['pix'],yaml3_data[image_name]['pix'],entry['z-pos']-yaml2_data[image_name]['z-pos'],yaml2_data[image_name]['z-pos']- yaml3_data[image_name]['z-pos'],plot_folder+'{}_{}.pdf'.format(exp_no,img))

        err_fp_g8_3Dpos.append(get_euclidean_err_3D(pos3D_from_pix(yaml1_data[image_name]['pix'],yaml1_data[image_name]['z-pos']),pos3D_from_pix(yaml3_data[image_name]['pix'],yaml3_data[image_name]['z-pos']))) 
        euc_err = [err_fp_g8_3Dpos]
        
    # np.save('euc_err_320',err_fp_g8_3Dpos)
    # MAE = (np.mean(np.array([abs(x) for x in err_fp_g8_3Dpos])), np.std(np.array([abs(x) for x in err_fp_g8_3Dpos])))
    # print(round(MAE[0],3),round(MAE[1],3))
    
    return err_data,euc_err

if __name__ == "__main__":
    rmse_x1,rmse_x2,rmse_y1,rmse_y2 = [],[], [], []
    fp_yaml_dict,id_yaml_dict = {},{}
    state_dict_fp = torch.load(fp_model,map_location=device)
    test_model_fp = test()
    test_model_fp.load_state_dict(state_dict_fp)
    test_model_fp = test_model_fp.to(device)
    test_model_id = load_quantized(test_model_fp,input_size,device)
    

    for exp_no in range(1,no_of_exp+1):
        # images_folder = experiments_folder+'exp_{}/'.format(exp_no)
        images_folder = experiments_folder
        
        resize_images(img_size,images_folder,experiments_folder+'resized/')

        # Run inference on captured images , via the FP model, output a dict with predictions
        fp_yaml_dict = inference_real_exp(test_model_fp,images_folder)
        # Store the dict into a yaml file
        with open(images_folder+'1_fp_'+ '{}.yaml'.format(exp_no), 'w') as fp:
            yaml.dump(fp_yaml_dict, fp, default_flow_style=False) 
    
        # Run inference on captured images , via the ID model, output a dict file with predictions
        id_yaml_dict = inference_real_exp(test_model_id,images_folder,id_stage=True)
        # Store the dict into a yaml file
        with open(images_folder+'2_id_'+ '{}.yaml'.format(exp_no), 'w') as id:
            yaml.dump(id_yaml_dict, id, default_flow_style=False) 

        
        # Compare results : FP vs ID, ID vs GAP8
        yaml_files = []
        filenames = os.listdir(images_folder)
        for name in sorted(filenames):
            if (name.endswith(".yaml")): # Check only .yaml
                yaml_files.append(name)
        
        errors,error_3d_pos = compare_yaml(images_folder,yaml_files,exp_no)    
        PlotViolin.plot(errors[2:],plot_folder+'{}_err_dist.pdf'.format(exp_no),'Distance Error (FP vs GAP8)')    
        PlotViolin.plot(errors[:2],plot_folder+'{}_err_pix.pdf'.format(exp_no),'2D Position Error (FP vs GAP8)')            

    PlotWhisker.plot(error_3d_pos,plot_folder+'E1.pdf','Euclidean Distance Error in 3D relative position')


    
    # euc_err = [np.load('euc_err_320.npy'),np.load('euc_err_224.npy'),np.load('euc_err_160.npy'),np.load('euc_err_96.npy')]
    # PlotWhiskerALL.plot(euc_err,'EUC_error_all','Euclidean Distance Error in 3D relative position')
    