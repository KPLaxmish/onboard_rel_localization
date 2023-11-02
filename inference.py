import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from time import perf_counter
import cv2
from utils import get_fc_quantum, re_project_on_image
from Loss import mse_loss, weighted_mse_loss

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def test_single_image(test_folder_path,test_model,testing_loader,test_image_name,img_size,id_stage=False):
    test_model.eval()
    for i, dataset in enumerate(testing_loader):
        image_names,inputs,labels= dataset
        image_names= list(image_names)
        #print(image_names,type(image_names))
        try:
            index = image_names.index(test_image_name)
            single_input = inputs[index]
            single_input = single_input[..., np.newaxis]
            single_input = np.swapaxes(single_input, 1, 3)
            single_input = np.swapaxes(single_input, 2, 3) # <1, 320, 320>

            pred = test_model(single_input)
            max_pred = np.argmax(pred[:,0:1,:,:].detach(), axis=None)
            ind = np.unravel_index(max_pred, pred.detach().shape)
            curH = ind[2]*8 # Height corresponding to y
            curW = ind[3]*8 # Width corresponding to x
            # print(max_pred)
            
            # Reproject the predicted pixel onto test image for manual verification
            if(id_stage==False):
                re_project_on_image(test_folder_path + test_image_name,(curW,curH),img_size,'single_projected_img/'+'fp_result_'+test_image_name)
            elif(id_stage==True):    
                re_project_on_image(test_folder_path + test_image_name,(curW,curH),img_size,'single_projected_img/'+'id_result_'+test_image_name)
            
            z = (pred[:,1,ind[2],ind[3]]).detach().numpy().item()
            # print(z)
            print("%s : predicted center (x,y) = (%d,%d), distance =%f" % (image_names[index],curW,curH,z))
            #print(image_names,curH,pix_gt[1],curW,pix_gt[0])
        except ValueError:
            index = -1

                
def testing_model(test_folder_path,test_model,testing_loader,img_size,id_stage=False):
    images = []
    img_dict = {}
    test_loss_weighted_conf,test_loss_z,test_loss_total = [],[],[]
    test_model.eval()
    start = perf_counter()
    with torch.no_grad():
        for i, dataset in enumerate(testing_loader):
            image_names,inputs,labels = dataset
            inputs = inputs.to(device)
            # print(inputs.shape)
            labels = labels.to(device)
            pred = test_model(inputs)
            
            # For ID stage propagate the network output quantum (i.e output quantum of last layer to the predicted values before error calculation)
            if(id_stage==True):
                fc_quantum = get_fc_quantum(test_model)
                pred_update = pred*fc_quantum
            else:
                pred_update = pred
                
            # np.argmax => returns the indices of the maximum values along an axis.By default, the index is into the flattened array.
            max_pred = np.argmax(pred_update[:,0:1,:,:].detach().cpu(), axis=None)
            # np.unravel_index(indices,shape) => get x,y = (ind/column_size, ind%column_size) i.e  from ind = (x*column_size+y)
            ind = np.unravel_index(max_pred, pred_update.detach().cpu().shape)
            
            # get predictions
            curH = ind[2]*8 # Height corresponding to y
            curW = ind[3]*8 # Width corresponding to x
            
            images.append(image_names[0])
        
            z =  (pred_update[:,1,ind[2],ind[3]]).detach().numpy().item()
                
            img_dict[str(image_names[0])]={'pix': [int(curW),int(curH)] ,'z-pos':z}   
            
            # Reproject the predicted pixel onto test image for manual verification
            # if(id_stage==True):
            #     re_project_on_image(test_folder_path + image_names[0],(curW,curH),img_size,'projected_img/'+'id_result_'+str(image_names[0]))
                
            # Test_loss estimations
            # criterion_weighted = nn.BCEWithLogitsLoss(reduction='mean', pos_weight=torch.tensor(1599.)) # negative/positive
            # loss_weighted_conf = criterion_weighted(pred[:,0:1,:,:], labels[:,0:1,:,:])    
            
            # loss_z = weighted_mse_loss(pred[:,1:2,:,:], labels[:,1:2,:,:], labels[:,0:1,:,:]) # 1599.*
            # loss_total = loss_weighted_conf + loss_z
            # test_loss_weighted_conf.append(loss_weighted_conf)    
            
    end = perf_counter()
    print("Time taken for test is {} min.".format((end-start)/60.))

    return img_dict#,test_loss_z,test_loss_total

if __name__ == "__main__":
    testing_model()