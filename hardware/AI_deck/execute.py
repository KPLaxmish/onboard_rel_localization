import os
import shutil
import yaml


NEMO_QUANTUM = 0.0471

def parse_GAP8_results(image_folder):
    img_dict = {}
    image_names,pix,z,measured_time,dist = [],[],[],[],[]
    count = 0
    names = os.listdir(image_folder)
    for name in sorted(names):
        if (name.endswith(".txt")): # Check only .txt file
            with open(image_folder+name, 'r') as f:
                for index, line in enumerate(f):
                    if ('Pix' in line) : # Use first 100 predictions only
                        dist.append(float(int(line.strip().split('dist is')[1])*NEMO_QUANTUM))
                        line = line.strip().split('Pix co-ordinates are ')[1].split(', measured time: ')
                        extract_pixel = line[0].lstrip("('").rstrip("')").split(",")
                        pix.append([int(extract_pixel[0]),int(extract_pixel[1])])
                        # measured_time.append(float(line[1]))
        # Keep track of all image names        
            # print(name.split('exp2_'))      
            id = int(name.split('exp2_')[1].split('.txt')[0])
            
            image_names.append(f"img_{id:06d}.png")                       
    # Store image_name, pix, and measured time in yaml
    for i,name in enumerate(image_names):
        img_dict[str(name)]={'pix': pix[i],'z-pos':dist[i],'measured time': 0}   
    
    return img_dict  

def parse_core_time(image_folder):
    img_dict = {}
    image_names,pix,z,dist = [],[],[],[]
    count = 0
    names = os.listdir(image_folder)
    for name in sorted(names):
        # print(name)
        measured_time = []
        if (name.endswith("core.txt")): # Check only .txt file
            with open(image_folder+name, 'r') as f:
                for index, line in enumerate(f):
                    if ('Pix' in line) : # Use first 100 predictions only
                        # dist.append(float(int(line.strip().split('dist is')[1])*NEMO_QUANTUM))
                        line = line.strip().split('Pix co-ordinates are ')[1].split(', measured time: ')
                        # print(line)
                        # extract_pixel = line[0].lstrip("('").rstrip("')").split(",")
                        # pix.append([int(extract_pixel[0]),int(extract_pixel[1])])
                        measured_time.append(float(line[1].split(', dist')[0]))
                print(round(sum(measured_time)/len(measured_time)*pow(10,-3),3),len(measured_time),name)
    
    return measured_time

folder_path = os.getcwd() + '/'

dst = folder_path+'hex/inputs.hex'

# for i in range(1,101):
#     src = folder_path+'load_images/inputs_{}.hex'.format(i)
#     print(src,dst)
#     shutil.copy(src, dst)
#     os.system("docker run --rm -it -v $PWD:/module/ --device /dev/ttyUSB0 --privileged -P bitcraze/aideck /bin/bash -c 'export GAPY_OPENOCD_CABLE=interface/ftdi/olimex-arm-usb-tiny-h.cfg; source /gap_sdk/configs/ai_deck.sh; cd /module/;  make build image flash run' | tee output_exp2_{}.txt".format(i))

# gap8_yaml_dict = (parse_GAP8_results(folder_path))
measured_time = (parse_core_time(folder_path))

# with open(folder_path+'3_gap8_core_1.yaml', 'w') as gap8:
#     yaml.dump(gap8_yaml_dict, gap8, default_flow_style=False) 