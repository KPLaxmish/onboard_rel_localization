import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec
import numpy as np
import os
import cv2

class PlotHist:
    def plot(eucl_error, name):
        plt.hist(eucl_error)
        plt.title('Histogram for Eucl. in m. with estimation')
        plt.savefig(name)
        # plt.show()
        
class PlotWhisker:
    def plot(eucl_error, name, title_name):
        fig = plt.figure(figsize =(5, 5),constrained_layout=True)
        plt.title(title_name)
        plt.ylabel('Euclidean Distance Error (in m)')
        plt.boxplot(eucl_error)
        plt.grid(axis='y')
        plt.xticks([1],['FP vs GAP8'])
        plt.savefig(name)
        # plt.show()
class PlotWhiskerALL:
    def plot(eucl_error, name, title_name):
        fig = plt.figure(figsize =(7, 5),constrained_layout=True)
        plt.title(title_name)
        plt.ylabel('Euclidean Distance Error (in m)')
        plt.boxplot(eucl_error)
        plt.grid(axis='y')
        plt.xticks([1,2,3,4],['320x320','224x224','160x160','160x96'])
        plt.savefig(name)
        # plt.show()        
class PlotScatter:
    def plot(x_t,y_t,x,y, name, title_name):
        ticks = [i for i in range(0,321,32)]
        if 'FP' in title_name:
            a = 'FP'
            b = 'ID'
        elif 'GAP8' in title_name:
            a = 'ID'
            b = 'GAP8'    
        fig = plt.figure(figsize =(5, 5))
        plt.title(title_name)
        plt.xlabel('x (pixels)')
        plt.ylabel('y (pixels)')

        plt.scatter(x_t,y_t, c='tab:blue',marker='x',s=100,alpha=.1,label=a)
        plt.scatter(x, y, c='tab:orange', alpha=.1,label=b)
        plt.xticks(ticks=ticks,labels=ticks)
        plt.yticks(ticks=ticks,labels=ticks)
        leg = plt.legend()
        for lh in leg.legendHandles: 
            lh.set_alpha(1)
        plt.savefig(name)
        plt.close()
        
# Invert plots to match openCV, if necessary        
# ax=plt.gca()                            # get the axis
# ax.set_ylim(ax.get_ylim()[::-1])        # invert the axis
# ax.xaxis.tick_top()                     # and move the X-Axis      
# ax.yaxis.set_ticks(np.arange(0, 16, 1)) # set y-ticks
# ax.yaxis.tick_left()                    # remove right y-Ticks
    
class PlotViolin:
    def plot(error, name, title_name):
        fig ,ax = plt.subplots(figsize =(5, 5),constrained_layout=True)
        ax.yaxis.grid(True)
        ax.set_xticks([y+1 for y in range(len(error))])
        ax.set_title(title_name,fontsize=18)
        if len(error)<2:
            ax.set_xticklabels(['d'],fontsize=18)
            ax.set_ylabel('Distance error (in m)',fontsize=18)
        else:
            ax.set_xticklabels(['xp','yp'],fontsize=18)    
            ax.set_ylabel('Position error (in pix)',fontsize=18)
        ax.violinplot(error,showmeans=True,showmedians=True)
        plt.savefig(name)
        # plt.show(),,
        plt.close() 
        
class PlotonImage:
    def plot(no,image,p1,p2,p3,dis1,dis2,name):
        fig ,ax = plt.subplots(figsize =(5, 5),constrained_layout=True)
        im = plt.imread(image)
        implot = plt.imshow(im)
        # To identify origin
        # plt.scatter([0], [0])
        plt.title('Prediction Comparison on Image {}'.format(no))
        plt.axis('off')
        plt.scatter(p1[0],p1[1], c='aqua',marker='x',s=120,label='FP',alpha=1)
        plt.scatter(p2[0],p2[1], c='tab:red',s=40,label='ID',alpha=1)
        plt.scatter(p3[0],p3[1], c='yellow',marker='o',s=15,label='GAP8',alpha=1)
        plt.text(p3[0],p3[1]-10, f'{round(dis1,3)}', fontsize = 10,color = 'w', bbox = dict(facecolor = 'w', alpha = 0.1)) 
        leg = plt.legend()
        plt.savefig(name)       
        plt.close()
        
        