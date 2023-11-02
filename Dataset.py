import torch
class Dataset():
    def __init__(self, images, labels, names):
        self.images = torch.from_numpy(images)
        self.labels = torch.from_numpy(labels)
        self.folder_img_name = names
        length = len(self.images)
        self.list_IDs = range(0, length)
    
    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        ID = index
        X = self.images[ID]
        y = self.labels[ID]
        img_name = self.folder_img_name[ID]
        return img_name, X, y