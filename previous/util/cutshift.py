import torch
import numpy as np
import random
import csv

class Cufshift(object):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes, length, coordinate):
        self.n_holes = n_holes
        self.length = length
        self.coordinate = coordinate

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        temp = img.clone()
        c = img.size(0)
        h = img.size(1)
        
        w = img.size(2)
        
        mask = np.ones((c, h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)
            
            x_coor = (int)((random.randrange(2) - 0.5) * 2)
            y_coor = (int)((random.randrange(2) - 0.5) * 2)
            
            y11 = np.clip(y1 + y_coor  * self.coordinate, 0, h)
            y22 = np.clip(y2 + y_coor  * self.coordinate, 0, h)
            x11 = np.clip(x1 + x_coor  * self.coordinate, 0, w)
            x22 = np.clip(x2 + x_coor  * self.coordinate, 0, w)

            y_mask = y2 - y1 
            x_mask = x2 - x1 
            y_mask2 = y22 - y11 
            x_mask2 = x22 - x11 
            
            if x_mask != x_mask2:
                if x_coor == -1 :
                    x22 = x22 + (x_mask - x_mask2)
                else :
                    x11 = x11 - (x_mask - x_mask2)
            if y_mask != y_mask2:
                if y_coor == -1 :
                    y22 = y22 + (y_mask - y_mask2)
                else :
                    y11 = y11 - (y_mask - y_mask2)
            
            mask[:, y11:y22, x11:x22] = temp[:, y1:y2, x1:x2]
            img[:, y11:y22, x11:x22] = 1.0

        mask = torch.from_numpy(mask)
        img = img * mask

        return img


def save_checkpoint(state, model_path):
    torch.save(state, model_path)
