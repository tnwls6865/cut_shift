import torch
import numpy as np
import random
import csv
import matplotlib.pyplot as plt


class Cutshift(object):
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
        
class rCutshift(object):
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
            
            x_coor = random.randint(-self.coordinate, self.coordinate)
            y_coor = random.randint(-self.coordinate, self.coordinate)
            
            y11 = np.clip(y1 + y_coor, 0, h)
            y22 = np.clip(y2 + y_coor, 0, h)
            x11 = np.clip(x1 + x_coor, 0, w)
            x22 = np.clip(x2 + x_coor, 0, w)

            y_mask = y2 - y1 
            x_mask = x2 - x1 
            y_mask2 = y22 - y11 
            x_mask2 = x22 - x11 
            
            if x_mask != x_mask2:
                if x_coor < 0 :
                    x22 = x22 + (x_mask - x_mask2)
                else :
                    x11 = x11 - (x_mask - x_mask2)
            if y_mask != y_mask2:
                if y_coor < 0 :
                    y22 = y22 + (y_mask - y_mask2)
                else :
                    y11 = y11 - (y_mask - y_mask2)
            
            mask[:, y11:y22, x11:x22] = temp[:, y1:y2, x1:x2]
            img[:, y11:y22, x11:x22] = 1.0

        mask = torch.from_numpy(mask)
        img = img * mask

        return img

class gaussian_Cutshift(object):
    def __init__(self, n_holes, length, coordinate):
        self.n_holes = n_holes
        self.length = length
        self.coordinate = coordinate


    def __call__(self, img):
        temp = img.clone()
        c = img.size(0)
        h = img.size(1)
        w = img.size(2)
        
        # gaussian 
        
        lam = np.random.beta(1, 1)
        cut_rat = np.sqrt(1. - lam)
        cut_w = np.int(w * cut_rat)
        cut_h = np.int(h * cut_rat)

        mu, sigma = int(h/2), 1.0   # mean and standard deviation
        s = np.random.normal(mu, sigma, 1)

        # uniform
        #cx = np.random.randint(W)
        #cy = np.random.randint(H)

        mask = np.ones((c, h, w), np.float32)

        r = np.random.rand(1)
        if r < 0.8:
            for n in range(self.n_holes):
                
                # gaussian
                s = np.random.normal(mu, sigma, 2)
                where = np.random.randint(4)
        
                x, y = s
                x1 = int(x - (self.length/2))
                y1 = int(y - (self.length/2))
                x2 = int(x + (self.length/2))
                y2 = int(y + (self.length/2))

                
                if where == 0:
                    x11 = 0
                    y11 = 0
                    x22 = self.length
                    y22 = self.length
                elif where  == 1:
                    x11 = w - self.length
                    y11 = 0
                    x22 = w
                    y22 = self.length
                elif where  == 2:
                    x11 = 0
                    y11 = h - self.length
                    x22 = self.length
                    y22 = h
                elif where  == 3:
                    x11 = w - self.length
                    y11 = h - self.length
                    x22 = w
                    y22 = h

                mask[:, x11:x22, y11:y22] = temp[:, x1:x2, y1:y2]
                img[:, x11:x22, y11:y22] = 1.0

        mask = torch.from_numpy(mask)
        img = img * mask

        return img

class gaussian_rCutshift(object):
    def __init__(self, n_holes, length, coordinate):
        self.n_holes = n_holes
        self.length = length
        self.coordinate = coordinate


    def __call__(self, img):
        temp = img.clone()
        c = img.size(0)
        h = img.size(1)
        w = img.size(2)
        
        # gaussian 
        
        lam = np.random.beta(1, 1)
        cut_rat = np.sqrt(1. - lam)
        cut_w = np.int(w * cut_rat)
        cut_h = np.int(h * cut_rat)

        mu, sigma = int(h/2), 1.0   # mean and standard deviation
        s = np.random.normal(mu, sigma, 1)

        # uniform
        #cx = np.random.randint(W)
        #cy = np.random.randint(H)

        

        mask = np.ones((c, h, w), np.float32)

        for n in range(self.n_holes):
            
            # gaussian
            s = np.random.normal(mu, sigma, 2)
            where = np.random.randint(4)
    
            x, y = s
            x1 = int(x - (self.length/2))
            y1 = int(y - (self.length/2))
            x2 = int(x + (self.length/2))
            y2 = int(y + (self.length/2))

            coord = np.random.randint(self.length)

            x_coor = (int)((random.randrange(2) - 0.5) * 2)
            y_coor = (int)((random.randrange(2) - 0.5) * 2)
            
            y11 = np.clip(y1 + y_coor  * coord, 0, h)
            y22 = np.clip(y2 + y_coor  * coord, 0, h)
            x11 = np.clip(x1 + x_coor  * coord, 0, w)
            x22 = np.clip(x2 + x_coor  * coord, 0, w)

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

            mask[:, x11:x22, y11:y22] = temp[:, x1:x2, y1:y2]
            img[:, x11:x22, y11:y22] = 1.0

        mask = torch.from_numpy(mask)
        img = img * mask

        return img

class reverse_Cutshift(object):
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
        
        # gaussian 
        mu, sigma = int(h/2), 1.0   # mean and standard deviation
        s = np.random.normal(mu, sigma, 1)

        # beta
        lam = np.random.beta(0.5, 0.5)

        # uniform
        #cx = np.random.randint(W)
        #cy = np.random.randint(H)

        mask = np.ones((c, h, w), np.float32)

        for n in range(self.n_holes):
            
            s = np.random.normal(mu, sigma, 2)
            lam = np.random.beta(0.5, 0.5)
            x, y = s

            cut_rat = np.sqrt(1. - lam)
            cut_w = np.int(W * cut_rat)
            cut_h = np.int(H * cut_rat)

            # uniform
            cx = np.random.randint(W)
            cy = np.random.randint(H)

            bbx1 = np.clip(cx - cut_w // 2, 0, w)
            bby1 = np.clip(cy - cut_h // 2, 0, h)
            bbx2 = np.clip(cx + cut_w // 2, 0, w)
            bby2 = np.clip(cy + cut_h // 2, 0, h)

            x_length = bbx2 - bbx1
            y_length = bby2 - bby1
            
            x11 = x - x_length//2
            y11 = y - x_length//2
            x22 = x + x_length//2
            y_22 = y + y_length//2

            mask[:, y11:y22, x11:x22] = temp[:, y1:y2, x1:x2]
            img[:, y11:y22, x11:x22] = 1.0

        mask = torch.from_numpy(mask)
        img = img * mask

        return img

def save_checkpoint(state, model_path):
    torch.save(state, model_path)
    
class getedge(object):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, beta, prob):
        self.beta = beta
        self.prob = prob


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
        

        

        # beta
        lam = np.random.beta(self.beta, self.beta)

        mask = np.ones((c, h, w), np.float32)
        r = np.random.rand(1)
        if r < self.prob:
            
            # gaussian 
            mu, sigma = int(h/2), 1.0   
            s = np.random.normal(mu, sigma, 2)
            lam = np.random.beta(0.5, 0.5)
            x, y = s

            #boundingbox 
            cut_rat = np.sqrt(1. - lam)
            cut_w = np.int(w * cut_rat)
            cut_h = np.int(h * cut_rat)

            # uniform bounding box 위치생성
            cx = np.random.randint(w)
            cy = np.random.randint(h)

            x1 = np.clip(cx - cut_w // 2, 0, w)
            y1 = np.clip(cy - cut_h // 2, 0, h)
            x2 = np.clip(cx + cut_w // 2, 0, w)
            y2 = np.clip(cy + cut_h // 2, 0, h)

            x_length = x2 - x1
            y_length = y2 - y1
            
            x11 = np.clip(int(x - (x_length//2)), 0, w)
            y11 = np.clip(int(y - (y_length//2)), 0, h)
            x22 = np.clip(int(x + (x_length//2)), 0, w)
            y22 = np.clip(int(y + (y_length//2)), 0, h)
            
            x_length2 = x22 - x11
            y_length2 = y22 - y11         
                  
            if x_length2 != x_length:
                temp_ = x_length - x_length2
                if temp_ > 0:
                    x11 = x11 - temp_
                if x11 < 0:
                    x11 = 0
                    x22 = x22 + temp_
            if y_length2 != y_length:
                temp_ = y_length - y_length2
                if temp_ > 0:
                    y11 = y11 - temp_
                if y11 < 0:
                    y11 = 0
                    y22 = y22 + temp_
            
            mask[:, x11:x22, y11:y22] = temp[:, x1:x2, y1:y2]
            img[:, x11:x22, y11:y22] = 1.0

        mask = torch.from_numpy(mask)
        img = img * mask    

        return img


