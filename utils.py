import csv
import numpy as np
import torch

class CSVLogger():
    def __init__(self, args, fieldnames, filename='log.csv'):

        self.filename = filename
        self.csv_file = open(filename, 'w')

        # Write model configuration at top of csv
        writer = csv.writer(self.csv_file)
        for arg in vars(args):
            writer.writerow([arg, getattr(args, arg)])
        writer.writerow([''])

        self.writer = csv.DictWriter(self.csv_file, fieldnames=fieldnames)
        self.writer.writeheader()

        self.csv_file.flush()

    def writerow(self, row):
        self.writer.writerow(row)
        self.csv_file.flush()

    def close(self):
        self.csv_file.close()

class Cutout(object):
    """Randomly mask out one or more patches from an image.

    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img

class Cut_shift(object):
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
            
            y11 = np.clip(y1 + self.coordinate, 0, h)
            y22 = np.clip(y2 + self.coordinate , 0, h)
            x11 = np.clip(x1 + self.coordinate, 0, w)
            x22 = np.clip(x2 + self.coordinate, 0, w)
            
            y_mask = y2 - y1 
            x_mask = x2 - x1 
            y_mask2 = y22 - y11 
            x_mask2 = x22 - x11 
            
            if x_mask != x_mask2:
                x11 = x11 - (x_mask - x_mask2)
            if y_mask != y_mask2:
                y11 = y11 - (y_mask - y_mask2)
            
            mask[:, y11:y22, x11:x22] = temp[:, y1:y2, x1:x2]
            img[:, y11:y22, x11:x22] = 1.0

        mask = torch.from_numpy(mask)
        img = img * mask

        return img

