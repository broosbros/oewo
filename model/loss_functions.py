import numpy as np
import cv2
import torch

class Losses:
    #L2 Loss
    def mean_squared_error(y,y_pred):
        return (np.sum(y-y_pred)**2)/np.size(y)

    #La Loss
    def mean_absolute_error(y,y_pred):
        return np.sum(np.abs(y-y_pred))/np.size(y)

    #Huber Loss
    def Huber(y,y_pred,delta):
        condition = np.abs(y-y_pred)<delta

        l = np.where(condition,0.5*(y-y_pred)**2,delta*(np.abs(y-y_pred)-0.5*delta))

        return np.sum(l)/np.size(y)
    
    #xl = LR image
    #xh = HR image
    #Attribute Loss
    def attributeLoss()


    #Sharp Loss -> For Sharpness
class USMSharp(torch.nn.Module): 
    def __init__(self, radius=50, sigma=0): 
        super(USMSharp, self).__init__() 
        if radius % 2 == 0: 
            radius += 1 
        self.radius = radius 
        kernel = cv2.getGaussianKernel(radius, sigma) 
        kernel = torch.FloatTensor(np.dot(kernel, kernel.transpose())).unsqueeze_(0) 
        self.register_buffer('kernel', kernel) 
    def forward(self, img, weight=0.5, threshold=10): 
        blur = filter2D(img, self.kernel) 
        residual = img - blur 
        mask = torch.abs(residual) * 255 > threshold 
        mask = mask.float() 
        soft_mask = filter2D(mask, self.kernel) 
        sharp = img + weight * residual 
        #sharp = torch.clamp(sharp, 0, 1) 
        #return soft_mask * sharp + (1 - soft_mask) *img
        return sharp