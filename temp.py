

import os 
import numpy as np 


poses = np.load(os.path.join('/raid/liujie/code_recon/data/ultrasound/spine_phantom/left1', 'poses.npy'))

print(poses.shape)
print(poses[0])

import imageio 

img = imageio.imread('/raid/liujie/code_recon/data/ultrasound/spine_phantom/left1/images/10.png') 

print(img.shape) 