import numpy as np
import imageio
from numpy.core.shape_base import block
from skimage import morphology
import matplotlib.pyplot as plt


INIT_DIST = 14
WORD_TOTAL_DISTANCE = 59
DOT_RADIUS = 5

def To_grayscale(img):    
    imgA = np.floor(img[:,:,0] * 0.299 + img[:,:,1] * 0.587 + img[:,:,2] * 0.114)
    imgA = imgA.astype('uint8')
    return imgA


# Quantisation to b bits
def Quantisation (img, b):
    img = np.right_shift(img, 8-b)
    return img

def check_letter(img,jump):
    for checks in range(0,3):
        if (img[(checks*int(img.shape[0]/3))+(int(int(img.shape[0]/3)/2)+1),(INIT_DIST+DOT_RADIUS)+jump] == 0):
            return True
    return False

# gera_palavra(img)
#     jump = 0
#     enquanto nao for o fim da palavra faça:
#         se check_letter(img,jump) então
#             letra = identifica_letra(img,jump)
#             jump = jump + 59
#             printa letra
#         se não 
#             jump = jump + 23
#             printa espaço

def generateWord(img, letters):
    jump = 0

    while



#################################################################################
imgName = input().rstrip()
img = imageio.imread(imgName)
b = int(input())

grayImg = To_grayscale(img)
img = Quantisation(grayImg, b)
img = morphology.closing(img, morphology.disk(4)).astype(np.uint8)
#################################################################################

print("Shape img:", img.shape)

n = 1

#################################################################################
print(check_letter(img,n*59))
#################################################################################

f, axarr = plt.subplots(2, 1)
f.set_size_inches(12, 5)

axarr[0].imshow(img)
axarr[1].imshow(img[:,n*59:n*59+59])

plt.show()