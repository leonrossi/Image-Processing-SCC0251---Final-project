import numpy as np
import imageio
import math
import scipy
import scipy.ndimage
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage import morphology

####### Functions definitions ######

# Function for transforming RGB to gray scale
def To_grayscale(img):    
    imgA = np.floor(img[:,:,0] * 0.299 + img[:,:,1] * 0.587 + img[:,:,2] * 0.114)
    imgA = imgA.astype('uint8')
    return imgA


# Quantisation to b bits
def Quantisation (img, b):
    img = np.right_shift(img, 8-b)
    return img


# Calculates and concatenates the descriptors
def Create_descriptors(img, b):
    
    dc = Normalized_histogram(img, (2**b))
    dt = Haralick_descriptor(img, (2**b)-1)
    dg = Gradients_histogram(img)

    desc = np.concatenate((dc, dt, dg))

    return desc

# Calculates normalized histogram, returns vector
def Normalized_histogram(img, b):
    hist = np.zeros(b, dtype=np.int32)
    for i in range(b):
		# sum all positions in which A == i is true
        pixels_value_i = np.sum(img == i)
		# store it in the histogram array
        hist[i] += pixels_value_i
    
    N,M = img.shape

    #simply divide h(k) by the total number of pixels in the image.
    p = hist / (N*M)
    p = p.astype('float64')

    pNorm = np.linalg.norm(p)
    if pNorm != 0: p = p / pNorm
    
    return p


# Compute Energy metric
def energy(c):
    return np.sum(c**2)


# Compute Entropy metric
def entropy(c):
    return -np.sum(c*np.log(c+0.001))


# Compute Contrast metric
def contrast(c):    
    linha, coluna = c.shape

    i = np.arange(linha)
    j = np.arange(coluna)

    xv, yv = np.meshgrid(i,j, sparse=False, indexing='xy')

    sum = np.sum(((xv-yv)**2)*c)/(linha*coluna)

    return sum


# Compute Correlation metric
def correlation(c):

    mi_i = 0 
    mi_j = 0
    delta_i = 0
    delta_j = 0
    correlation = 0
    linha, coluna = c.shape

    # mi_i
    for i in range(linha):
        alt = 0
        for j in range(coluna):
            alt += c[i, j]

        mi_i += i * alt    

    # mi_j
    for j in range(linha):
        alt = 0
        for i in range(coluna):
            alt += c[i, j]

        mi_j += j * alt

    # delta_i
    for i in range(linha):
        alt = 0
        for j in range(coluna):
            alt += c[i, j]
        delta_i += ((i-mi_i)**2) * alt

    # delta_j
    for j in range(linha):
        alt = 0
        for i in range(coluna):
            alt += c[i, j]
        delta_j = ((j-mi_j)**2) * alt


    if delta_i != 0 and delta_j != 0:
        i=j=np.arange(linha)
        xv, yv = np.meshgrid(i,j, sparse=False, indexing='xy') 
        correlation = (np.sum(xv * yv * c) - mi_i * mi_j)/(delta_i * delta_j)
    else: correlation = 0

    return correlation


# Compute Homogeneity metric
def homogeneity(c):
    linha, coluna = c.shape

    i=np.arange(linha)
    j=np.arange(coluna)

    xv, yv = np.meshgrid(i,j, sparse=False, indexing='xy')

    sum = np.sum(c/(1 + abs(xv - yv)))

    return sum


# Compute co-ocurrence matrix
def coocurrence(g, intensities):
    linha, coluna = g.shape

    c = np.zeros([intensities+1, intensities+1])

    for i in range(linha-1):
        for j in range(coluna-1):
            # intensity level co-ocurrence matrix
            c[int(g[i, j]), int(g[i+1, j+1])] += 1

    # dividing c by the total sum of values
    c = c / np.sum(c)

    return c


# Calculates haralick texture descriptor, returns vector
def Haralick_descriptor(img, intensities):
    
    c = coocurrence(img, intensities)

    dt = np.zeros(5)

    dt[0] = energy(c)
    dt[1] = entropy(c)
    dt[2] = contrast(c)
    dt[3] = correlation(c)
    dt[4] = homogeneity(c)

    dt = dt / np.linalg.norm(dt)

    return dt


# Calculates histogram of oriented gradients, returns vector
def Gradients_histogram(img):

    # convert to float to avoid underflor and overflow
    img = img.astype('float64')

    # filters
    wsx = [ [-1, -2, -1],
            [ 0,  0,  0],
            [ 1,  2,  1]]

    wsy = [ [-1,  0,  1],
            [-2,  0,  2],
            [-1,  0,  1]]

    # calculating the cross-correlation between the image and each filter
    gradientX = scipy.ndimage.convolve(img, wsx)
    gradientY = scipy.ndimage.convolve(img, wsy)

    # calculating the magnitude matrix
    soma = np.sum(np.sqrt(gradientX**2 + gradientY**2))
    m = np.sqrt(gradientX**2 + gradientY**2)/soma

    # supressing warnings
    np.seterr(divide='ignore', invalid='ignore')

    # calculating the angle matrix
    fi = np.arctan(gradientY/gradientX)

    # shift the angles in pi/2 radians
    fi = fi + np.pi/2

    # converting the angles to degrees
    fi = np.degrees(fi)

    # digitise the angles into 9 bins
    bins = np.arange(20, 180, 20)
    fi_d = np.digitize(fi, bins, right=False)

    linhas, colunas = img.shape

    dg = np.zeros(9)

    # accumulating magnitudes using the bins
    for i in range(linhas):
        for j in range(colunas):
            dg[fi_d[i,j]] += m[i,j]

    dg = dg / np.linalg.norm(dg)

    return dg


# Function that compares two descriptors
def Difference(a, b):
    np.seterr(all='ignore')
    return np.sqrt(np.sum((a-b)**2))

####################################


####### input reading #######

_img_ref = input().rstrip() # read reference image's name
_quant_par = 1 # read quantisation parameter b <= 8
##############################


####### reading image #######
dot_left = imageio.imread("_left_dot.png")
dot_right = imageio.imread("_right_dot.png")
ref_img = imageio.imread(_img_ref)
#############################

###### Preprocessing and Quantisation ######
grayscale_left = To_grayscale(dot_left)
grayscale_right = To_grayscale(dot_right)
grayscale_left = Quantisation(grayscale_left, _quant_par)
grayscale_right = Quantisation(grayscale_right, _quant_par)

g_ref_img = To_grayscale(ref_img)
g_ref_img = Quantisation(g_ref_img, _quant_par)

grayscale_left = morphology.closing(grayscale_left, morphology.disk(4)).astype(np.uint8)
grayscale_right = morphology.closing(grayscale_right, morphology.disk(4)).astype(np.uint8)
g_ref_img = morphology.closing(g_ref_img, morphology.disk(4)).astype(np.uint8)
############################################


###### Creating Image Descriptors ##########
descriptor_left = Create_descriptors(grayscale_left, _quant_par)
descriptor_right = Create_descriptors(grayscale_right, _quant_par)
############################################


###### Finding the object #########
N,M = g_ref_img.shape
print("n: ", N)
print("m: ", M)

windows = np.empty((int(M/50)*6, 18, 25))
wind_descr = np.empty((int(M/50)*6, (2**_quant_par)+5+9), dtype=np.double)


a = 0
for i in range (0, M-1, 50): # creating windows and their descriptors
        temp_wind = g_ref_img[:,i:i+50]
        n,m = temp_wind.shape
        windows[a] = temp_wind[:int(n/3),:int(m/2)] # 1
        windows[a+1] = temp_wind[:int(n/3),int(m/2):m] # 01
        windows[a+2] = temp_wind[int(n/3):int(2*n/3),:int(m/2)] # 00/1
        windows[a+3] = temp_wind[int(n/3):int(2*n/3),int(m/2):m] # 00/01
        windows[a+4] = temp_wind[int(2*n/3):n,:int(m/2)] # 00/00/1
        windows[a+5] = temp_wind[int(2*n/3):n,int(m/2):m] # 00/00/01

        wind_descr[a] = Create_descriptors(windows[a], _quant_par)
        wind_descr[a+1] = Create_descriptors(windows[a+1], _quant_par)
        wind_descr[a+2] = Create_descriptors(windows[a+2], _quant_par)
        wind_descr[a+3] = Create_descriptors(windows[a+3], _quant_par)
        wind_descr[a+4] = Create_descriptors(windows[a+4], _quant_par)
        wind_descr[a+5] = Create_descriptors(windows[a+5], _quant_par)

        plt.show()
        a += 6

min_dist = sys.maxsize
close = -1

# calculating the distances and finding the closest window
a = 0
for y in range (int(M/50)):
    letter_value = np.zeros(6, dtype=np.int8)
    for i in range(6):
        if Difference(descriptor_left, wind_descr[a+i]) <= 0.0001 or Difference(descriptor_right, wind_descr[a+i]) <= 0.0001:
            letter_value[i] = 1

    #fig, ax = plt.subplots(4,2)
    #ax[0][0].imshow(windows[a])
    #ax[0][1].imshow(windows[a+1])
    #ax[1][0].imshow(windows[a+2])
    #ax[1][1].imshow(windows[a+3])
    #ax[2][0].imshow(windows[a+4])
    #ax[2][1].imshow(windows[a+5])
    #ax[3][0].imshow(grayscale_left)
    #ax[3][1].imshow(grayscale_right)
    #plt.show()

    print(letter_value)
    a += 6
#################################



###### Printing the results #####
print("Janela ", close)
#################################
