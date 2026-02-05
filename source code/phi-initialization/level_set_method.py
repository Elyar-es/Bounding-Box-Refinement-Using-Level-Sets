import numpy as np
from PIL import Image


def default_phi(x, mode=1, width=5):
    
    if mode == 1:
        phi = -1 * np.ones([x.shape[0], x.shape[1]])
        phi[int(x.shape[0] / 2) - width:int(x.shape[0] / 2) + width,
            int(x.shape[1] / 2) - width:int(x.shape[1] / 2) + width] = 1
        
    elif mode == 2:
        phi = 1. * np.ones([x.shape[0], x.shape[1]])
        phi[5:x.shape[0] - width, width:x.shape[1] - width] = -1.
        
    return phi

def sobel_filter(image):
    
    Kx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    
    Ix = convolve(image, Kx)
    Iy = convolve(image, Ky)
    
    G = np.sqrt(Ix**2 + Iy**2)
    
    return Ix, Iy, G

def gaussian_blur(image, kernel_size=3, sigma=1):
    
    kernel = np.zeros((kernel_size, kernel_size))
    
    for x in range(kernel_size):
        for y in range(kernel_size):
            kernel[x, y] = (1 / (2 * np.pi * sigma ** 2)) * np.exp(-((x - (kernel_size - 1) / 2) ** 2 + (y - (kernel_size - 1) / 2) ** 2) / (2 * sigma ** 2))
    
    kernel /= np.sum(kernel)
    
    return convolve(image, kernel)

def convolve(image, kernel):
    
    image_height, image_width = image.shape
    kernel_height, kernel_width = kernel.shape
    
    pad_height = kernel_height // 2
    pad_width = kernel_width // 2
    
    padded_image = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width)), mode='constant')
    
    output = np.zeros_like(image)
    
    for i in range(image_height):
        for j in range(image_width):
            region = padded_image[i:i+kernel_height, j:j+kernel_width]
            output[i, j] = np.sum(region * kernel)
            
    return output

def custom_gradient(arr, axis=None):
    
    arr = np.asarray(arr, dtype=float)
    
    if axis is None:
        return [custom_gradient(arr, axis=i) for i in range(arr.ndim)]

    # Get the shape and the slices for computation
    dim = arr.shape[axis]
    slice_prefix = (slice(None),) * axis
    slice_center = slice_prefix + (slice(1, -1),)
    slice_front = slice_prefix + (slice(0, 1),)
    slice_back = slice_prefix + (slice(-1, None),)

    # Compute gradient
    grad = np.empty_like(arr, dtype=float)
    grad[slice_center] = (arr[slice_prefix + (slice(2, None),)] - arr[slice_prefix + (slice(None, -2),)]) / 2.0
    grad[slice_front] = arr[slice_prefix + (slice(1, 2),)] - arr[slice_front]
    grad[slice_back] = arr[slice_back] - arr[slice_prefix + (slice(-2, -1),)]

    return grad




def binary_opening(image, structure):
    
        eroded = binary_erosion(image, structure=structure)
        opened = binary_dilation(eroded, structure=structure)
        
        return opened

def binary_closing(image, structure):
    
        dilated = binary_dilation(image, structure=structure)
        closed = binary_erosion(dilated, structure=structure)
        
        return closed

def binary_erosion(image, structure):
    
        height, width = image.shape
        structure_height, structure_width = structure.shape
        
        pad_height = structure_height // 2
        pad_width = structure_width // 2
        
        padded_image = np.zeros((height + 2 * pad_height, width + 2 * pad_width))
        padded_image[pad_height:pad_height + height, pad_width:pad_width + width] = image
        
        output = np.zeros_like(image)
        
        for i in range(height):
            for j in range(width):
                region = padded_image[i:i+structure_height, j:j+structure_width]
                output[i, j] = np.min(region[structure.astype(bool)])
                
        return output

def binary_dilation(image, structure):
    
        height, width = image.shape
        structure_height, structure_width = structure.shape
        
        pad_height = structure_height // 2
        pad_width = structure_width // 2
        
        padded_image = np.zeros((height + 2 * pad_height, width + 2 * pad_width))
        padded_image[pad_height:pad_height + height, pad_width:pad_width + width] = image
        
        output = np.zeros_like(image)
        
        for i in range(height):
            for j in range(width):
                region = padded_image[i:i+structure_height, j:j+structure_width]
                output[i, j] = np.max(region[structure.astype(bool)])
                
        return output




def remove_noise(binary_image, structure=None):
    
    if structure is None:
        structure = np.ones((3, 3))
        
    cleaned = binary_closing(binary_opening(binary_image, structure=structure), structure=structure)
    
    return cleaned


def mean_squared_error(y_true, y_pred):
    
    error = np.mean((y_true - y_pred) ** 2)
    
    return error


def lss(img, dt=3.5, freq=10, rad=3, b=1, mode=2, phi_init_func=None):
    
    # print("Original image mode:", img.mode)
    
    img = img.convert('L')
    
    # print("Converted to grayscale:", img.mode)
    
    img = np.array(img)
    img = img - np.mean(img)

    img = gaussian_blur(img)  # Apply Gaussian blur
    
    # print("Gaussian blur applied.")


    # new
    if b == 2:
        dx, dy, Du = sobel_filter(img)
        v = 1. / (1. + 0.5 * Du)   
    
    
    
    # old
    if b == 1:
        dx, dy = np.gradient(img)
        Du = np.sqrt(dx**2 + dy**2)
        v = 1. / (1. + Du) 
    
    
    # print("Sobel edge strength calculated.")
    if phi_init_func:
        u = phi_init_func(img, mode=mode)
    else:
        u = default_phi(img, mode=2)
    data = [Du, u]
    u_old = u
    
    niter = 0
    MSE_OLD = 1e+03
    change = 1e+03

    while change > 1e-15:
        
        niter += 1
        
        dx = custom_gradient(u, axis=1)
        dy = custom_gradient(u, axis=0)
        
        Du = np.sqrt(dx ** 2 + dy ** 2)  # internal energy
        
        u += dt * v * Du
        u = np.where(u < 0, -1., 1.)

        MSE = mean_squared_error(u_old, u)
        
        u_old = u
        
        change = abs(MSE - MSE_OLD)
        
        MSE_OLD = MSE
        
        if niter % freq == 0:
            data.append(u)

    u = np.where(u < 0, 1., 0.)

    # Remove noise from binary output
    u = remove_noise(u)
    data.append(u)
    
    return data, niter






