
import numpy as np
from PIL import Image, ImageOps, ImageFilter, ImageDraw
import sklearn
from sklearn.metrics import mean_squared_error
import glob
import time
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector, PolygonSelector, LassoSelector
from matplotlib.path import Path
from utils import load_model, extract_features

import level_set_method


# Path to the samples folder
base_folder = 'Samples'
samples = sorted(glob.glob(base_folder + '/*'))


def model_based_phi(image, mode=1):
    """
    Initialize the phi using the bounding box predicted by the trained model.
    """
    # Extract features from the input image
    if image.mode != "RGB":
        image = image.convert("RGB")  # Ensure the image has 3 channels

    image_array = np.array(image) / 255.0 
    
    features = extract_features(image_array)

    model_path = "./xgb.pkl"
    model = load_model(model_path)

    # Predict bounding box
    bbox = model.predict([features])[0]  # bbox = [xmin, ymin, xmax, ymax]
    xmin, ymin, xmax, ymax = map(int, bbox)

    xmin = max(0, min(xmin, image_array.shape[1] - 1))
    ymin = max(0, min(ymin, image_array.shape[0] - 1))
    xmax = max(0, min(xmax, image_array.shape[1] - 1))
    ymax = max(0, min(ymax, image_array.shape[0] - 1))

    # Plot the bounding box on the image
    plt.figure(figsize=(8, 6))
    plt.imshow(image)
    plt.gca().add_patch(plt.Rectangle(
        (xmin, ymin), xmax - xmin, ymax - ymin, 
        edgecolor='red', linewidth=2, fill=False
    ))
    plt.title("Bounding Box Prediction")
    plt.axis("off")
    plt.show()

    # Initialize phi based on the predicted bounding box
    phi = -1 * np.ones(image_array.shape[:2])  # Initialize with -1
    if mode == 1:
        phi[ymin:ymax, xmin:xmax] = 1
    elif mode == 2:
        phi = np.ones(image_array.shape[:2])  # Initialize with 1
        phi[ymin:ymax, xmin:xmax] = -1
    
    return phi


def user_drawn_phi(image, mode=1):
    """
    Allows the user to draw a freeform region on the image to initialize the level set.
    """
    image_array = np.array(image)
    fig, ax = plt.subplots()
    ax.imshow(image_array, cmap="gray")
    plt.title("Draw a freeform region and release the mouse to finish")
    coords = {"vertices": None}

    # Callback to capture vertices when the LassoSelector is completed
    def onselect(vertices):
        coords["vertices"] = vertices
        plt.close(fig)

    # Create the LassoSelector
    lasso = LassoSelector(ax, onselect)

    plt.show()

    # Initialize phi based on the drawn lasso
    phi = -1 * np.ones(image_array.shape)
    if coords["vertices"] is not None:
        # Create a binary mask from the lasso path
        path = Path(coords["vertices"])
        x, y = np.meshgrid(np.arange(image_array.shape[1]), np.arange(image_array.shape[0]))
        points = np.vstack((x.flatten(), y.flatten())).T
        mask = path.contains_points(points).reshape(image_array.shape)
        if mode == 1:
            phi[mask] = 1
        elif mode == 2:
            phi = np.ones(image_array.shape)
            phi[mask] = -1
    return phi


# Default phi initialization function
def default_phi(image, mode=1, width=5):
    if mode == 1:
        phi = -1 * np.ones([image.size[1], image.size[0]])
        phi[int(image.size[1]/2) - width:int(image.size[1]/2) + width,
            int(image.size[0]/2) - width:int(image.size[0]/2) + width] = 1
    elif mode == 2:
        phi = np.ones([image.size[1], image.size[0]])
        phi[width:image.size[1] - width, width:image.size[0] - width] = -1
    return phi

# User-defined phi initialization function
def user_defined_phi(image, mode=1):
    image_array = np.array(image)
    fig, ax = plt.subplots()
    ax.imshow(image_array, cmap='gray')
    coords = {"x1": None, "y1": None, "x2": None, "y2": None}

    def onselect(eclick, erelease):
        coords["x1"], coords["y1"] = int(eclick.xdata), int(eclick.ydata)
        coords["x2"], coords["y2"] = int(erelease.xdata), int(erelease.ydata)
        plt.close(fig)

    rect_selector = RectangleSelector(
        ax, onselect, interactive=True, useblit=True, button=[1],
        minspanx=5, minspany=5, spancoords="pixels", drag_from_anywhere=True
    )
    plt.title("Draw a rectangle to initialize the level set")
    plt.show()

    phi = -1 * np.ones(image_array.shape)
    if None not in coords.values():
        if mode == 1:
            phi[coords["y1"]:coords["y2"], coords["x1"]:coords["x2"]] = 1
        elif mode == 2:
            phi = np.ones(image_array.shape)
            phi[coords["y1"]:coords["y2"], coords["x1"]:coords["x2"]] = -1
    return phi

# # Level set segmentation function
# def lss(img, dt=1, freq=20, rad=3, phi_init_func=None, mode=1):
#     img = img.convert('L')
#     img = np.array(img) - np.mean(img)
#     img = Image.fromarray(img).convert('L')
#     if phi_init_func:
#         u = phi_init_func(img, mode=mode)
#     else:
#         u = default_phi(img, mode=mode)

#     dx, dy = np.gradient(img.filter(ImageFilter.GaussianBlur(radius=rad)))
#     Du = np.sqrt(dx**2 + dy**2)
#     v = 1. / (1. + Du)
#     data = [Du, u]
#     u_old = u
#     niter = 0
#     MSE_OLD = 1e+03
#     change = 1e+03

#     start_time = time.time()
#     while change > 1e-15:
#         niter += 1
#         dx, dy = np.gradient(u)
#         Du = np.sqrt(dx**2 + dy**2)
#         u += dt * v * Du
#         u = np.where(u < 0, -1., 1.)
#         MSE = mean_squared_error(u_old, u)
#         u_old = u
#         change = abs(MSE - MSE_OLD)
#         MSE_OLD = MSE
#         if niter % freq == 0:
#             data.append(u)
#     convergence_time = time.time() - start_time
#     u = np.where(u < 0, 1., 0.)
#     data.append(u)
#     return data, niter

# Function to plot the boundary on the image
def plot_boundary(img, segment):
    rad = 3
    img = img.convert("RGB")
    edge = Image.fromarray(segment).convert("L").filter(ImageFilter.GaussianBlur(radius=rad)).filter(ImageFilter.FIND_EDGES)
    for x in range(rad, img.size[0] - rad):
        for y in range(rad, img.size[1] - rad):
            if edge.getpixel((x, y)) != 0:
                for i in [-1, 0, 1]:
                    for j in [-1, 0, 1]:
                        img.putpixel((x + i, y + j), (255, 0, 255))
    return img

# Function to create and save GIF of boundary evolution
def evolution_gif(img, data, file_name='boundary_evolution.gif', duration=400, loop=2):
    images = [plot_boundary(img, frame) for frame in data[2:]]
    images[0].save(file_name, save_all=True, append_images=images[1:], optimize=False, duration=duration, loop=loop)

if __name__ == "__main__":
    # Load the trained model
    model_path = "./xgb.pkl"
    model = load_model(model_path)

    # Load the sample image
    img = Image.open(samples[1])  # Replace with your image path if necessary
    plt.figure(figsize=(8, 6))
    plt.imshow(img)
    plt.title("Original Image", fontsize=14, fontweight="bold")
    plt.axis("off")
    plt.show()

    # Run with default_phi
    print("Running with default_phi...")
    data_default, niter_default = level_set_method.lss(
        img, dt=1, freq=10, rad=3, phi_init_func=default_phi, mode=1
    )
    default_gif = "default_phi_evolution.gif"
    evolution_gif(img, data_default, file_name=default_gif)

    # Run with user_defined_phi (rectangle)
    print("Running with user_defined_phi...")
    data_user, niter_user = level_set_method.lss(
        img, dt=1, freq=10, rad=3, phi_init_func=user_defined_phi, mode=1
    )
    user_gif = "user_defined_phi_evolution.gif"
    evolution_gif(img, data_user, file_name=user_gif)

    # Run with user_drawn_phi (freeform shape)
    print("Running with user_drawn_phi...")
    data_freeform, niter_freeform = level_set_method.lss(
        img, dt=1, freq=10, rad=3, phi_init_func=user_drawn_phi, mode=1
    )
    freeform_gif = "freeform_phi_evolution.gif"
    evolution_gif(img, data_freeform, file_name=freeform_gif)

    # Run with model-based initialization
    print("Running with model_based_phi...")
    data_model, niter_model = level_set_method.lss(
        img, dt=1, freq=10, rad=3, phi_init_func=model_based_phi, mode=1
    )
    model_gif = "model_based_phi_evolution.gif"
    evolution_gif(img, data_model, file_name=model_gif)

    # Display results
    print(f"Default Initialization: Iterations = {niter_default}, Time = {time_default:.2f} seconds")
    print(f"User Rectangle Initialization: Iterations = {niter_user}, Time = {time_user:.2f} seconds")
    print(f"User Freeform Initialization: Iterations = {niter_freeform}, Time = {time_freeform:.2f} seconds")
    print(f"Model Initialization: Iterations = {niter_model}, Time = {time_model:.2f} seconds")
    print(f"Default GIF saved as: {default_gif}")
    print(f"User Rectangle GIF saved as: {user_gif}")
    print(f"User Freeform GIF saved as: {freeform_gif}")
    print(f"Model-based GIF saved as: {model_gif}")

    # Show the converged boundary results side by side
    fig, axes = plt.subplots(1, 4, figsize=(24, 6))
    methods = ["Default Phi", "User Rectangle Phi", "User Freeform Phi", "Model Based Phi"]
    results = [data_default[-1], data_user[-1], data_freeform[-1], data_model[-1]]
    colors = ['#3A5795', '#637BAD', '#ADB9D3', '#333333']

    for ax, method, result, color in zip(axes, methods, results, colors):
        ax.imshow(plot_boundary(img, result))
        ax.set_title(method, fontsize=12, color=color)
        ax.axis("off")

    plt.suptitle("Segmentation Results Comparison", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig("segmentation_results_comparison.png", dpi=300, bbox_inches="tight")
    plt.show()

    # Plot iteration counts
    methods = ['Default Phi', 'User Rectangle Phi', 'User Freeform Phi', 'Model Based Phi']
    iterations = [niter_default, niter_user, niter_freeform, niter_model]
    colors = ['#3A5795', '#637BAD', '#ADB9D3', '#333333']

    plt.figure(figsize=(10, 6))
    bars = plt.bar(methods, iterations, color=colors, alpha=0.9)

    # Add values on top of bars
    for bar in bars:
        plt.text(
            bar.get_x() + bar.get_width() / 2,  # X-coordinate
            bar.get_height() + 2,  # Y-coordinate
            f'{int(bar.get_height())}',  # Text
            ha='center', va='bottom', fontsize=10, fontweight="bold"
        )

    # Add titles and labels
    plt.title("Iteration Count Comparison", fontsize=14, fontweight="bold")
    plt.ylabel("Number of Iterations", fontsize=12)
    plt.xlabel("Initialization Method", fontsize=12)
    plt.grid(axis="y", linestyle="--", alpha=0.6)

    # Save the plot
    plt.savefig("iteration_count_comparison.png", dpi=300, bbox_inches="tight")
    plt.show()
