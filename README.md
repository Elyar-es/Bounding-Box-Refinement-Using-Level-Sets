# Refining the Predicted Bounding Boxes of Object Detection Models Using Level Sets

## Project Overview

This project focuses on refining the predicted bounding boxes of object detection models using **Level Sets**. The goal is to improve segmentation accuracy through various initialization strategies for the **Level Set function (`phi`)**.

---

## Phi Initialization

The **Level Set function (`phi`)** is a key component of the segmentation process. The project provides four methods for initializing `phi`:

#### 1. Default Phi Initialization

- Initializes `phi` with a centered rectangular region.  
- **Function:** `default_phi(image, mode=1, width=5)`  
   - `mode=1`: Rectangle inside is positive (+1), outside is negative (-1).  
   - `mode=2`: Rectangle inside is negative (-1), outside is positive (+1).  
- **Use Case:** Best for testing and quick initialization without user input.

---

#### 2. User-Defined Phi Initialization

- Allows the user to draw a rectangular region interactively.  
- **Function:** `user_defined_phi(image, mode=1)`  
   - The user selects a rectangle to initialize `phi`.  
- **Use Case:** Suitable for defining specific areas of interest manually.

---

#### 3. User-Drawn Freeform Phi Initialization

- Lets the user draw a freeform region for initialization using a Lasso tool.  
- **Function:** `user_drawn_phi(image, mode=1)`  
   - Freeform regions drawn by the user are converted into a mask to define `phi`.  
- **Use Case:** Ideal for non-rectangular regions where precise initialization is needed.

---

#### 4. Model-Based Phi Initialization

- Leverages a pretrained model (`xgb.pkl`) to predict a bounding box for the region of interest.  
- Pretrained model is trained using the notebook `bounding_box_model.ipynb` on **Google Colab Pro**.  
- **Function:** `model_based_phi(image, mode=1)`  
   - The predicted bounding box is used to initialize `phi`.  
   - Bounding box is visualized on the image.  
- **Use Case:** Automated initialization for consistent and efficient segmentation.

---

## Experimental Results

### 1. Dataset and Experimental Setup

- **Dataset:** Pascal VOC 2012 (subset of 50 images) and synthetic images.  
- **Image Preprocessing:** Images converted to grayscale and then gaussian blur is used.  
- **Evaluation Metrics:**  
   - **Intersection over Union (IoU)**  
   - **Precision**  
   - **Recall**  
   - **F1 Score**
   - **Accuracy**

---

### 2. Segmentation Methods Evaluated

- **Level Set**  
- **Sobel Level Set**  

---

### 3. Results Overview

- **Level Set:** Achieved the **highest IoU and F1 Score** on Pascal VOC.  
- **Sobel Level Set:** Performed exceptionally well in synthetic noise conditions but underperformed on real-world data.  

---

## Key Findings

1. The **Level Set** method demonstrates consistent performance across datasets due to its balanced contour-based energy minimization strategy.  
2. The **Sobel Level Set** method excels in synthetic noise conditions but struggles with irregular textures in real-world images.  


