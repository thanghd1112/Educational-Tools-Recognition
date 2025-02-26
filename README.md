# Educational-Tools-Recognition

## Project Overview

This project focuses on the recognition and classification of educational tools using image processing techniques. We experimented with various machine learning models such as MLP, CNN, VGG16, and ResNet50, including transfer learning approaches. Additionally, an image segmentation model was built using the UNet architecture, and all models were deployed using Streamlit.

## Main Responsibilities

- Collecting and labeling image data.
- Experimenting with image classification models (MLP, CNN, VGG16, ResNet50) and transfer learning (VGG16, ResNet50).
- Building an image segmentation model using the UNet architecture.
- Deploying models with Streamlit.

## Recognition and Gains

- Acquired skills in image data collection, processing, and labeling.
- Gained a deeper understanding of MLP, CNN, VGG16, ResNet50, and UNet models.
- Applied techniques to improve model performance.
- Developed problem-solving and teamwork skills throughout the project.

## Dataset Overview

### **1. `Data` Folder**
The `Data` folder contains the primary dataset used for training, validation, and testing of the model. The images are organized into three subsets:

- **Train Set** (`Train`): 3,790 images across 5 classes.
- **Validation Set** (`Valid`): 616 images across 5 classes.
- **Test Set** (`Test`): 537 images across 5 classes.

Each of these subdirectories contains images categorized into five different classes:
✅ **Eraser**  
✅ **Pen**  
✅ **Ruler**  
✅ **Pencil Sharpener**  
✅ **Scissors**

This dataset is structured following a typical deep learning image classification format, where each class has its own subfolder.

### **2. `Data-2` Folder**
The `Data-2` folder appears to be an additional dataset variant, likely used for training the UNet segmentation model. Inside, there are three subdirectories:

- **train** (Training images)
- **valid** (Validation images)
- **test** (Testing images)

These sets are used specifically for segmentation tasks, complementing the classification dataset.

### **Purpose of Each Dataset**
- The `Data` folder is structured for **image classification**, where the goal is to categorize images into one of the five classes.
- The `Data-2` folder is likely used for **image segmentation**, where models like UNet are trained to detect and outline objects in images.

## Installation Guide and Important Notes

### **Installation & Setup**
1. Due to the large number of images, the trained model files were consolidated into a single script. Please ensure that each model is executed separately.
    - **1.1.** Transfer learning models may have naming conflicts with custom-built models. Reset the kernel before switching between models.
    - **1.2.** The consolidated script runs on Jupyter Notebook via Anaconda. Adjust accordingly if using a different environment.

2. **Running Streamlit for Web Deployment:**
    - **2.1.** Ensure the web application scripts are saved as `.py` files (e.g., `Segmentation.py` for segmentation and `Webapp.py` for classification models).
    - **2.2.** The execution environment must have all necessary dependencies installed.
    - **2.3.** To run Streamlit:
      ```sh
      cd "path_to_script_directory"
      streamlit run Webapp.py
      ```
    - **Note:** If using Anaconda, activate the appropriate environment before running Streamlit:
      ```sh
      conda activate my_env
      ```

3. **For `U-net Segmentation.ipynb`**, the training dataset is located in the `Data-2` folder.

## Demo Videos

### 1. UNet Segmentation Demo on Google Colab
[📹 Demo UNet Google Colab](https://drive.google.com/file/d/1Bg49sXd_Znj1TsngEQ7MbGM5toUfuEIl/view?usp=drive_link)

### 2. Report & Build on Streamlit
[📹 Report & Build on Streamlit](https://drive.google.com/file/d/1FDNyShu000GNPy8QmQ42AKjS2_Gahi8w/view?usp=drive_link)

