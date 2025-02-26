import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

# Load model
model_path = "unet_segmentation_final.h5"
model = tf.keras.models.load_model(model_path)

# Define prediction functiimport tensorflow as tf

# Example shape
variable_shape = (3, 3, 512, 1024)

# Create a tf.Variable with the correct shape
initial_value = tf.random.normal(variable_shape)
variable = tf.Variable(initial_value, shape=variable_shape)
# Define prediction function
def predict_segmentation(image, model):
    # Resize image to model's input shape
    input_image = image.resize((224, 224))
    
    # Convert image to numpy array and normalize
    input_array = np.array(input_image) / 255.0
    
    # Add batch dimension and predict segmentation
    prediction = model.predict(np.expand_dims(input_array, axis=0))[0]
    
    # Threshold prediction to get binary mask
    thresholded_prediction = (prediction > 0.7).astype(np.uint8) * 255
    
    return thresholded_prediction


# Streamlit UI
def main():
    st.title("U-Net Segmentation Demo")
    
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    
    if uploaded_image is not None:
        # Load uploaded image
        image = Image.open(uploaded_image)
        
        # Display original image
        st.image(image, caption='Original Image', use_column_width=True)
        
        if st.button('Segment'):
            with st.spinner('Segmenting...'):
                # Perform segmentation
                segmented_image = predict_segmentation(image, model)
                
            # Display segmented image
            st.image(segmented_image, caption='Segmented Image', use_column_width=True)

if __name__ == '__main__':
    main()
