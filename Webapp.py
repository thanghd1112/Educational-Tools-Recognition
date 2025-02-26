import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

# Định nghĩa các đường dẫn đến các file mô hình
model_paths = {
    "CNN": "CNN.h5",
    "ResNET50 Transfer": "ResNET50 Transfer Learning.h5",
    "ResNET50" :"ResNet50_DL.h5",
    "MLP":"MLP.h5",
    "VGG16":"VGG16.h5",
    "VGG16 Transfer":"VGG16(TRANSFER).h5",
    # Thêm các đường dẫn mô hình khác ở đây
}

# Tải mô hình dựa trên lựa chọn của người dùng
def load_model(model_path):
    return tf.keras.models.load_model(model_path)

# Tên của các lớp cho mỗi mô hình
class_names = {
    "CNN": ['Eraser', 'Pen', 'Pencil_Sharpener', 'Ruler', 'Scissors'],  
    "ResNET50 Transfer": ['Eraser', 'Pen', 'Pencil_Sharpener', 'Ruler', 'Scissors'], 
    "ResNET50" :['Eraser', 'Pen', 'Pencil_Sharpener', 'Ruler', 'Scissors'],
    "MLP":['Eraser', 'Pen', 'Pencil_Sharpener', 'Ruler', 'Scissors'],
    "VGG16":['Eraser', 'Pen', 'Pencil_Sharpener', 'Ruler', 'Scissors'],
    "VGG16 Transfer":['Eraser', 'Pen', 'Pencil_Sharpener', 'Ruler', 'Scissors'],
    # Thêm các danh sách lớp tương ứng với các mô hình khác ở đây
}

# Xác định hàm dự đoán
def predict(image, model_name):
    model = load_model(model_paths[model_name])
    
    # Tiền xử lý ảnh
    image = image.resize((224, 224))  # Điều chỉnh kích thước ảnh
    img_array = np.array(image) / 255.0  # Chuẩn hóa giá trị pixel
    img_array = np.expand_dims(img_array, axis=0)  # Thêm chiều batch
    
    # Dự đoán
    predictions = model.predict(img_array)
    
    # Chuyển đổi chỉ số dự đoán thành tên lớp tương ứng
    predicted_class_index = np.argmax(predictions[0])
    predicted_class = class_names[model_name][predicted_class_index]
    
    return predicted_class

# Giao diện người dùng
def main():
    st.title('Image Classification Demo')
    
    # Chọn mô hình
    selected_model = st.selectbox("Select Model", list(model_paths.keys()))
    
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    
    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
        if st.button('Predict'):
            with st.spinner('Predicting...'):
                prediction = predict(image, selected_model)
                
            st.write(f"Prediction: {prediction}")

if __name__ == '__main__':
    main()
