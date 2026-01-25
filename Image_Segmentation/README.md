# Image Segmentation
This project is for training a neural network using `The Oxford-IIIT Pet Dataset`

It uses the `U-net` architecture for training the model

# Usage

For training, open the `classification.ipynb` file in colab using the GPU acceleration

When the training is done, download the `unet_pet_final_dualloss.h5` file into the project directory for the next step

For inference, use the streamlit interface provided with
```
streamlit run streamlit.py
```
