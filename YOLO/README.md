# Image Segmentation
This project is for training a neural network using [this dataset](https://universe.roboflow.com/people-detection-qnzez/people-detection-kpqn3)

It is used for people segmentation, especially dense crowds.

It uses the `YOLO` architecture for training the model

# Usage

For training, open the `yolo.ipynb` file in Google Colab and run it using the GPU acceleration

When the training is done, download the `best_people.pt` file into the project directory for the next step

For inference, use the streamlit interface provided with

```
streamlit run streamlit.py
```

Intefrace provided by this allows to segment images and also provides a table for the segmentation data, namely the amount of people found and the bounding box coordinates.