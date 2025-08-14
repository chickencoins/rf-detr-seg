# rf-detr-seg

This project extends the original **RF-DETR** model to support instance segmentation tasks.

The core idea is to generate segmentation masks by leveraging the multi-scale feature maps from the RF-DETR backbone. We have implemented a lightweight, FPN-like neck that takes these features as input to predict a high-quality mask for each detected instance.

This approach was tested by training and running predictions on a custom dataset. The results below confirm that the implementation is functional. Please note that no formal accuracy metrics have been measured at this stage; this was purely a test to verify that the model works as intended.

![predicted_segmentation_data_01055](https://github.com/user-attachments/assets/5cb15d06-3e85-479a-acf5-bf6ad45b49f8)


### Code Release
The source code may be made public in the future, once it is further refined and circumstances permit.


