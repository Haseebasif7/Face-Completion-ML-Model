# Multi-Ouput-Regressor


This project involves predicting the lower part of a face image based on the upper part using **Random Forest** and **Extra Trees** models, both enhanced with **MultiOutputRegressor**. The goal is to predict the missing pixels of a face image (2048 pixels) given the upper part of the face. The model utilizes **Principal Component Analysis (PCA)** to reduce the dimensionality of both the input and output data, ensuring efficient computation.

## Key Features

- **Multi-Model Prediction:** Implements multiple regression models, including Random Forest and Extra Trees, to predict the lower part of the face.
- **Dimensionality Reduction:** PCA is applied to the training and testing data to reduce the number of features, optimizing the model’s performance and computational efficiency.
- **Hyperparameter Tuning:** The model can be further enhanced by adjusting the hyperparameters, such as `n_estimators` ,`min_samples_split`,`max_depth`, for better predictions (though hyperparameter optimization was simplified for this project). Didnt done as due to computation issues
- **MultiOutputRegressor:** Handles the multi-output regression task by predicting an array of pixel values at once for each image.


### Results

The models output completed faces where the predicted lower part is merged with the upper part, resulting in a full face image. The evaluation results include MAE for each model, allowing comparison of performance.

Note: Hyperparameters should be tuned accordingly using optimization techniques for better performance (optimization was simplified for demonstration purposes).

![image](https://github.com/user-attachments/assets/f7bb00aa-fc21-4727-afec-888da94aee6a)

