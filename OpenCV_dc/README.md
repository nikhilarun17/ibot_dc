## Author
Nikhil Arun
Date: 15/1/2026

## Acknowledgements
Aryan's Day3 pdf
ChatGPT
OpenCV documentation

## About Code (train.py, evaluate.py)

### train.py
- Uses ImageNets ResNet18 model to train using the kaggle dogs vs cats database to decide whether an image is an cat or dog.
- Does data augmentation to all those data such that there are more images and that the machine learns independent do lighting, rotation and position of the dog or cat wrt to the image.
- Controls learning rate using StepLR scheduler, optimises using Adam optimiser, uses crossentropyloss as the criterion.
- Runs the training loop for 5 epochs and then saves the best model as best_resnet18.pth
- Plots the loss and the accuracy of the train and test data using matplotlib and saves the file
- Gives a heatmap of the confusion_matrix (imported from sklearn.metrics) and saves the file

### evaluate.py
- Uses the best_resnet18 and the evaluation dataset of images to test the accuracy of the model
and prints the accuracy

#Dependecies:
- torch
- torchvision
- seaborn
- sklearn metrics
- Numpy
- Matplotlib

## Output
- Trains the model and saves it as best_resnet18.pth
- Saves the plots (loss + accuracy) as training_curves.png and (confusion_matrix) as confusion_matrix.png

## Difficulties faced
- Understanding the working of CNNs and how to incorporate it using torch and the working of loss
- importing and downloading the database

## Future Improvement
- Unfreeze lower layers and change values of learning rate
- Compare with other models (MobileNetV2) and see results
- Visualise errors with pictures

Note: All future improvements together are implemented in BonusDay3.py