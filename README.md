# Food Recognition using Bounding Box Model
Creation of a model in DeepLearning for a detection and classification of a specific item in a tray of food using a Bounding Box

Based on the VGG16 architecture a model was created to detect an object and classifies it. 
The model was trained with in the inputs the coordinates of a bounding box specifying where the object is
and the label of the size of the box. 

For the metric, we choose the IOU metric for the regression problem and the accuracy for the classification problem. 
