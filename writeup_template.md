#**Behavioral Cloning**

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

#####

Important files:


1. Preprocessing data

- filtering
- data augmentation
    ...
- normalization
- cropping

2. Model architecture

avoiding overfitting
    dropouts
    validation

    epochs limits

generator



#####


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"
[augmented]: ./examples/augmented.png "augmented"
[class_dist]: ./examples/class_dist.png "class distribution"
[data_filt_light]: ./examples/data_filt_light.png "data filtered lightly"
[data_filt]: ./examples/.png "data filtered"
[data_ori]: ./examples/data_ori.png "Original data"
[gauss]: ./examples/near_gauss.png "nearly gaussian distribution"
[track2_aug]: ./examples/track2_aug.png "track 2 augmented"
[track2]: ./examples/track2.png "track 2 augmented"
[track1_aug]: ./examples/track1_aug.png "track 1 augmented"
[track1]: ./examples/track1.png "track 1"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network
* preprocess.py containing the data preprocessing/augmentation pipeline and data generators
* test.ipynb the notebook used for visualization and demonstration
* writeup_report.md or writeup_report.pdf summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network.
The file preprocess.py contains the image augmentation and preprocessing pipeline.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

I experimented with various model architectures including Nvidia's, the LeNet and a drastically simplified model which I'm currently employing.

Nividia's architecture is very powerful and robust as it's meant to be operated in a real world scenario which is much more complex and unpredictable than our basic simulator. The param count of this architecture is very high, going from 1.5 MIL to 14 MIL depending on input image size. Therefore, training it consumes substantial time an resources and that is why I explored LeNet and a much more simplified model inspired from LeNet as they are suffciently deep enough to bring out the simulator track patterns and operate with high accuracy with the added benefit of very low parameter count. My current simplified model uses ~24,122 params.

This simplified model is a convolution neural network based off the LeNet arch. It starts with a normalizing layer for the input input to the range of -0.5 and 0.5 after which the images are cropped to remove 70 pixels from the top and 20 from the bottom.

Then a convolutional layer with 5x5 kernel and (6) depth is used, with ReLU activation for nonlinearity. A pooling layer with 3x3 kernel follows after which a second convolutional layer with 5x5 kernel and (15) depth, with ReLU activation. After this, a pooling layer of 3x3 kernel follows.

Starting here, the simplified model is differentiated from the LeNet by the introduction of a fully-connected layer of size 100, followed by a dropout with 0.65 probability of dropout. Next, the output is flattened and followed by the output layer of size 1, to give the steering angle.

####2. Attempts to reduce overfitting in the model

The model contains a dropout layer to reduce overfitting as mentioned before.

The model was trained and validated on different data sets to ensure that the model was not overfitting. The validation split was 20%.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.
I use a batch size of 128.
I use 10 epochs to train the model without overfitting.

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road. I used all three cameras including left, center and right to train the model so it learns how to get back to the center of the road. I added/subtracted steering coefficients for the left/right cameras to adjust the central angle.

The data is highly skewed towards straight driving and I use an extensive data augmentation and filter pipeline to correct that as described in the upcoming sections.

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to use the LeNet architecture to create a simplified convolution network which can gauge the patterns of the tracks approproately.

My first step was to use a convolution neural network model similar to Nvidia. I thought this model might be appropriate because of it's established efficacy for self driving problems. But the problem with that is it has too many parameters and takes much longer to train.

My next step was to use a CNN similar to the LeNet architecture as suggested in the lessons. I thought this model might be appropriate because the edges of the track were simple lines and it worked decently on the complex traffic signs problem. Since the model only needed to output a steering angle, three convolutional layers were followed by a single fully connected layer and a dropout layer.


In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting.

I added the dropout regularization layer to combat overfitting which was added just before the flattening layer as more pathways will have more robust and redundant representations of the useful features in the network.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road. The vehicle does seem to veer back and forth on the road, which may be due to inaccuracies in choosing steering angles or the normalization of the data.

####2. Final Model Architecture

The final model architecture consisted of a convolution neural network with the layers and layer sizes as visualized.

The layer closest to the output is huge which brings out the fact that this model can be further meticnulously engineered by selecting better kernel/step sizes or adding more layers which will further optimize the model.

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

____________________________________________________________________________________________________
Layer (type)                     Output Shape          Param #     Connected to
====================================================================================================
lambda_2 (Lambda)                (None, 160, 320, 3)   0           lambda_input_2[0][0]
____________________________________________________________________________________________________
cropping2d_2 (Cropping2D)        (None, 70, 320, 3)    0           lambda_2[0][0]
____________________________________________________________________________________________________
convolution2d_3 (Convolution2D)  (None, 66, 316, 6)    456         cropping2d_2[0][0]
____________________________________________________________________________________________________
maxpooling2d_3 (MaxPooling2D)    (None, 22, 105, 6)    0           convolution2d_3[0][0]
____________________________________________________________________________________________________
convolution2d_4 (Convolution2D)  (None, 18, 101, 15)   2265        maxpooling2d_3[0][0]
____________________________________________________________________________________________________
maxpooling2d_4 (MaxPooling2D)    (None, 6, 33, 15)     0           convolution2d_4[0][0]
____________________________________________________________________________________________________
dense_3 (Dense)                  (None, 6, 33, 100)    1600        maxpooling2d_4[0][0]
____________________________________________________________________________________________________
dropout_2 (Dropout)              (None, 6, 33, 100)    0           dense_3[0][0]
____________________________________________________________________________________________________
flatten_2 (Flatten)              (None, 19800)         0           dropout_2[0][0]
____________________________________________________________________________________________________
dense_4 (Dense)                  (None, 1)             19801       flatten_2[0][0]
====================================================================================================
Total params: 24,122
Trainable params: 24,122
Non-trainable params: 0

####3. Creation of the Training Set & Training Process

After doing the initial data exploration, I found that the data is completely skewed in favor of straight driving. And the turns which we do have are mostly skewed in favor of left.

![alt text][data_ori]

I had a lot of fun building a huge preprocessing and data augmentation pipeline for this project which is in the file 'preprocess.py'.
It has a ton of parameters which can be tuned to completely change the face of the data probabilistically.

It can perform the following:

Image pre-processing:
* cropping
* resizing

Image augmentation:
* flipping (changes steer angle)
* shear (changes steer angle)
* change brightness
* 2D shift (changes steer angle)
* rotate
* change brightness

Filter data:
* filter specific steering angles with a certain probability (useful for curbing the huge amount of 0 steering angle data)
* filter steering angles by their distribution bins, with settings like the bin distance from origin and bin frequency contributing to the filter probability
    This is useful to prep the data to make sure the outliers (i.e. the higher steering angles which are rare but more important for tricky turns) aren't overshadowed


Using the above techniques, I could change the data distribution as visualized below:

This is achieved by only using filters. The filters result in loss of training data which is a con.

![alt text][data_filt] ![alt text][data_filt_light]

This nearly gaussian distribution is achieved by augmentation techniques.

![alt text][gauss]

The images before and after the pipeline are shown below:

The image visualizations are present in the test.ipynb / report.html

![alt text][track1] ![alt text][track1_aug]
![alt text][track2] ![alt text][track1_aug]

I also flip all images (and angles) and add to the dataset and use all three camera images. Flipping the images helps the model generalize to both right and left turns better.

I used the udacity provided data and added some of my own data for recovery from the sides to take the car through trickier stretches of the track.

Then I repeated this process on track two in order to get more data points.

After the collection process, I had 48630 number of data points. I then preprocessed this data by normalizing the image to a range of (-1, 1) and cropping the image by removing 70 pixels from the top and 20 from the bottom.


I finally randomly shuffled the data set and put 20% of the data into a validation set.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 10 as evidenced by the plateauing validation loss on going any further. I used an adam optimizer so that manually training the learning rate wasn't necessary.
