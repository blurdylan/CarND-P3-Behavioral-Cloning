#**Behavioral Cloning**

##Write-up Template

###A write-up describing my work on the behavioral cloning project.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


## Rubric Points

###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results obtained

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.json
```

####3. Submission code is usable and readable

My model.py file contains the code for training and saving the convolution neural network. The file shows the all pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

1. **Layer 1**: Convolutional layer with 32 5x5 filters, followed by ELU activation
2. **Layer 2**: Convolutional layer with 16 3x3 filters, ELU activation, Dropout(0.4) and 2x2 max pool
3. **Layer 3**: Convolutional layer with 16 3x3 filters, ELU activation, Dropout(0.4)
4. **Layer 4**: Fully connected layer with 1024 neurons, Dropout(0.3) and ELU activation
5. **Layer 5**: Fully connected layer with 1024 neurons, Dropout(0.3) and ELU activation

####2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting. (line 137 & 143)

####3. Model parameter tuning

The learning rate wasn't tuned manually, the model used an adam optimizer. (line 167)

####4. Appropriate training data

I had little influence on the data, I massively used the data provided by udacity

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to get a model that would adapt easily with the changing steering angles and be able to train fully with less epochs (though I am using a high batch size)

Details on the model architecture can be viewed in the rubric point below

No test set was used in training this model because, the veracity of the model would be proven when the car will be driving autonomously, so I intentionally omitted the test set.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I forced the steering angle to a particular value when approaching the side (either left or right).

Adjusting the brightness also helps to focus on the road. During this process one thing I also discovered is that the other channels (Blue & Green) apart from RED are a little useless when looking at the road and their limits, they could also be used to get a better focus

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture consisted of a convolution neural network with the following layers and layer sizes

```
____________________________________________________________________________________________________
Layer (type)                     Output Shape          Param #     Connected to                     
====================================================================================================
convolution2d_7 (Convolution2D)  (None, 32, 32, 32)    2432        convolution2d_input_3[0][0]      
____________________________________________________________________________________________________
elu_11 (ELU)                     (None, 32, 32, 32)    0           convolution2d_7[0][0]            
____________________________________________________________________________________________________
convolution2d_8 (Convolution2D)  (None, 30, 30, 16)    4624        elu_11[0][0]                     
____________________________________________________________________________________________________
elu_12 (ELU)                     (None, 30, 30, 16)    0           convolution2d_8[0][0]            
____________________________________________________________________________________________________
dropout_7 (Dropout)              (None, 30, 30, 16)    0           elu_12[0][0]                     
____________________________________________________________________________________________________
maxpooling2d_3 (MaxPooling2D)    (None, 15, 15, 16)    0           dropout_7[0][0]                  
____________________________________________________________________________________________________
convolution2d_9 (Convolution2D)  (None, 13, 13, 16)    2320        maxpooling2d_3[0][0]             
____________________________________________________________________________________________________
elu_13 (ELU)                     (None, 13, 13, 16)    0           convolution2d_9[0][0]            
____________________________________________________________________________________________________
dropout_8 (Dropout)              (None, 13, 13, 16)    0           elu_13[0][0]                     
____________________________________________________________________________________________________
flatten_3 (Flatten)              (None, 2704)          0           dropout_8[0][0]                  
____________________________________________________________________________________________________
dense_7 (Dense)                  (None, 1024)          2769920     flatten_3[0][0]                  
____________________________________________________________________________________________________
dropout_9 (Dropout)              (None, 1024)          0           dense_7[0][0]                    
____________________________________________________________________________________________________
elu_14 (ELU)                     (None, 1024)          0           dropout_9[0][0]                  
____________________________________________________________________________________________________
dense_8 (Dense)                  (None, 512)           524800      elu_14[0][0]                     
____________________________________________________________________________________________________
elu_15 (ELU)                     (None, 512)           0           dense_8[0][0]                    
____________________________________________________________________________________________________
dense_9 (Dense)                  (None, 1)             513         elu_15[0][0]                     
====================================================================================================
Total params: 3,304,609
Trainable params: 3,304,609
Non-trainable params: 0
```

####3. Creation of the Training Set & Training Process

I used only the dataset provided by udacity (The dataset contains JPG images of dimensions 160x320x3), this could be made much better with my own dataset however, time constraints me.

To augment the data set, I also flipped images and angles thinking that this would help workout for the bias that exists on the track. This could also be solved by driving reversely on the track, this may add more right turns to the data set

After the collection process, I had about number of data points. I then preprocessed this data by cropping and resizing (iteratively) them to a reasonable size which will help improve training


I finally randomly shuffled the data set and put 3000 of the 20.000 images generated using the generator data into a validation set. No test set was used.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5 but I used 3 because it also worked correctly, each epoch took about 300seconds using a GPU. I used an adam optimizer to ensure that  the manually training the learning rate won't be necessary.

## Video
Low resolution, for low size. recorded with Apowersoft screen recorder.
[The Video](./video.mp4)

