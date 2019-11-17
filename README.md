# Semantic_Segmentation_Urban_Driving


### Importance of Semantic Segmentation :

<p align="justify">
In Autonomous Driving scenario, perception of environment plays a critical role in determining the behaviour of the vehicle. It is important to know about the different methods which leaded up to semantic segmentation. Object detection is the technique that deals with distinguishing between objects in an image or video. While it is related to classification, it is more specific in what it identifies, applying classification to distinct objects in an image/video and using bounding boxes to tells us where each object is in an image/video. However, using just image detection gave the bounding box values in which the objects are confined. Rather segmentation could go one step further to distinguish objects in pixel level accuracy. The models created by image classification/recognition architectures gives the output of which class is the dominant object class in the image. Instead of predicting a probability distribution for an entire image, the image is divided into several blocks and each block is assigned its own probability distribution. This block-wise assigning goes to pixel-levels and each pixel is classified. For each pixel in the image, the network is trained to predict which class the pixel belongs to. This allows the network not only to identify several object classes in the image but also to determine the location of the objects.

 </p>


### Approach of the Project :

The objective of this project is to label the pixels of a road image using the Fully Convolutional Network (FCN) described in the [Fully Convolutional Networks for Semantic Segmentation](https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf) by Jonathan Long, Even Shelhamer, and Trevor Darrel. The project is the implementation of the challenge posted by Udacity to perform semantic segmentation on <a href="https://www.cityscapes-dataset.com/" name="p2_code">cityscapes dataset.</a>



### Why FCNs :
Appending a fully connected layer enables the network to learn something using global information where the spatial arrangement of the input falls away. There are two main reasons to move from CNN to FCN (in the case of segmentation):

Spatial information: Fully connected layer generally causes loss of spatial information - because its “fully connected”: all output neurons are connected to all input neurons. This kind of architecture can’t be used for segmentation, if you are working in a huge space of possibilities.

Computational cost and representation power: There is also a distinction in terms of compute vs storage between convolutional layers and fully connected layers. For instance, in AlexNet the convolutional layers comprised of 90% of the weights (~representational capacity) but contributed only to 10% of the computation; and the remaining (10% weights => less representation power, 90% computation) was eaten up by fully connected layers. Thus usually researchers are beginning to favor having a greater number of convolutional layers, tending towards fully convolutional networks for everything.

The fully convolutional net tries to learn representations and make decisions based on local spatial input - which is rightly needed in this case.


