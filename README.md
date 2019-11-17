# Semantic_Segmentation_Urban_Driving


### Importance of Semantic Segmentation :

<p align="justify">
In Autonomous Driving scenario, perception of environment plays a critical role in determining the behaviour of the vehicle. It is important to know about the different methods which leaded up to semantic segmentation. Object detection is the technique that deals with distinguishing between objects in an image or video. While it is related to classification, it is more specific in what it identifies, applying classification to distinct objects in an image/video and using bounding boxes to tells us where each object is in an image/video. However, using just image detection gave the bounding box values in which the objects are confined. Rather segmentation could go one step further to distinguish objects in pixel level accuracy. The models created by image classification/recognition architectures gives the output of which class is the dominant object class in the image. Instead of predicting a probability distribution for an entire image, the image is divided into several blocks and each block is assigned its own probability distribution. This block-wise assigning goes to pixel-levels and each pixel is classified. For each pixel in the image, the network is trained to predict which class the pixel belongs to. This allows the network not only to identify several object classes in the image but also to determine the location of the objects.

 </p>


### Approach of the Project :

The objective of this project is to label the pixels of a road image using the Fully Convolutional Network (FCN) described in the [Fully Convolutional Networks for Semantic Segmentation](https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf) by Jonathan Long, Even Shelhamer, and Trevor Darrel. The project is the implementation of the challenge posted by Udacity to perform semantic segmentation on <a href="https://www.cityscapes-dataset.com/" name="p2_code">cityscapes dataset </a> 5,000 finely annotated images or 20,000 coarsely annotated images.



### Why FCNs :
Appending a fully connected layer enables the network to learn something using global information where the spatial arrangement of the input falls away. There are two main reasons to move from CNN to FCN (in the case of segmentation):

Spatial information: Fully connected layer generally causes loss of spatial information - because its “fully connected”: all output neurons are connected to all input neurons. This kind of architecture can’t be used for segmentation, if you are working in a huge space of possibilities.

Computational cost and representation power: There is also a distinction in terms of compute vs storage between convolutional layers and fully connected layers. For instance, in AlexNet the convolutional layers comprised of 90% of the weights (~representational capacity) but contributed only to 10% of the computation; and the remaining (10% weights => less representation power, 90% computation) was eaten up by fully connected layers. Thus usually researchers are beginning to favor having a greater number of convolutional layers, tending towards fully convolutional networks for everything.

The fully convolutional net tries to learn representations and make decisions based on local spatial input - which is rightly needed in this case. FCN achieves these information by integrating three special techniques.

1. Linear 1X1 convolution.
2. Upsampling through transposed convolution.
3. Skip Connection

#### Linear 1X1 Convolution :

A 1x1 convolution can increase or decrease the number of effective kernels by doing a weighted sum of responses in a depth column at any location on the feature map. A depth column is a set of feature responses at the same position but on different feature maps. That is, at any given position of the input feature map, a depth column is a response vector of different kernels. You could use a convolution layer with larger kernel size, say 3x3, to reduce the number of feature maps, but a 1x1 layer gets the job done with less number of parameters.

With a single filter of dimension 3x3x3 (depth of filter has to match input volume depth) we get as output 4x4x1 (assuming stride 1 and no padding). The key point to note here is the output is collapsed from depth 3 to depth 1 (granted width and height changed too but we could have kept that same as input by proper choice of padding of input).

![image](https://user-images.githubusercontent.com/37708330/69010136-9b938900-095c-11ea-8eee-cadfbb248d50.png)


Now instead of using a 3 x 3 x 3 filter, if we use a 1 x 1 x 3 (often called 1x1 since the depth is a variable and forced to match input volume depth - which is perhaps why it is so confusing), The output again has depth 1 as in previous case, except since we convolved with a 1x1 filter, the width and height of the input remains unchanged.

![image](https://user-images.githubusercontent.com/37708330/69010144-b7972a80-095c-11ea-8e95-3c5fa7f068d7.png)

However, if we increase the number of filters we can control the depth of the output. For instance, using two filters (each of depth 3) in the figure below , the output depth is 3.

![image](https://user-images.githubusercontent.com/37708330/69010152-da294380-095c-11ea-90a0-7c3a54a42bf2.png)


#### Upsampling through Transposed Convolution (Deconvolution):

<p align="justify">
Transposed Convolutions help in upsampling the previous layer to a higher resolution or dimension. Upsampling is a classic signal processing technique which is often accompanied by interpolation. The term transposed can be confusing since we typicallly think of transposing as changing places, such as switching rows and columns of a matrix. In this case when we use the term transpose, we mean transfer to a different place or context. We can use a transposed convolution to transfer patches of data onto a sparse matrix, then we can fill the sparse area of the matrix based on the transferred information
 </p>

![image](https://user-images.githubusercontent.com/37708330/69010251-350f6a80-095e-11ea-8771-c0369c397f15.png)

#### Skip Connection:
<p align="justify">

Skip Connections are used to explicitly copy features from earlier layers into later layers. This prevents neural networks from having to learn identity functions if necessary. Usually, some information is captured in the initial layers and is required for reconstruction during the up-sampling done using the fully connected network layer. If we would not have used the skip architecture that information would have been lost (or should say would have turned too abstract for it to be used further). So an information that we have in the primary layers can be fed explicitly to the later layers using the skip architecture. The Vanishing Gradient problem occurs when the signal parsing of SGD (Stochastic Gradient Descent) or other forms of GD (Gradient Descent) becomes so small or the signal becomes so approximative small that the loss is neglected. skip architecture is a common solution to overcome it.
 </p>

### FCN Architecture :

![image](https://user-images.githubusercontent.com/37708330/69012142-cb4d8b80-0972-11ea-9944-23f41529eeef.png)


Encoders are usually a deep neural network such as VGG or ResNet and decoders upsamples the label results and matches with the image size. In this project VGG-16 network is used as a network and neccessary deconvolution layer and skip connections are implemented.

![image](https://user-images.githubusercontent.com/37708330/69011873-83793500-096f-11ea-95ab-5f637c02a2fb.png)

- One convolutional layer (1X1) with kernel 1 from VGG's layer 7.
- One deconvolutional layer with kernel 4 and stride 2 from the first convolutional layer.
- One convolutional layer (1X1) with kernel 1 from VGG's layer 4.
- The two layers above are added to create the first skip layer.
- One deconvolutional layer with kernel 4 and stride 2 from the first ship layer.
- One convolutional layer (1X1) with kernel 1 from VGG's layer 3.
- The two layers above are added to create the second skip layer.
- One deconvolutional layer with kernel 16 and stride 8 from the second skip layer.





