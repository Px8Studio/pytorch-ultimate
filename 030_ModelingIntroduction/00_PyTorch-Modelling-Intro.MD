In this module, we will introduce you to PyTorch modeling. You will learn to build and train models from scratch, including linear regression. The module covers batch processing, datasets, and dataloaders to manage data effectively. You will also explore techniques for saving, loading, and optimizing models, including hyperparameter tuning, to enhance your machine learning workflow.
Learning Objectives
Construct and train machine learning models using PyTorch.
Implement batch processing, datasets, and dataloaders for efficient data management.
Save, load, and optimize models through hyperparameter tuning.

Foundations and Core Concepts of PyTorch
Module 7
Section Overview

0:01
Hello and welcome to our next section which is covering modeling.
Play video starting at ::6 and follow transcript0:06
And here, I'm going to provide you some short section overview so you will see what you can expect of this.
Play video starting at ::14 and follow transcript0:14
So basically, you will
Play video starting at ::16 and follow transcript0:16
learn about the model training with all the details. So you will learn about how to setup a model by creating and inheriting from the nn.Module class. Then you will see how to setup the training loop so that your model can be trained successfully.
Play video starting at ::37 and follow transcript0:37
After the model was trained successfully, you will be able to evaluate how good the model was trained.
Play video starting at ::44 and follow transcript0:44
Then also we are going to work with data sets, and you will see how beneficial it is to separate your data from your modeling. And here the data sets and also the data loaders are coming into play.
Play video starting at :1:3 and follow transcript1:03
And then we're putting another level of complexity to it. You will learn about batches, what batches are, and why you're using them.
Play video starting at :1:13 and follow transcript1:13
You will see different activation functions at action, and to get the best results you need to tune the parameters. So we will also work on that. And you will learn about hyperparameter tuning.
Play video starting at :1:29 and follow transcript1:29
And last but not least you will not want to retrain the model every time you're opening up your coding environment. So basically you want to just save the model and
Play video starting at :1:44 and follow transcript1:44
load that at some later point so that you can right away start with the model. All right. So you can see there is a bunch of topics which we will cover in this lecture and this section is basically laying the foundation for everything else. So
Play video starting at :2:4 and follow transcript2:04
the other lecturers later on are covering specific topics, very specific architectures of network. But here all the ground work is done. So I would really recommend to you not to skip this one, and pay a lot of attention in the next lectures. Thank you very much, and see you there.

Foundations and Core Concepts of PyTorch
Module 7
Linear Regression from Scratch (Coding, Model Training)

0:01
Hello and welcome to this lecture. In this lecture we're going to create our very first model, and we are creating it completely from scratch. So I hope you are excited to build your very first neural network.
Play video starting at ::16 and follow transcript0:16
So I have created here two different files in the folder ModelingIntroduction. One is called end, which is having all the code that we are going to develop. And the other one is called start, because with all the import statements of the packages and some data retrieval and stuff like that, everything that is not related to the actual creating of network. So you can use this start script, and then we will build the next steps in here. If you're more a different learning type, and if you rather want to follow along on what I'm doing, just watch the video. But I would highly recommend to you that you're actually code with me because that is a much more productive way to understand actually what is happening here. All right, let's get more space. Now we have everything setup, we have here our start script
Play video starting at :1:17 and follow transcript1:17
and we are going to start by importing the packages. So we are using numpy, pandas, torch.
Play video starting at :1:27 and follow transcript1:27
And then it's usual practice to load certain submodules of torch under different alias. So something like torch .nn is usually then imported as nn, just to save some typing.
Play video starting at :1:43 and follow transcript1:43
We're also going to use seaborn for some simple virtualization. And that is basically everything that we need for creating the first network. Then we're going to work with some data on cars. You can see the head of this dataframe. So this is a very classical dataframe of which we are going to use two different properties or two different columns which are the weight and the mpg.
Play video starting at :2:13 and follow transcript2:13
And then if we plot this, you can see here
Play video starting at :2:17 and follow transcript2:17
the observations. So we have here some decreasing trend. So not surprisingly, if the weight of a car is increasing, then the mpg is decreasing. And then on top, I also put here some regression line because this is what we're going to replicate with our regression model that we're going to create.
Play video starting at :2:43 and follow transcript2:43
So at first, we're going to convert the data to tensors.
Play video starting at :2:48 and follow transcript2:48
So we have the data already in cars. wt.values.
Play video starting at :2:56 and follow transcript2:56
That would be our first data. And then we can convert that to numpy.
Play video starting at :3:3 and follow transcript3:03
I will show you different ways how to do that actually.
Play video starting at :3:7 and follow transcript3:07
So we have the list that we have created before now converted to some array.
Play video starting at :3:14 and follow transcript3:14
Then we also need to specify some data type. So this is some float data
Play video starting at :3:19 and follow transcript3:19
and we need to reshape it.
Play video starting at :3:25 and follow transcript3:25
Let's take a look at it
Play video starting at :3:28 and follow transcript3:28
X .np.
Play video starting at :3:32 and follow transcript3:32
So now we have here
Play video starting at :3:35 and follow transcript3:35
our data with a certain shape, which is basically the way that PyTorch is expecting. So we have 32 observations and 1 column, 1 independent feature. And then we are also going to work with the target variable, which is y.
Play video starting at :3:55 and follow transcript3:55
This we are going to extract as a list,
Play video starting at :4:6 and follow transcript4:06
and now we can convert that directly to some tensors. So I will show you two different ways. One is if the data is already provided in some numpy format, then you can use the function from numpy.
Play video starting at :4:23 and follow transcript4:23
And then there is another way if the data is just provided as a list, for example, then you can create a tensor out of it with the known function towards the tensor.
Play video starting at :4:39 and follow transcript4:39
Okay. So now we have the data in the proper format. This is the x tensor, and this is our target feature.
Play video starting at :4:49 and follow transcript4:49
Okay. We're going to create now our weights, the weights are randomly initialized.
Play video starting at :5:1 and follow transcript5:01
Basically, it's just one value we have to specify the property requires_grad, because this is going to be updated. And then we're also going to create that as a plot.
Play video starting at :5:16 and follow transcript5:16
Similarly, we are going to create our bias. So usually with linear regression you have a term for uh, a term for the slope, and another one for the bias. So here we have the weight and the bias.
Play video starting at :5:35 and follow transcript5:35
Then we can specify the number of epochs.
Play video starting at :5:39 and follow transcript5:39
Let's see, the number of epochs is 1000, and we specify one specific learning rate.
Play video starting at :5:48 and follow transcript5:48
The data is fed to the neural network for training. And once all the data is provided to the network, then this is called one epoch, and then the weights are adapted, and the next iteration is starting, and then this is the next epoch. So just to know what the epochs are, because now we are saying we want to iterate 1000 times and provide the data 1000 times to the network to the training.
Play video starting at :6:20 and follow transcript6:20
So we are specifying here that we are iterating over all these epochs with this operation, and then each time we are passing one observation.
Play video starting at :6:40 and follow transcript6:40
So this is our inner loop,
Play video starting at :6:43 and follow transcript6:43
and now in this inner loop we're going to calculate our predictions, let's call them y_pred. That is our current observation of the independent feature. So x at the position of i times our weight tensor plus our bias tensor.
Play video starting at :7:6 and follow transcript7:06
So this is our forward pass.
Play video starting at :7:13 and follow transcript7:13
Then we are going to calculate our loss.
Play video starting at :7:20 and follow transcript7:20
So our last tensor, this is going to be
Play video starting at :7:25 and follow transcript7:25
torch. pow. So we're creating the square of the difference of y_pred,
Play video starting at :7:33 and follow transcript7:33
which we just calculated, minus the true value at this current position i.
Play video starting at :7:41 and follow transcript7:41
Okay.
Play video starting at :7:43 and follow transcript7:43
And now we're having our backward pass in which we are calculating the gradients. So here, all we have to do is we have to call the backward function on loss tensor.
Play video starting at :7:57 and follow transcript7:57
Now we are extracting the losses.
Play video starting at :8:1 and follow transcript8:01
So this value is basically stored in our losses object dot data at the first position.
Play video starting at :8:14 and follow transcript8:14
And now we are going to update the weights and biases.
Play video starting at :8:21 and follow transcript8:21
And for this, we're going to use the functionality torch.no_ grad.
Play video starting at :8:36 and follow transcript8:36
So in this area we are now deactivating the auto gradient because now we want to make use of the calculated gradients and assign it to the tensors.
Play video starting at :8:51 and follow transcript8:51
And then afterwards we are zeroing out the gradients so that the next iteration can start,
Play video starting at :8:59 and follow transcript8:59
you can see also one of the functions that is having some underline at the end. This is basically just saying that this is some in place operation and that the gradients are directly affected with this. So we don't have to assign anything like w.grad equals to something, though it's directly some in place operation,
Play video starting at :9:24 and follow transcript9:24
good. Then we are close to the end of this loop. Then we want to see at each end of the epoch the loss value, and we can print it directly.
Play video starting at :9:39 and follow transcript9:39
Okay. Let's run that
Play video starting at :9:43 and follow transcript9:43
and see if it's working. And it's super fast because it's a very small dataset. You can see here the losses are decreasing. That is actually what we want to see.
Play video starting at :9:53 and follow transcript9:53
And we are done.

