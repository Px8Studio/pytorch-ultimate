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


Foundations and Core Concepts of PyTorch
Module 7
Linear Regression from Scratch (Coding, Model Evaluation)

0:00
So now we are here to check the results.
Play video starting at ::3 and follow transcript0:03
So let's just print with the F string notation our weight,
Play video starting at ::13 and follow transcript0:13
and that is accessible with our tensor w, and
Play video starting at ::20 and follow transcript0:20
the function item.
Play video starting at ::25 and follow transcript0:25
And we also want to show the bias,
Play video starting at ::29 and follow transcript0:29
and that is stored in b. item.
Play video starting at ::35 and follow transcript0:35
So these are the parameters that our model has calculated. If we plot again our graph,
Play video starting at ::44 and follow transcript0:44
we can see here that we have our slope, the slope is -5. So you can roughly cross check if that is making sense. It's easier with the bias. So the bias is the intersection with the Y axis and the Y axis would be hit at 36.8. You have to consider that this is not showing all the way, it's starting at 1.5. So the 0 would be here. It's, it's making sense. Okay. Now we can also calculate our predictions.
Play video starting at :1:23 and follow transcript1:23
So for this, we're just using our
Play video starting at :1:26 and follow transcript1:26
independent feature as X, multiply that with our weights, and to that we are adding the bias, and then we are detaching it
Play video starting at :1:38 and follow transcript1:38
from PyTorch and converting it to a numpy array.
Play video starting at :1:43 and follow transcript1:43
So let's take a look at y_pred.
Play video starting at :1:47 and follow transcript1:47
So these are our predictions.
Play video starting at :1:52 and follow transcript1:52
Now it's always good to have some visualization for it. So we're going to create some scatter plot
Play video starting at :2:5 and follow transcript2:05
here. We can use
Play video starting at :2:8 and follow transcript2:08
our values which are stored in X_ list, and in y_list.
Play video starting at :2:18 and follow transcript2:18
So that is our true observations. And then if we put on some line float
Play video starting at :2:29 and follow transcript2:29
for the same x value X_list, and then we're taking our predictions that we have created, and then we're putting a line through it. And actually, it's good that you see that this is actually not working. We are ending up with some error. The data must be one dimensional. Let's see what is happening here. So we have y_pred, we can check the shape and yes, it's two dimensional. So this is something that you will end up seeing a lot. We have to always make sure that the dimensions of your objects are right. So one way to solve this is to call the reshape function, -1 to remove one dimension. And then you can call it, now it's working. So this is now our regression line which is actually coming from our deep learning model. So that's actually quite cool. If you compare that to the statistical
Play video starting at :3:31 and follow transcript3:31
model that we have here, it's basically the same.
Play video starting at :3:35 and follow transcript3:35
So
Play video starting at :3:37 and follow transcript3:37
even if it doesn't look like much, you have already created your very first neural network and you were able to train it correctly and it's providing the right values. That is actually awesome.
Play video starting at :3:54 and follow transcript3:54
You can just believe me, but in statistics it's always better to trust some numbers. So we can just make sure
Play video starting at :4:3 and follow transcript4:03
to see that we get the same results by creating some real linear regression model.
Play video starting at :4:11 and follow transcript4:11
So in here, this is just some sklearn stuff that we're doing here.
Play video starting at :4:18 and follow transcript4:18
So we are a fitting
Play video starting at :4:21 and follow transcript4:21
our
Play video starting at :4:23 and follow transcript4:23
numpy object X_np, and we're passing our target variable. And with this, we create our regression model
Play video starting at :4:33 and follow transcript4:33
and then we can extract our slope and offset.
Play video starting at :4:39 and follow transcript4:39
The slope is stored in the regression coefficient, and the intercept is stored in the property intercept_. So let's see that. So the true values are, for the slope -5.3 and the intercept is 37.28. And actually what we have here is coming very close to this,
Play video starting at :5:6 and follow transcript5:06
as you can see here. So the weights and the bias are pretty much what the statistical linear regression would also provide,
Play video starting at :5:16 and follow transcript5:16
good.
Play video starting at :5:18 and follow transcript5:18
Then the last part, that is just bonus. So we don't have to do that because it's requiring some additional package installation. I have shown you this here
Play video starting at :5:32 and follow transcript5:32
at the end. So we can create a graph visualization of the graph that was used for this. For this you can use the package called GraphViz, which can be downloaded under this,
Play video starting at :5:46 and follow transcript5:46
but there are also some additional things to be considered. So if it doesn't work for you, it's definitely not a problem. I also had to do some changes just to make it work. But in the end, I just want to quickly show you that once everything is installed and we have added it to the path variable, then you can call the function make_dot, from this package torchviz and you get this representation of our graph. So this is the first neural network that you have created.
Play video starting at :6:22 and follow transcript6:22
We can see here which of the functions are called, which of them are depending on other functions and so on. And all the way back to the very beginning where we have the slope and the bias.
Play video starting at :6:36 and follow transcript6:36
So this was our first neural network that we have created. And from now on we are building up on what we have learned here. So we are putting more complexity to it. We are going to improve the performance and the way our code looks. But the general structure is going to be pretty much like what we have seen here in this first lecture.
Play video starting at :7:2 and follow transcript7:02
So thank you very much for staying with me in this coding lecture, and see you in the next one.

#%% packages
import numpy as np
import pandas as pd
import torch
import torch.nn as nn 
import seaborn as sns

#%% data import
cars_file = 'https://gist.githubusercontent.com/noamross/e5d3e859aa0c794be10b/raw/b999fb4425b54c63cab088c0ce2c0d6ce961a563/cars.csv'
cars = pd.read_csv(cars_file)
cars.head()

#%% visualise the model
sns.scatterplot(x='wt', y='mpg', data=cars)
sns.regplot(x='wt', y='mpg', data=cars)

#%% convert data to tensor
X_list = cars.wt.values
X_np = np.array(X_list, dtype=np.float32).reshape(-1,1)
y_list = cars.mpg.values.tolist()
y_np = np.array(y_list, dtype=np.float32).reshape(-1,1)
X = torch.from_numpy(X_np)
y = torch.tensor(y_list)


#%% training
w = torch.rand(1, requires_grad=True, dtype=torch.float64)
b = torch.rand(1, requires_grad=True, dtype=torch.float64)

num_epochs = 100
learning_rate = 1e-3
for epoch in range(num_epochs):
  for i in range(len(X)):
    # x, y = torch.tensor(X_list[i]), torch.tensor(y_list[i])
    # forward pass
    y_predict = X[i] * w + b
    # calculate loss
    loss_tensor = torch.pow(y_predict - y[i], 2)
    # backward pass
    loss_tensor.backward()
    # extract losses
    loss_value = loss_tensor.data[0]
    # update weights and biases
    with torch.no_grad():
      w -= w.grad * learning_rate
      b -= b.grad * learning_rate
      w.grad.zero_()
      b.grad.zero_()
  print(loss_value)

#%% check results
print(f"Weight: {w.item()}, Bias: {b.item()}")
# %%
y_pred = (torch.tensor(X_list)*w+b).detach().numpy()
# %%
sns.scatterplot(x=X_list, y=y_list)
sns.lineplot(x=X_list, y=y_pred, color='red')
# %% (Statistical) Linear Regression
from sklearn.linear_model import LinearRegression
reg = LinearRegression().fit(X_np, y_list)
print(f"Slope: {reg.coef_}, Bias: {reg.intercept_}")


# %% create graph visualisation
# make sure GraphViz is installed (https://graphviz.org/download/)
# if not computer restarted, append directly to PATH variable
# import os
# from torchviz import make_dot
# os.environ['PATH'] += os.pathsep + 'C:/Program Files (x86)/Graphviz/bin'
# make_dot(loss_tensor)
# %%



Foundations and Core Concepts of PyTorch
Module 7
Model Class (Coding)

0:01
Hello and welcome to this coding lecture.
Play video starting at ::4 and follow transcript0:04
In this coding lecture we are going to work with the Model class, because so far we have created our model directly from scratch, but now I'm going to show you how you can set up your model in a separate class which you then can easily instantiate and use in the further code.
Play video starting at ::24 and follow transcript0:24
All right. For this I have opened the linear regression model plus start script which is just providing some code for loading the required packages, and I run this first chunk,
Play video starting at ::40 and follow transcript0:40
then it's also having the
Play video starting at ::42 and follow transcript0:42
data import ready to start.
Play video starting at ::47 and follow transcript0:47
And you're going to see this is the same dataset that we have used in the lectures before.
Play video starting at ::54 and follow transcript0:54
And then we're converting these lists and numpy arrays to tensors.
Play video starting at :1:3 and follow transcript1:03
Okay. Everything ready. Now we can create our Model class. So for this we have to type the class, and we have to give a name to this class. So let's say we want to call it linearRegressionTorch
Play video starting at :1:18 and follow transcript1:18
class.
Play video starting at :1:20 and follow transcript1:20
And then this one has to inherit from the PyTorch submodule nn. Module.
Play video starting at :1:30 and follow transcript1:30
So this is the general structure you have to create this class. The class is inheriting from nn. Module. And now we have to specify the init function
Play video starting at :1:43 and follow transcript1:43
which is called when an object is instantiated from this class. So we can create this init function. It needs to have the keyword self which is referring to the objects inside the class. And then we can specify two parameters. I will call them input_size and output_size.
Play video starting at :2:7 and follow transcript2:07
And then in here, when the class is initialized, we're creating an internal object which is called linear. And this one is a linear layer coming from the nn.Module.
Play video starting at :2:23 and follow transcript2:23
And this linear class is requiring two parameters. One is for the in_features and one is for the out_features.
Play video starting at :2:33 and follow transcript2:33
So in here we're passing the size of the model. So we're telling that there's one node going in and one node going out.
Play video starting at :2:48 and follow transcript2:48
And then every model size needs to have a forward function which is provided with one variable typically that's called X.
Play video starting at :2:56 and follow transcript2:56
And
Play video starting at :2:58 and follow transcript2:58
since our model is so simple, we can just follow this proposal here. We are just calling the linear layer that we have created before and storing the results in an object called out. And we are returning this one. Later on when we have more complex models, we will set up more layers in the init function, and then we stack them together in the forward function,
Play video starting at :3:28 and follow transcript3:28
right? And that's already it. So this is our Model class. Now we can specify some parameters. Our input dimension is 1, our output dimension is 1, and we are ready to instantiate an object of this class.
Play video starting at :3:48 and follow transcript3:48
So here we just create a new object called model, which is instantiated from this class with the parameters we have just specified
Play video starting at :3:58 and follow transcript3:58
and we can run that.
Play video starting at :4: and follow transcript4:00
And our Model class is very perfect.
Play video starting at :4:3 and follow transcript4:03
Now, the next one is to specify a loss function.
Play video starting at :4:10 and follow transcript4:10
And since we're working with linear regression,
Play video starting at :4:14 and follow transcript4:14
the loss type of mean square error seems to be quite appropriate. So we will use this specific function
Play video starting at :4:23 and follow transcript4:23
and assign it to the object called loss_fun.
Play video starting at :4:27 and follow transcript4:27
Then we need to specify the optimizer. And here we can also specify another parameter called LR for learning rate.
Play video starting at :4:36 and follow transcript4:36
And that specify the value for it.
Play video starting at :4:43 and follow transcript4:43
And then we need to specify the optimizer. So the optimizer is created from the submodule optim. And then we're going to use Stochastic Gradient Descent.
Play video starting at :4:57 and follow transcript4:57
And when this function is called, it is requiring some parameters. So here we are passing the model parameters which we can extract with model.parameters. And we also can specify the learning rate. And with this, we are ready to create our training loop.
Play video starting at :5:16 and follow transcript5:16
So this time we are also going to save the losses and the slope and the bias in individual variables.
Play video starting at :5:27 and follow transcript5:27
And these ones are lists which are completely empty at the beginning.
Play video starting at :5:32 and follow transcript5:32
And we also set one parameter, the number of epochs to 1000.
Play video starting at :5:40 and follow transcript5:40
And now we can create a loop for it. So we are iterating over 1000 epochs,
Play video starting at :5:50 and follow transcript5:50
and here we are now starting. So
Play video starting at :5:55 and follow transcript5:55
we are at first setting the gradients to zero.
Play video starting at :6: and follow transcript6:00
And we're doing this by calling optimizer.zero_grad
Play video starting at :6:6 and follow transcript6:06
Then we have the forward pass. So we're passing our observations through the model that we have created to get the predictions.
Play video starting at :6:16 and follow transcript6:16
And this is exactly the line we want to run for this.
Play video starting at :6:21 and follow transcript6:21
So we're passing x,
Play video starting at :6:24 and follow transcript6:24
our data that we have set up here to the model, and we get as a return y_pred.
Play video starting at :6:34 and follow transcript6:34
Now we have predictions, and we also have the true values. So we are at the point that we can calculate the loss. So for this, we can call our loss function pass in the predictions, and the true values, and we will get back the losses which we store in the object called loss.
Play video starting at :6:57 and follow transcript6:57
And now we can update the weights
Play video starting at :7:2 and follow transcript7:02
by calling
Play video starting at :7:6 and follow transcript7:06
optimizer.step. So that would be it if we wouldn't want to learn more of the model. But right here, we want to see a bit more into the insight on what is happening. So we will get the parameters, and I will show you how we will get them, and then we are going to store them to the losses and to the slope and bias parameters. So we can iterate through
Play video starting at :7:36 and follow transcript7:36
the parameters. So in model.named_parameters,
Play video starting at :7:46 and follow transcript7:46
have all the parameters stored, and we'll get back the name and the parameter, and we're only going to check the parameters that are having a gradient required.
Play video starting at :8:2 and follow transcript8:02
And then only in the case that
Play video starting at :8:6 and follow transcript8:06
the name is
Play video starting at :8:10 and follow transcript8:10
linear dot weight
Play video starting at :8:14 and follow transcript8:14
So here we're now going to append our slope,
Play video starting at :8:21 and follow transcript8:21
and we're going to use param dot data dot num,
Play video starting at :8:30 and follow transcript8:30
and we are going to use
Play video starting at :8:35 and follow transcript8:35
the very first object.
Play video starting at :8:38 and follow transcript8:38
Okay. And we are doing something very similar for the bias term.
Play video starting at :8:44 and follow transcript8:44
So in here, we're going to append the current bias value of the current epoch.
Play video starting at :8:53 and follow transcript8:53
So then we're going to store the losses
Play video starting at :9:2 and follow transcript9:02
very similar we're taking or losses objects. And we are appending
Play video starting at :9:7 and follow transcript9:07
what is stored in loss data as a float.
Play video starting at :9:16 and follow transcript9:16
And now we can print the loss
Play video starting at :9:19 and follow transcript9:19
for every 1/100 epoch. We're going to print the epoch and the losses.
Play video starting at :9:29 and follow transcript9:29
All right, let's run that.
Play video starting at :9:32 and follow transcript9:32
And you can see here it's not working, the losses are not decreasing. They are constant. Actually, there is a bug in my code and I found out that I have
Play video starting at :9:44 and follow transcript9:44
missed one small line, one very important one, the one in which the gradients are calculated. So we spoke before about the automatic gradient calculation which is triggered by calling loss.backward. So this one was missing. Small line, big impact. So this is very important to calculate the gradients,
Play video starting at :10:12 and follow transcript10:12
good. Let's just clear up everything and run everything again.
Play video starting at :10:19 and follow transcript10:19
Just to see now this is the expected result. So you can see we have losses which are decreasing with the number of epoch; and at some point, they are getting quite stable, this is good.
Play video starting at :10:32 and follow transcript10:32
So now we can for example, visualize the model training by creating a scatter plot in which we are passing in the number of epoch
Play video starting at :10:49 and follow transcript10:49
on the X axis. And then we are going to plot the losses. So this is the whole purpose why we have stored this object.
Play video starting at :11: and follow transcript11:00
And you can see this is a typical representation on how the losses develop. So you start with very high losses at the beginning and then the gradient descent is in action, you can see that you're getting closer and closer to a good result. And at some point, the returns are diminishing more and more. And at some point, you get a very flat line at the end at which point it's not really worth having more calculations. So at that point, you could basically stop the training and work with this kind of model.
Play video starting at :11:41 and follow transcript11:41
We can also do that for the bias terms.
Play video starting at :11:45 and follow transcript11:45
So we can in a very similar fashion create a scatter plot on which we have a range of the 1000 epochs. And then we are going to plot the bias term.
Play video starting at :12:1 and follow transcript12:01
You can see the same, we are starting with a very poor result, we are starting basically at around zero and then we are getting better and better and we are getting at some point to the statistical result of 36 point whatever. So we can also do the same for the slopes.
Play video starting at :12:27 and follow transcript12:27
Same story, we're starting out with some poor results and then a very steady trend towards a constant value at some point. Here it's looking like there is still some significant slope. So we could even think about having more epochs
Play video starting at :12:45 and follow transcript12:45
looking at that graph.
Play video starting at :12:48 and follow transcript12:48
OK. So we now can also create our predictions again by calling our model object.
Play video starting at :12:57 and follow transcript12:57
And in here, we're going to access the data and convert them to numpy,
Play video starting at :13:4 and follow transcript13:04
and we need to remove one dimension to make it work. In the end with the plot we're going to create
Play video starting at :13:12 and follow transcript13:12
and then we can create our scatterplot
Play video starting at :13:15 and follow transcript13:15
with our
Play video starting at :13:19 and follow transcript13:19
x values which are stored in that list
Play video starting at :13:23 and follow transcript13:23
and our y_list. So these are the true values, and now we can put on top the predictions, and we want to be able to differentiate them. So we specify a color
Play video starting at :13:38 and follow transcript13:38
and you can see it actually its better if we take a line plot in this regard.
Play video starting at :13:47 and follow transcript13:47
Here we are. So we have our observations, the blue dots, and this is our final regression line. So this looks already quite good. All right. So you have learned how to setup the Model class, and how to create the training for this specific model.


#%% packages
import numpy as np
import pandas as pd
import torch
import torch.nn as nn 
import seaborn as sns

#%% data import
cars_file = 'https://gist.githubusercontent.com/noamross/e5d3e859aa0c794be10b/raw/b999fb4425b54c63cab088c0ce2c0d6ce961a563/cars.csv'
cars = pd.read_csv(cars_file)
cars.head()

#%% visualise the model
sns.scatterplot(x='wt', y='mpg', data=cars)
sns.regplot(x='wt', y='mpg', data=cars)

#%% convert data to tensor
X_list = cars.wt.values
X_np = np.array(X_list, dtype=np.float32).reshape(-1,1)
y_list = cars.mpg.values
y_np = np.array(y_list, dtype=np.float32).reshape(-1,1)
X = torch.from_numpy(X_np)
y_true = torch.from_numpy(y_np)

#%%
class LinearRegressionTorch(nn.Module):
    def __init__(self, input_size, output_size):
        super(LinearRegressionTorch, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
    
    def forward(self, x):
        return self.linear(x)

input_dim = 1
output_dim = 1
model = LinearRegressionTorch(input_size=input_dim, output_size=output_dim)


# %% Mean Squared Error
loss_fun = nn.MSELoss()

#%% Optimizer
LR = 0.02
# test different values of too large 0.1 and too small 0.001
# best 0.02
optimizer = torch.optim.SGD(model.parameters(), lr=LR)

#%% perform training
losses, slope, bias = [], [], []
NUM_EPOCHS = 1000
for epoch in range(NUM_EPOCHS):
    
    # set gradients to zero
    optimizer.zero_grad()

    # forward pass
    y_pred = model(X)

    # calculate loss
    loss = loss_fun(y_pred, y_true)
    loss.backward()

    # update parameters
    optimizer.step()

    # get parameters
    for name, param in model.named_parameters():
        if param.requires_grad:
            if name == 'linear.weight':
                slope.append(param.data.numpy()[0][0])
            if name == 'linear.bias':
                bias.append(param.data.numpy()[0])


    # store loss
    losses.append(float(loss.data))
    # print loss
    if (epoch % 100 == 0):
        print(f"Epoch {epoch}, Loss: {loss.data}")

# %% visualise model training
sns.scatterplot(x=range(NUM_EPOCHS), y=losses)

#%% visualise the bias development
sns.lineplot(x=range(NUM_EPOCHS), y=bias)
#%% visualise the slope development
sns.lineplot(x=range(NUM_EPOCHS), y=slope)

# %% check the result
y_pred = model(X).data.numpy().reshape(-1)
sns.scatterplot(x=X_list, y=y_list)
sns.lineplot(x=X_list, y=y_pred, color='red')
# %%



Foundations and Core Concepts of PyTorch
Module 7
Exercise: Learning Rate and Number of Epochs

