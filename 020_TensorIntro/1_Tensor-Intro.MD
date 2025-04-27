Foundations and Core Concepts of PyTorch
Module 6
Section Overview

0:01
Hello and welcome to this section of Tensors. In this introduction, you will learn about the backbone of PyTorch which are Tensors. You will learn what Tensors are, what you can do with them, how to create them, and so on. So let's see what we can expect from this section.
Play video starting at ::22 and follow transcript0:22
So I said we're going to mainly work with Tensors in the coding lecture, and you will see how to create them and how to modify them. But there's also one very important aspect. You will learn about how the Tensors are interlinked with each other. So you will learn how one Tensor is linked to another Tensor. And this feature is pretty related to something that is called Autograd.
Play video starting at ::54 and follow transcript0:54
So all these details are going to be covered in this lecture. So let's get started.

Foundations and Core Concepts of PyTorch
Module 6
From Tensors to Computational Graphs (101)

0:01
Hello and welcome to the 101 on Tensors.
Play video starting at ::6 and follow transcript0:06
Lets start with some very basic introduction. We have our PyTorch tensors which are basically forming the structure to work with the variables.
Play video starting at ::18 and follow transcript0:18
And you can imagine to be the tensor is pretty similar to numpy arrays, but they are much more powerful as you can see because that is one of the reasons they can automatically calculate the gradients. And I will show that in a second how this is working.
Play video starting at ::38 and follow transcript0:38
And then they're also storing the information about dependencies to other tensors. So these two last items here are the main features which are making the tensors so extremely powerful. But let's take a look at some examples.
Play video starting at ::55 and follow transcript0:55
So let's speak about automatic gradient.
Play video starting at ::58 and follow transcript0:58
That is meaning like the name is saying the gradients can be calculated automatically. And you can see here I have created a tensor X
Play video starting at :1:8 and follow transcript1:08
with a value, and also with the property requires_grad equals True. And then we have another tensor called y which is a function of x basically.
Play video starting at :1:23 and follow transcript1:23
And then just by calling the backward function on y,
Play video starting at :1:27 and follow transcript1:27
you are getting all the gradients updated. And then if you're printing the gradient of x, you can see like magic that you get back the value of 31.
Play video starting at :1:40 and follow transcript1:40
And if you plot that like you can see it here at the right hand side, you can see here this is actually a representation of this function. So we have y over x, and then if we calculate the derivative of Y to X, then this will provide for a given position of 1, the derivative of 31,
Play video starting at :2:6 and follow transcript2:06
which is basically the slope at this position of the curve.
Play video starting at :2:11 and follow transcript2:11
Great.
Play video starting at :2:13 and follow transcript2:13
But what can you do with that? So we will come to that.
Play video starting at :2:18 and follow transcript2:18
Let's imagine that you're working with a very simple network. So you have an input x, and you use it to calculate another variable y which is then itself being used to calculate z.
Play video starting at :2:33 and follow transcript2:33
So here you can see it, we have x, y, and z
Play video starting at :2:36 and follow transcript2:36
and we have the interlink between these. So y is based on x, and z is based on y.
Play video starting at :2:45 and follow transcript2:45
So then we have certain formulas defining how these links are defined. And this is basically what we can call a forward pass. Later on when we're working with our networks, we are using the very same terminology, so we are always saying we are making now a forward pass through the network or a backward pass. Now here we'll see what this basically means on the most simple examples,
Play video starting at :3:15 and follow transcript3:15
because then the backward pass
Play video starting at :3:18 and follow transcript3:18
that is basically the back propagation, which means that in this step the derivatives are calculated. So the derivative of z to y and the derivative of y to x.
Play video starting at :3:35 and follow transcript3:35
So basically you can
Play video starting at :3:37 and follow transcript3:37
calculate the change of z based on a change of x. So you can see the derivative of z to x. And for this, you can apply the chain rule. And if you're doing that based on these formulas which are given here, then you can calculate this first derivative to be 5, and the second one to be 3 x squared.
Play video starting at :4:8 and follow transcript4:08
All right. So far, this is not a very super network. We only have an input node and output node and a hidden node, and we don't even have the weights. So let's now take a look at how the weights could be added, and how the weights can be updated. So here we have the same network x, y, and z; and then we have the weights assigned. So we have a weight assigned to z, and a weight assigned to y.
Play video starting at :4:36 and follow transcript4:36
So we want to calculate our output z in the end. And
Play video starting at :4:42 and follow transcript4:42
we can expect that this is just a calculation because the true output deviates from our calculation. So we are only getting closer and closer to the reality. And for this, we are calculating some error, which is basically the difference of z and t and
Play video starting at :5:4 and follow transcript5:04
to have only positive errors usually this error is squared.
Play video starting at :5:11 and follow transcript5:11
And then you can see here I have attached the weights,
Play video starting at :5:15 and follow transcript5:15
and actually the weights can also be considered to be nodes very similar to the other ones.
Play video starting at :5:21 and follow transcript5:21
And then if you treat them as nodes, then you can consider that to be a function of y and w2.
Play video starting at :5:32 and follow transcript5:32
And then you can go backwards in the train. And basically what the optimizer in our network is doing, it's updating the weights based on these gradients that we are seeing here.
Play video starting at :5:45 and follow transcript5:45
And then to get closer and closer to a real network, all we have to do is scaling. So let's assume now that we have multiple inputs. So now we have two inputs, X 11 and X 21 and then a hidden layer and then an output layer, then we can also assume that we have a fully connected network. So our X 11 is connected to the first node of the hidden layer, and the second node of the hidden layer; the same holds for X 21.
Play video starting at :6:19 and follow transcript6:19
And then you can see the dependencies. So X 12 is depending on the first input node and the second input node; and the same holds for X 22. And then the output layer is taking a look at one layer before, and it's depending on X 12 and X 22.
Play video starting at :6:42 and follow transcript6:42
So this is basically forming our forward pass.
Play video starting at :6:46 and follow transcript6:46
And then for updating the weights,
Play video starting at :6:50 and follow transcript6:50
all we have to do is calculate the derivatives, and then we can get the weights basically. So let's take a look at some example. Let's say we are having these formulas which are defining the relationship between X 12 and the input nodes, and X 22.
Play video starting at :7:11 and follow transcript7:11
And also y is depending with this formula on the hidden layer,
Play video starting at :7:19 and follow transcript7:19
then all we have to do is calculating the gradients.
Play video starting at :7:24 and follow transcript7:24
So you can pause the video and recollect your math knowledge and check all the calculations that I've done here.
Play video starting at :7:34 and follow transcript7:34
And I come up with these two different derivatives; and then with a given X 11, we can calculate the slope to be 44, for dy to dx 11; and correspondingly the other derivative. All right. That was the introduction on our tensors. And now in the coding lecture, you will be able to see all this in action. We are going to create these graphs on our own and perform these calculations directly with PyTorch. Thank you and see you there.

Foundations and Core Concepts of PyTorch
Module 6
Tensor (Coding)

0:01
Hello and welcome to this coding lecture. In this lecture we're going to work with Tensors.
Play video starting at ::8 and follow transcript0:08
So you can see here, I've already opened up my Visual Studio Code, and I'm working from my working folder,
Play video starting at ::16 and follow transcript0:16
and I can go to file, New Text File, or I can just press Ctrl + N,
Play video starting at ::23 and follow transcript0:23
and then I'm asked to select the language and we are going to say this is a Python script that we are going to develop, right? You can see already this small icon here which is pointing to Python. You can see here in the bottom, right? This is the conda environment that we are going to use and that we have created earlier. So everything is setup and now we are ready to start. So
Play video starting at ::50 and follow transcript0:50
we are working with chunks, so we can create chunks with this notation hashtag percent and then everything inside this area until the next chunk is starting is going to be executed. So the first chunk that we're going to use is dedicated to the packages. And in here, we're saying we want to import torch, and we want to import numpy
Play video starting at :1:20 and follow transcript1:20
as np, the classical alias.
Play video starting at :1:25 and follow transcript1:25
And for some small visualization, we're going to use Seaborn, and we are importing this with the typical abbreviation sns. All right, you can click on Run Cell, and you can see that this interactive window is opening, and that this code chunk is now executed. And once everything was run successfully, you can see this green checkmark. Great, no error, everything worked. All the packages are found and can be loaded. Perfect. So we are up and running.
Play video starting at :2:4 and follow transcript2:04
So let's start very simple, and we are going to create a vector,
Play video starting at :2:10 and follow transcript2:10
and we're going to create a tensor.
Play video starting at :2:14 and follow transcript2:14
So let's call the tensor x, and that's going to be created by calling torch.tensor,
Play video starting at :2:22 and follow transcript2:22
and providing a value 5.5, we can run that. So now we have created this tensor, we can do some very simple calculation.
Play video starting at :2:35 and follow transcript2:35
So for example, if we say the variable y is dependent on x plus whatever offset,
Play video starting at :2:44 and follow transcript2:44
then we can also print our variable y. And you can see this is behaving very much like numpy. Everything is the same so far.
Play video starting at :2:56 and follow transcript2:56
And one difference here is actually if we check, the property requires_grad,
Play video starting at :3:5 and follow transcript3:05
that is False. So if you create a tensor in that way, and if you don't specify the parameter requires_grad=True, then the default is actually that it doesn't require gradient.
Play video starting at :3:21 and follow transcript3:21
So let's move on,
Play video starting at :3:25 and follow transcript3:25
let's create
Play video starting at :3:28 and follow transcript3:28
x again. So that's again going to be a tensor,
Play video starting at :3:33 and follow transcript3:33
and it's having the value 2.0. And this time we are setting the flag requires_grad=True.
Play video starting at :3:41 and follow transcript3:41
And then if we are checking that
Play video starting at :3:44 and follow transcript3:44
property x.requires_grad, then we are seeing, okay this is now enabled, great.
Play video starting at :3:52 and follow transcript3:52
What we're going to do now is that we are going to create a function for the variable y. And then first of all, we are going to check how x and y are behaving, and then we are going to calculate the gradients. So basically the slopes of y versus x at specific points. So let's define our function. And let's take a look at
Play video starting at :4:23 and follow transcript4:23
the function plot.
Play video starting at :4:27 and follow transcript4:27
So let's just create a function for calculating the y values
Play video starting at :4:34 and follow transcript4:34
and we directly return
Play video starting at :4:39 and follow transcript4:39
the value -3
Play video starting at :4:43 and follow transcript4:43
times value,
Play video starting at :4:47 and follow transcript4:47
value -6
Play video starting at :4:50 and follow transcript4:50
times value
Play video starting at :4:54 and follow transcript4:54
-4. All right, that's our function.
Play video starting at :4:58 and follow transcript4:58
And we are now, just for visualization purposes we are creating a range which we call x_ range, and we can call from numpy the function linspace for linearly spacing for a certain range. And we want to have the range from 0 to 10,
Play video starting at :5:21 and follow transcript5:21
and 101 points along this line.
Play video starting at :5:27 and follow transcript5:27
And then we can calculate the y values accordingly. So this is basically
Play video starting at :5:34 and follow transcript5:34
just
Play video starting at :5:36 and follow transcript5:36
using list comprehension
Play video starting at :5:39 and follow transcript5:39
for each element.
Play video starting at :5:45 and follow transcript5:45
For each element i in the range of x, we are going to apply the y function that we have defined,
Play video starting at :5:54 and follow transcript5:54
and then we are saving everything in a list,
Play video starting at :5:59 and follow transcript5:59
and
Play video starting at :6:1 and follow transcript6:01
then we can plot what we have here. So we are having
Play video starting at :6:7 and follow transcript6:07
an x variable
Play video starting at :6:9 and follow transcript6:09
which is x_range and y is equal to y_range.
Play video starting at :6:14 and follow transcript6:14
And then you can see this is basically the function that we are going to work with. And we will see how we can directly calculate the gradients of our function at specific x locations with the help of PyTorch.
Play video starting at :6:33 and follow transcript6:33
Okay. So far we have defined our tensor x. Here we go with the gradients enabled. Now we're going to calculate y.
Play video starting at :6:47 and follow transcript6:47
So the same function that we have defined before is going to be used right now. And you can see here, GitHub Copilot in action. So the proposal was perfectly right.
Play video starting at :7:1 and follow transcript7:01
We have now the same function here for the definition of y and now we can call y.backward.
Play video starting at :7:10 and follow transcript7:10
And as we have heard before, this is the function to actually calculate the gradients.
Play video starting at :7:17 and follow transcript7:17
So once we call that,
Play video starting at :7:21 and follow transcript7:21
and then if we print
Play video starting at :7:24 and follow transcript7:24
the gradient of x,
Play video starting at :7:27 and follow transcript7:27
then you can see this is having the value of 14, this is basically the slope at the position of x equals 2.
Play video starting at :7:40 and follow transcript7:40
So it might not look like much, but it is actually quite cool because all we have setup upto this point is our two tensors x and y, y is a function of x; and just calling this function and we get the slope of x.
Play video starting at :7:57 and follow transcript7:57
So that's, that's quite cool. And that is something that is going to be used a lot later on with the neural networks that we are going to develop. All we have to do then to calculate all gradients is basically calling the backward function, and then the complete computational graph is calculated. So that's, that's quite awesome.
Play video starting at :8:19 and follow transcript8:19
Lets do it with a second example.
Play video starting at :8:24 and follow transcript8:24
Let's say we have defined x,
Play video starting at :8:32 and follow transcript8:32
and this is having the value 1, and requires_grad is enabled.
Play video starting at :8:38 and follow transcript8:38
Then we are saying y, this is
Play video starting at :8:42 and follow transcript8:42
a function depending on x,
Play video starting at :8:46 and follow transcript8:46
and
Play video starting at :8:48 and follow transcript8:48
the graph goes on. Now we have another variable called z,
Play video starting at :8:53 and follow transcript8:53
that one is itself depending on y, with whatever formula, all the values don't play any role here. All I want to show you is basically how everything interacts with each other.
Play video starting at :9:7 and follow transcript9:07
Now, we can start with the last node of this which is called z, and we are calling the backward function to calculate all the gradients.
Play video starting at :9:17 and follow transcript9:17
Let's run that. It ran successfully as you can see here. And now in the next run
Play video starting at :9:25 and follow transcript9:25
we can print our gradient with
Play video starting at :9:30 and follow transcript9:30
taking our object x, and the property grad.
Play video starting at :9:34 and follow transcript9:34
And this is now returning the value of 5.
Play video starting at :9:40 and follow transcript9:40
So far we only had some very linear approach. We had the first node which is connected to the second node which was connected to the third node. Now we are going to get closer and closer to a real network, meaning that we are having now two inputs x11
Play video starting at :10:2 and follow transcript10:02
which is going to be created by calling torch .tensor.
Play video starting at :10:7 and follow transcript10:07
It's having a value of 2.0 and requires_grad equals True. Then we have another node in the input layer.
Play video starting at :10:17 and follow transcript10:17
I'm already using the terminology of neural networks, although this is just a simple computational graph.
Play video starting at :10:25 and follow transcript10:25
So here we have now a tensor with a value of 3,
Play video starting at :10:30 and follow transcript10:30
and now we have the second layer. So here we have a formula. I've just come up with certain values. You can also use different ones and then try to calculate how everything works together by calculating the derivatives.
Play video starting at :10:52 and follow transcript10:52
So let's say we have now the second node which is defined as two times
Play video starting at :10:59 and follow transcript10:59
x
Play video starting at :11:1 and follow transcript11:01
11 script
Play video starting at :11:5 and follow transcript11:05
plus two times
Play video starting at :11:8 and follow transcript11:08
x21;
Play video starting at :11:14 and follow transcript11:14
and then we have another layer. So this is our output layer. Now we have four times X12
Play video starting at :11:22 and follow transcript11:22
plus three times X22.
Play video starting at :11:28 and follow transcript11:28
So basically, if you would draw a graph based on that, or go back to the last slides that I have shown on this, we have here two nodes on the input layer, then one hidden layer, and then the output layer with one node.
Play video starting at :11:46 and follow transcript11:46
All right. So now we can at the end, just call the backward function on our final node.
Play video starting at :11:54 and follow transcript11:54
And
Play video starting at :11:55 and follow transcript11:55
finally, we can
Play video starting at :11:58 and follow transcript11:58
take all objects x11 and x21, and show the gradients for this. And then you can see here these are the slopes at these positions that we have defined for this specific network.
Play video starting at :12:16 and follow transcript12:16
You can also go back to the introduction slide set, and check the values;
Play video starting at :12:23 and follow transcript12:23
or you could just modify everything, and play with it on your own. So what I wanted to show you in this very first coding lecture was actually how simple it is to create some tensors, and how the different tensors can be connected to each other. And then basically how the backward pass can be employed to get back the gradients of the variables.
Play video starting at :12:53 and follow transcript12:53
So that is bringing us very much closer to some real neural network. You have seen here the Autograd functionality in action. And with this, I want to conclude this lecture. Thank you very much, and see you in the next one.

#%% packages
import torch
import seaborn as sns
import numpy as np

#%% create a tensor
x = torch.tensor(5.5)

# %% simple calculations
y = x + 10
print(y)

# %% automatic gradient calculation
print(x.requires_grad)  # check if requires_grad is true, false if not directly specified

x.requires_grad_() # set requires grad to true, default True

#%% or set the flag directly during creation
x = torch.tensor(2.0, requires_grad=True)
print(x.requires_grad)
#%% function for showing automatic gradient calculation
def y_function(val):
    return (val-3) * (val-6) * (val-4)

x_range = np.linspace(0, 10, 101)
x_range
y_range = [y_function(i) for i in x_range]
sns.lineplot(x = x_range, y = y_range)

# %% define y as function of x
y = (x-3) * (x-6) * (x-4)
print(y)
# %%

# %% x -> y
# create a tensor with gradients enabled
x = torch.tensor(1.0, requires_grad=True)
# create second tensor depending on first tensor
y = (x-3) * (x-6) * (x-4)
# calculate gradients
y.backward()
# show gradient of first tensor
print(x.grad)
# %% x -> y -> z
x = torch.tensor(1.0, requires_grad=True)
y = x**3
z = 5*y - 4

# %%
z.backward()
print(x.grad)  # should be equal 5*3x**2
# %% more complex network
x11 = torch.tensor(2.0, requires_grad=True)
x21 = torch.tensor(3.0, requires_grad=True)
x12 = 5 * x11 - 3 * x21
x22 = 2 * x11**2 + 2 * x21
y = 4 * x12 + 3 * x22
y.backward()
print(x11.grad)
print(x21.grad)
# %%
