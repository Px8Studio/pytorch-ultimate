In this module, we will guide you through the process of constructing a neural network from scratch. You will start with data preparation and model initialization and proceed to implement essential functions such as forward and backward propagation. The module also covers training and evaluation techniques to help you build and assess your neural network model effectively.
Learning Objectives
Build a neural network from scratch, including data preparation and model initialization.
Implement forward and backward functions for data propagation and gradient calculation.
Train and evaluate a custom neural network model.

Foundations and Core Concepts of PyTorch
Module 5
Section Overview

0:01
Hello and welcome to this section. In this section we are going to develop our own neural network completely from scratch,
Play video starting at ::10 and follow transcript0:10
and we are following the steps as shown here. I will start with theory. So this is our 101 lecture on Neural Network from scratch.
Play video starting at ::22 and follow transcript0:22
Then we have a small coding lecture on the dot product. And finally, we have the larger topic, a lot of coding lectures on implementing this directly in Python.
Play video starting at ::37 and follow transcript0:37
Actually, I wouldn't say that this section is mandatory, but I really recommend you to follow along because in the end it's much simpler for you to understand how all these things are working if you have at least once implemented all of this on your own. So in the end, this is helping you a lot to understand neural networks in total and it's really good to follow along. So with this, let's just start and take a look at the theory in the next lecture. Thank you very much and see you there.

Foundations and Core Concepts of PyTorch
Module 5
Neural Network from Scratch (101)

0:01
Hello and welcome to this lecture. In this lecture, we are going to implement a neural network completely from scratch. So here there is no PyTorch involved. We are developing every single function on our own. And you learn at first the theory on how to do this.
Play video starting at ::19 and follow transcript0:19
So let's start with the forward pass. We've seen this graph before, I've slightly it. So we're starting with our training data X, and then we are passing it to our first layer, then we have some activation which is applied from which we then get our predictions. And then the predictions together with the true values are used to calculate the loss function and we have the loss score. Then the optimizer is taking place in the backward path and updating the weights. So this is the overall picture, but now we make it a bit more precise.
Play video starting at ::57 and follow transcript0:57
So we have our first layer, here our weights and biases come into play. So here we take our input data or train data, and multiply them with weights, and we add our biases here.
Play video starting at :1:14 and follow transcript1:14
Then we have our activation function and we are going to develop a model for classification. So we are going to use the sigmoid activation function.
Play video starting at :1:27 and follow transcript1:27
So then this is the function that is defining the sigmoid function. And then this sigmoid layer is dependent on the output of the first layer. So S is a function of l1. And this is the definition on how to calculate it.
Play video starting at :1:45 and follow transcript1:45
Then our predictions, these are again depending on the output of the previous layer.
Play video starting at :1:53 and follow transcript1:53
And once we have our predictions, we can calculate our losses.
Play video starting at :1:59 and follow transcript1:59
And for the losses, we are just using the squared error. So we are taking the difference of our predictions and true values, and square them. And we will see why this is beneficial to do.
Play video starting at :2:12 and follow transcript2:12
And that's our forward pass. So the question that now arises is, how do we actually update our weights?
Play video starting at :2:20 and follow transcript2:20
So how does the backward path work
Play video starting at :2:23 and follow transcript2:23
for this? We have a short x course. Let's maybe take a look here at what we got from our forward path. We've spoken about the error. And here you can see we have our
Play video starting at :2:37 and follow transcript2:37
difference of the predictions and the outputs, this is shown horizontally. And the vertical axis is corresponding to our loss that is depending on it. And since we are squaring it, you can see that then this curve is rising, and we want to minimize our loss. So basically, we want to end up here in this point at the center where we have 0. But actually, we won't have 0. We will have any value, like this point L1 or L2. And of course, then the measures are different. So if we have a point of L1,
Play video starting at :3:13 and follow transcript3:13
so where the predictions are smaller than the true values, we're ending up in the lefthand side of this, then we would have to increase our predictions.
Play video starting at :3:24 and follow transcript3:24
On the other hand, if we would have a point of L2, then we would have to decrease the difference that we can see here to also get to this point. So he wants to get to larger values, he wants to get to lower values. So how do we do that?
Play video starting at :3:40 and follow transcript3:40
Well, this is going back to math. So you know that you can put a tangent on a curve. So then we would have these tangents, the red lines which are basically in math called the gradient; and the gradient that is the derivative of a function.
Play video starting at :3:58 and follow transcript3:58
And here we are using the loss function. So this is called our derivative or the gradient, which is also the reason why it is called gradient descent. And we can use this gradient to update the weights. So then we can have different measures whether we are on L1 or L2. The gradient is pointing then into the right direction,
Play video starting at :4:21 and follow transcript4:21
and one positive property of our loss function is actually that if we have larger errors, these are also resulting in larger absolute gradients. So if you're further away from the perfect minimum, then the way you're going to get closer to the minimum is also larger, so larger errors also result in larger absolute gradients. So now we want to speak about a specific aspect which is implemented in all neural networks, which is the learning rate.
Play video starting at :4:53 and follow transcript4:53
Because if we have our, if we have our error L1, and we are adapting our weights, it could be that if we don't have a learning rate that we are jumping to the other side. So we are directly ending up in L2, and then we are adapting again, and we are going directly to the other side. So it's fluctuating or just going in very small steps to the minimum. So for this, we're not adapting the weights totally
Play video starting at :5:20 and follow transcript5:20
with the gradient, but we add a small margin of it. So this is called the learning rate. You can see the formula down here. So here when we are updating our weights, the new weights are calculated based on the old weights and we are subtracting our gradient. But yeah, if we take the total gradient, then we could end up with this problem. So we just take a smaller value. So we adapt this second term
Play video starting at :5:49 and follow transcript5:49
and we do this by multiplying it with a learning rate. So this is just a static number
Play video starting at :5:56 and follow transcript5:56
and the number is below one and above zero, of course. And then if we do so, and we are starting in this point L1, then we are ideally just moving along the curve and getting closer and closer to our minimum. And as I told you, it's between zero and one, but the typical values are 0.01 or 0.001. So something like that we can play with,
Play video starting at :6:24 and follow transcript6:24
but you can also visualize the development of the losses over the time. And then you can see if the learning rate is too large or too small. Okay. The next thing that we're going to need is the Chain Rule. We're still on our way to understand the backward path. So stay with me, we will get there. So the next thing is the chain rule.
Play video starting at :6:50 and follow transcript6:50
Let's take some example. Here we have our input vector X, and then this is passed to a function Y which is actually a function of X, and then we have the next path. Then the next function is called Z. This is a function of the previous one of X. So this is actually very much similar to what we have seen before in our neural network. And how do we get the derivative of Z to X?
Play video starting at :7:22 and follow transcript7:22
Well, we have to take the partial derivatives, and apply the chain rule. So at each step, we can calculate the derivative. So here at the first one Y is a function of X, then we can calculate the derivative Y to the X.
Play video starting at :7:41 and follow transcript7:41
At the next point
Play video starting at :7:43 and follow transcript7:43
here we have Z which is a function of Y. So we are calculating the derivative of this, of the Z to the Y
Play video starting at :7:51 and follow transcript7:51
and
Play video starting at :7:52 and follow transcript7:52
we just have to multiply them, and we have our dz to dx, and thats our chain rule. And this is exactly what we are going to apply in the backward path.
Play video starting at :8:5 and follow transcript8:05
Now, going back to our neural network, here we have our forward path again. And now we want to see how the backward path is working. So we're starting with our losses, we calculate the derivative dL to Dy pred. So then this is just two times y pred minus y true. The next point is to calculate, dy pred to dl 1. So this is the derivative of the sigmoid function. And at the end, we can calculate the derivative of our first layer, dlayer1 to dw, and this is just X.
Play video starting at :8:44 and follow transcript8:44
So our final derivative dL to dw is just the product of all these individual derivatives. So just keep this in mind, this is exactly what we are going to implement in our network. Good. The next point is simple, we just have to do the same for the biases. So the data steps are the same. It's just the first derivative which is adapted, if we calculate the derivative dlayer1 to db , this is then just 1. So this is the only difference here.
Play video starting at :9:22 and follow transcript9:22
And actually that's it. There's just one more technical aspect which I want to tell you about, which is the dot product. So we are using the dot product because finally our weights shall be adapted to create a mapping between our input data and the outputs.
Play video starting at :9:41 and follow transcript9:41
And we have some sample input data or some sample input vector X. And we have two different weight vectors. And the question is, which one is more similar to X. So for this, we can apply the dot product, and you can see the mathematical representation of this, how you can calculate it. So we take the magnitude of the first vector times the magnitude of the second vector times the angle between both vectors.
Play video starting at :10:14 and follow transcript10:14
And the weight is more similar to our input vector if the magnitudes are similar; and if the angles are similar
Play video starting at :10:25 and follow transcript10:25
and the dot product is making our life a bit easier because otherwise we would have to calculate everything quite time intensive. So, so we will see in a second how we can calculate this with the help of numpy. So again, here is just the, the magnitude which is corresponding to the length of the vector and we have our vector in between.
Play video starting at :10:48 and follow transcript10:48
So how do we calculate that? So actually, we don't have to calculate any angles in between. There is a simpler way because the dot product is making it easier. Here we can see our input vector X and our weight one, and some specific values. And the dot product is working the following way. We are just taking our first observations of this vector X and the vector w1, and we multiply them. So this is then the first product, and we do the same for the second index. So this is then ending here, and then we just have to calculate the sum of all of them. And this is the dot product. So this is actually not too complicated. And we will see in the next video how we can calculate this with Python. Thank you very much, and see you there.

Foundations and Core Concepts of PyTorch
Module 5
Calculating the dot-product (Coding)

0:01
Hello and welcome to this lecture. In this lecture we are going to talk about the dot product. So you can find this in the subfolder 015_NeuralNetworkFromScratch. We can just hide this to have more space. And all we need here is to import numpy as np.
Play video starting at ::20 and follow transcript0:20
And then we have here our input vector X and our weight vector w1 and w2.
Play video starting at ::28 and follow transcript0:28
So this is our start. And now at first we calculate this from scratch. So if we want to calculate our dot product of X_w1, then this would be
Play video starting at ::43 and follow transcript0:43
something like this. We have to take the first element of our input vector X, multiply it with the first element of our input vector w1.
Play video starting at ::56 and follow transcript0:56
And then we are taking our next element of the input vector and multiplying it with the second one of our weight vector.
Play video starting at :1:13 and follow transcript1:13
And this is quite annoying. So actually the developers of numpy have developed something better as we will see.
Play video starting at :1:22 and follow transcript1:22
So let's just take a look at the results. We have 3 for this, and now we can just
Play video starting at :1:29 and follow transcript1:29
do the same and adapt it slightly for our second
Play video starting at :1:35 and follow transcript1:35
weight vector.
Play video starting at :1:43 and follow transcript1:43
All right. So we have two different values 3 and 1.8. So which one is more similar to our input vector? Well, it's the other one, it's the weight vector w1,
Play video starting at :1:58 and follow transcript1:58
because larger values correspond here to more similarity, good. But I told you that there is a more simple way to do that. And it's actually really simple because the developers of numpy have implemented this dot functionality where we just have to put in X and w1.
Play video starting at :2:17 and follow transcript2:17
And then we get the same value. And similarly for X and w2.
Play video starting at :2:24 and follow transcript2:24
Perfect. So this was the last piece that we need to understand to be able to develop a complete network from scratch. So this is what we're going to do and start right now in the next lecture. Thank you and see you there.

Foundations and Core Concepts of PyTorch
Module 5
Neural Network from Scratch (Data Prep)

0:01
Hello and welcome to this lecture. In this lecture we are going to develop our neural network completely from scratch, and I've developed here our datascript so that we don't have to import the packages together. I've already prepared that. So we're going to use numpy and Pandas, Seaborn, and then for the data processing, and then for the preprocessing of our training data, we are going to scale it with StandardScalar, and we're also going to use our train_test_split, and at the end we are checking our results with the confusion_matrix.
Play video starting at ::41 and follow transcript0:41
So let's load all of these packages, and let's take a look at the data that we're going to use.
Play video starting at ::49 and follow transcript0:49
So the data is focusing on heart attacks, and we can check that on the Kaggle page and you can see it, it's a classification problem for detecting heart attacks. And we have here
Play video starting at :1:4 and follow transcript1:04
different features like the age, the sex,
Play video starting at :1:8 and follow transcript1:08
and some other health related independent features. And at the end, we have a target which is already binary, it's mapped as 0 or 1, in which 0 is mapped to, there is less chance of a heart attack, and 1 means there is a higher chance of a heart attack.
Play video starting at :1:30 and follow transcript1:30
So this is the data that we're going to work with.
Play video starting at :1:33 and follow transcript1:33
I've already downloaded it and placed it in the working folder so we can import it with pandas and check out the first observations. Here we can see the data that we have. And finally, there is the output
Play video starting at :1:48 and follow transcript1:48
Okay. The first part is that we need to separate our independent from the dependent features. So all the independent features are ending up in X; and our target, our dependent feature is Y, and everything is converted to a numpy array. So here we take all the different columns except for our output column. And the opposite holds for our target value. Here, we only take our output column.
Play video starting at :2:17 and follow transcript2:17
Let's check that.
Play video starting at :2:19 and follow transcript2:19
So now we have here our shapes, we have in total 303 observations, and 13 independent features, and then one target feature. Good. Now we can apply our train_test_split so that we have our training data reserved, which are used in the training phase. And then we have our test data.
Play video starting at :2:43 and follow transcript2:43
So we take as usual the function train_test_split, we put in our independent and dependent feature and we have to specify the test size. I just say I keep 20% for the testing. And
Play video starting at :2:58 and follow transcript2:58
I want to control the randomness and specify here a specific random state.
Play video starting at :3:6 and follow transcript3:06
Okay.
Play video starting at :3:8 and follow transcript3:08
And the last thing that we need to do before we can actually start with our neural network modeling is that we have to scale our data. So if you take a look at the range or the pure numbers inside the columns, you can see here that they're really varying a lot. So we want to scale them and apply some normal distribution based scaling. So we apply the StandardScalar. And for this, we calculate here our X_train_scale object which is the scaled version of our training data. And similarly, then we apply the scaling factors derived from the first step and apply it to our test data. So here we then apply the transform function in contrast to the fit_transform as we have used before. So we directly use just the scaling factors from X_train and apply them to X_test.
Play video starting at :4:10 and follow transcript4:10
Good. So this is let's say classical machine learning. And now we are going to develop our model.


Foundations and Core Concepts of PyTorch
Module 5
Neural Network from Scratch Modeling __init__ Function

0:00
So the first thing is that we are going to create a class.
Play video starting at ::11 and follow transcript0:11
You wouldn't have to structure it in a class. Actually, I like it much more. And it's also similar to how we are going to use it later with PyTorch where we're also creating network classes. Okay? And if we have our class name defined, the first thing is that we need to define some __init__ function.
Play video starting at ::34 and follow transcript0:34
And what we're going to pass during the instantiation of the class is a learning rate;
Play video starting at ::41 and follow transcript0:41
then our train data, X_train, Y_train; and our test data X_test, Y_test.
Play video starting at ::51 and follow transcript0:51
Good.
Play video starting at ::53 and follow transcript0:53
And then we at first start with our weights, they have to be initialized, and we're just going to randomly initialize them. So we can use here
Play video starting at :1:6 and follow transcript1:06
the function randn,
Play video starting at :1:8 and follow transcript1:08
and,
Play video starting at :1:10 and follow transcript1:10
and we'll have here as many weights as we have input columns. So we are using the shape at position one for this, and this is our baseline weights for the start, and we're going to do something very similar with our biases. Here we have,
Play video starting at :1:40 and follow transcript1:40
so here we have just one bias. Okay. Good. Now, we have to create our
Play video starting at :1:49 and follow transcript1:49
internal properties, and assign this with the values that were passed during the instantiation.
Play video starting at :1:56 and follow transcript1:56
So now we have our learning rate created,
Play video starting at :2: and follow transcript2:00
and we do the same for X_train.
Play video starting at :2:27 and follow transcript2:27
So now we have it for the train and the test data. And I'm also going to initialize some other objects,
Play video starting at :2:37 and follow transcript2:37
one for the training losses which I call L_train.
Play video starting at :2:41 and follow transcript2:41
And another one called L_test.
Play video starting at :2:45 and follow transcript2:45
That was the first part.

Foundations and Core Concepts of PyTorch
Module 5
Neural Network from Scratch Modeling Helper Functions

0:00
And now we can create a few more functions. So the first one is our activation function.
Play video starting at ::9 and follow transcript0:09
So the activation function is, as we have said before this is our Sigmoid function.
Play video starting at ::20 and follow transcript0:20
And then we can directly return the function that is defining Sigmoid.
Play video starting at ::33 and follow transcript0:33
You can always check that with the previous video
Play video starting at ::37 and follow transcript0:37
in which I have shown this in the slides. So this is the function for the Sigmoid activation.
Play video starting at ::43 and follow transcript0:43
And then we need our derivative of the activation function, so the derivative of the Sigmoid function,
Play video starting at ::52 and follow transcript0:52
I did not show you the mathematical calculation of this. You can just look it up
Play video starting at ::59 and follow transcript0:59
and it's defined in the following way.
Play video starting at :1:15 and follow transcript1:15
So it's the sigmoid
Play video starting at :1:18 and follow transcript1:18
times
Play video starting at :1:20 and follow transcript1:20
1 - the Sigmoid.
Play video starting at :1:30 and follow transcript1:30
Okay. If you want to check that, you can just go to the internet, look for it. And then you can find here the definition, the derivative of Sigmoid is the Sigmoid function multiplied by 1 -, the Sigmoid function. Good.
Play video starting at :1:44 and follow transcript1:44
That's the next part. And that was what we needed to prepare before we can actually do our forward pass.

Foundations and Core Concepts of PyTorch
Module 5
Neural Network from Scratch Modeling Forward Function

0:00
So we are defining a function called forward.
Play video starting at ::5 and follow transcript0:05
And this one is taking in all the internal parameter states and some
Play video starting at ::13 and follow transcript0:13
variable X.
Play video starting at ::15 and follow transcript0:15
So here we are calculating now layer by layer the results. In the first hidden we calculate our dot product.
Play video starting at ::25 and follow transcript0:25
So the dot product of our input object X
Play video starting at ::31 and follow transcript0:31
with the weights.
Play video starting at ::33 and follow transcript0:33
And then we are
Play video starting at ::36 and follow transcript0:36
also adding here the biases.
Play video starting at ::39 and follow transcript0:39
So this is our first hidden state.
Play video starting at ::43 and follow transcript0:43
And then we have our activate layer,
Play video starting at ::47 and follow transcript0:47
and here we just have to call our activation function and pass in the result from the previous step.
Play video starting at ::55 and follow transcript0:55
So this is just the mathematical representation of the slides we've seen before.
Play video starting at :1:1 and follow transcript1:01
And here we have a super simple network, we can directly then return the result of this layer. And this is our complete forward pass.

Foundations and Core Concepts of PyTorch
Module 5
Neural Network from Scratch Modeling Backward Function

0:00
Now we need to setup our backward pass.
Play video starting at ::4 and follow transcript0:04
So here we put in some input vector of the independent features X and also some true values, because we need to calculate some losses in here.
Play video starting at ::16 and follow transcript0:16
So here we're going to calculate our gradients.
Play video starting at ::21 and follow transcript0:21
So we need at first again our hidden layer,
Play video starting at ::28 and follow transcript0:28
I can just copy paste this,
Play video starting at ::33 and follow transcript0:33
then we can calculate our predictions.
Play video starting at ::39 and follow transcript0:39
We can tell you to do this with the forward pass.
Play video starting at ::44 and follow transcript0:44
We're just stacking up all the different pieces that we are going to use in the end to calculate the derivative of the loss function with respect to the weight vector and with respect to the bias vector.
Play video starting at ::59 and follow transcript0:59
OK. So this is the second ingredient we need. Then we need our derivative of the losses to the predictions.
Play video starting at :1:9 and follow transcript1:09
And that as we've seen is two times
Play video starting at :1:14 and follow transcript1:14
y_pred
Play video starting at :1:16 and follow transcript1:16
minus y_true.
Play video starting at :1:20 and follow transcript1:20
Good. That's the next piece.
Play video starting at :1:23 and follow transcript1:23
And then we have our
Play video starting at :1:26 and follow transcript1:26
derivative of their predictions to the hidden layer,
Play video starting at :1:31 and follow transcript1:31
dpred_ dhidden1.
Play video starting at :1:38 and follow transcript1:38
So here we can use our derivative of the activation function.
Play video starting at :1:43 and follow transcript1:43
So for this purpose, we have created this additional function,
Play video starting at :1:48 and follow transcript1:48
and here we put in our hidden layer.
Play video starting at :1:55 and follow transcript1:55
Good. And then finally we have our derivative of the hidden layer 1
Play video starting at :2:1 and follow transcript2:01
and our biases, we have seen that as 1, and dhidden1
Play video starting at :2:9 and follow transcript2:09
to dw, that is just X.
Play video starting at :2:14 and follow transcript2:14
So we're just taking all the pieces from the slide and putting it into code. And now we can calculate our derivatives dL_db,
Play video starting at :2:26 and follow transcript2:26
that is dL
Play video starting at :2:28 and follow transcript2:28
_dpred. So the product is here calculated of the different pieces. Now we have the,
Play video starting at :2:36 and follow transcript2:36
dhidden1,
Play video starting at :2:40 and follow transcript2:40
and we have dhidden1_db
Play video starting at :2:44 and follow transcript2:44
good. That's our first. And then we want to calculate the derivative of the layer to dw,
Play video starting at :2:54 and follow transcript2:54
that is very similar to the dL_
Play video starting at :2:57 and follow transcript2:57
d pred times
Play video starting at :3:1 and follow transcript3:01
d pred_ dhidden1 times
Play video starting at :3:6 and follow transcript3:06
d hidden1_ dw
Play video starting at :3:11 and follow transcript3:11
good. And once we've done that, we can return everything.
Play video starting at :3:16 and follow transcript3:16
So we can return the derivative of the loss to the bias, and the loss to the weights,
Play video starting at :3:26 and follow transcript3:26
good. That is our backward path. So this was the most complicated part in here because here we have all these different derivatives and yeah, this is already done now. And the next parts are simpler.


Foundations and Core Concepts of PyTorch
Module 5
Neural Network from Scratch Modeling Optimizer Function

0:00
So the optimizer,
Play video starting at ::2 and follow transcript0:02
this is just consuming some values like their derivatives are passed in here.
Play video starting at ::9 and follow transcript0:09
And then based on this, we can update the weights.
Play video starting at ::18 and follow transcript0:18
So the bias is just calculated as the bias minus
Play video starting at ::25 and follow transcript0:25
the derivative, dL_db
Play video starting at ::30 and follow transcript0:30
times, and now the important part is that we multiply this with the learning rate,
Play video starting at ::37 and follow transcript0:37
good. And the same we're doing for the weights. So this is just our weight. And here again, we are subtracting the derivative of the losses to the weights, and multiply it with the learning rate,
Play video starting at ::56 and follow transcript0:56
good. So now we have updated our weights.

Foundations and Core Concepts of PyTorch
Module 5
Neural Network from Scratch Modeling Train Function

0:00
And now we need to specify how the training is done. So during the training, we are iterating over all the observations in the dataset. And we need to set up this train loop. We're going to do this in a function called train.
Play video starting at ::18 and follow transcript0:18
And here just pass in value ITERATIONS.
Play video starting at ::23 and follow transcript0:23
And here we can now iterate over the range of the ITERATIONS.
Play video starting at ::32 and follow transcript0:32
So at first
Play video starting at ::35 and follow transcript0:35
we get some random position,
Play video starting at ::41 and follow transcript0:41
so it's often a good idea to just pick the values randomly out of the train data. So for this, we set up a variable called random_pos.
Play video starting at ::54 and follow transcript0:54
And here we can get a random integer value with randint.
Play video starting at :1:2 and follow transcript1:02
We just have to make sure that it's in a certain range. So we want to get only values from zero to the length of the train data.
Play video starting at :1:19 and follow transcript1:19
So here we are just defining the indices for picking the right element,
Play video starting at :1:27 and follow transcript1:27
good. Now we can implement the forward pass.
Play video starting at :1:31 and follow transcript1:31
So here, first for a comparison, later on we're
Play video starting at :1:35 and follow transcript1:35
picking our true train value, and this is stored in self.X_train at the position random_pos which we have just defined.
Play video starting at :1:48 and follow transcript1:48
And now we need to get our prediction y_train_pred.
Play video starting at :1:54 and follow transcript1:54
And for this, we are using our forward pass, and in here we pass our train data
Play video starting at :2:2 and follow transcript2:02
at the same position.
Play video starting at :2:6 and follow transcript2:06
So by now we have a prediction value and a true value. This is all that we need to calculate the losses,
Play video starting at :2:16 and follow transcript2:16
the training losses.
Play video starting at :2:20 and follow transcript2:20
So this loss is calculated by summing up
Play video starting at :2:26 and follow transcript2:26
all the different squared differences.
Play video starting at :2:30 and follow transcript2:30
So we have the np.square function, and in here we have our y_train pred,
Play video starting at :2:37 and follow transcript2:37
and
Play video starting at :2:39 and follow transcript2:39
we subtract y_ train_ true
Play video starting at :2:51 and follow transcript2:51
good. And finally, we want to check that in a graph. So we will just append our training losses with the current loss that we have just calculated. So this is just for checking out the result afterward.
Play video starting at :3:8 and follow transcript3:08
And now we can calculate the gradients
Play video starting at :3:13 and follow transcript3:13
dL_db, and DL_ dw.
Play video starting at :3:19 and follow transcript3:19
And for this, we can call our backward function.
Play video starting at :3:24 and follow transcript3:24
In here, this is just consuming some data like the train data,
Play video starting at :3:30 and follow transcript3:30
X_train at the position that we have defined, and Y_train
Play video starting at :3:39 and follow transcript3:39
at the random position,
Play video starting at :3:41 and follow transcript3:41
good. Now we have our derivatives, and now we can update the weights.
Play video starting at :3:47 and follow transcript3:47
So for this, we can call self dot optimizer,
Play video starting at :3:52 and follow transcript3:52
and just pass in our derivatives that we have just defined dL_db and dL_dw,
Play video starting at :4:7 and follow transcript4:07
good.
Play video starting at :4:8 and follow transcript4:08
Actually, that's it. I just want to check how this is performing also on the test data. So at every end of one iteration, we are going to calculate the error for the testing.
Play video starting at :4:27 and follow transcript4:27
Here again I set up some variable the loss sum is zero at the beginning. And now we're iterating over very similarly to before
Play video starting at :4:40 and follow transcript4:40
the length of our test data.
Play video starting at :4:45 and follow transcript4:45
So we pick again our true value y_true. This is just picked from our
Play video starting at :4:54 and follow transcript4:54
y_test at the position j,
Play video starting at :4:59 and follow transcript4:59
and the predictions
Play video starting at :5:2 and follow transcript5:02
are calculated by calling our forward function,
Play video starting at :5:11 and follow transcript5:11
here we put in our index j.
Play video starting at :5:15 and follow transcript5:15
So now we have our predictions, now we can update our sum.
Play video starting at :5:22 and follow transcript5:22
So just do it in one step. I use this plus equal sign because now we can calculate our sum for this specific loss and directly add it to our object L_sum.
Play video starting at :5:36 and follow transcript5:36
So here we are using our square function, and we can calculate the difference of y_pred, and y_true,
Play video starting at :5:48 and follow transcript5:48
good. And once we have that, once the loop is done, we can add it to the final loss list and append here our L_sum,
Play video starting at :6:4 and follow transcript6:04
good.
Play video starting at :6:7 and follow transcript6:07
Alright. And then we are at the end of our train function, we can just say something like, training successful.


Foundations and Core Concepts of PyTorch
Module 5
Neural Network from Scratch Model Training

0:01
Good. Let's just see if this all works good. The class can be created. Now, we can set up some hyperparameters. So the first one is the learning rate.
Play video starting at ::13 and follow transcript0:13
Let's say we start with a value of 0.1. And the iterations,
Play video starting at ::20 and follow transcript0:20
here we're using 1000 iterations.
Play video starting at ::26 and follow transcript0:26
Now we're going to create an instance of our model and I just call it nn,
Play video starting at ::33 and follow transcript0:33
for saving some typing. So here we have our model class NeuralNetworkFromScratch, so the learning rate, this is LR, we just pass all the different parameters that we have defined here. X_train is X_
Play video starting at ::51 and follow transcript0:51
train_scale. Just make sure that you use the right one, Y_train
Play video starting at ::58 and follow transcript0:58
is passed as y_train
Play video starting at :1:1 and follow transcript1:01
and then X_test
Play video starting at :1:4 and follow transcript1:04
is passed as X_test_scale; and y_test,
Play video starting at :1:9 and follow transcript1:09
here we also pass the y_test;
Play video starting at :1:15 and follow transcript1:15
good. And now we can check if everything is working. We can call our train function,
Play video starting at :1:21 and follow transcript1:21
and pass in the ITERATIONS,
Play video starting at :1:27 and follow transcript1:27
and then the training should run.
Play video starting at :1:33 and follow transcript1:33
Okay. There is a small error. Let's see here. This, of course, I forgot to call the property shape. Okay? Lets just run it again
Play video starting at :1:45 and follow transcript1:45
and see if it's now working. training successful, that's good.

