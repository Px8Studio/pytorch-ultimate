In this module, we will introduce you to PyTorch modeling. You will learn to build and train models from scratch, including linear regression. The module covers batch processing, datasets, and dataloaders to manage data effectively. You will also explore techniques for saving, loading, and optimizing models, including hyperparameter tuning, to enhance your machine learning workflow.

### Learning Objectives
- Construct and train machine learning models using PyTorch.
- Implement batch processing, datasets, and dataloaders for efficient data management.
- Save, load, and optimize models through hyperparameter tuning.

---

### Foundations and Core Concepts of PyTorch
#### Module 7
#### Section Overview

Hello and welcome to our next section which is covering modeling. Here, I'm going to provide you some short section overview so you will see what you can expect of this.

Basically, you will learn about the model training with all the details. You will learn about how to set up a model by creating and inheriting from the `nn.Module` class. Then you will see how to set up the training loop so that your model can be trained successfully.

After the model is trained successfully, you will be able to evaluate how good the model was trained.

Then also we are going to work with datasets, and you will see how beneficial it is to separate your data from your modeling. Here, the datasets and also the dataloaders come into play.

We are putting another level of complexity to it. You will learn about batches, what batches are, and why you're using them.

You will see different activation functions in action, and to get the best results you need to tune the parameters. So we will also work on that. You will learn about hyperparameter tuning.

Last but not least, you will not want to retrain the model every time you're opening up your coding environment. Basically, you want to just save the model and load it at some later point so that you can right away start with the model.

All right. You can see there is a bunch of topics which we will cover in this lecture, and this section is basically laying the foundation for everything else. The other lecturers later on are covering specific topics, very specific architectures of networks. But here all the groundwork is done. I would really recommend to you not to skip this one, and pay a lot of attention in the next lectures. Thank you very much, and see you there.

---

### Foundations and Core Concepts of PyTorch
#### Module 7
#### Linear Regression from Scratch (Coding, Model Training)

Hello and welcome to this lecture. In this lecture, we're going to create our very first model, and we are creating it completely from scratch. I hope you are excited to build your very first neural network.

I have created here two different files in the folder `ModelingIntroduction`. One is called `end`, which is having all the code that we are going to develop. The other one is called `start`, because it contains all the import statements of the packages and some data retrieval and stuff like that—everything that is not related to the actual creation of the network. You can use this start script, and then we will build the next steps in here. If you're a different learning type, and if you rather want to follow along on what I'm doing, just watch the video. But I would highly recommend to you that you actually code with me because that is a much more productive way to understand what is happening here.

Now we have everything set up. We have here our start script, and we are going to start by importing the packages. We are using `numpy`, `pandas`, and `torch`.

It is usual practice to load certain submodules of `torch` under different aliases. For example, `torch.nn` is usually imported as `nn`, just to save some typing.

We're also going to use `seaborn` for some simple visualization. That is basically everything that we need for creating the first network. Then we're going to work with some data on cars. You can see the head of this dataframe. This is a very classical dataframe of which we are going to use two different properties or two different columns: the weight and the mpg.

If we plot this, you can see the observations. We have here some decreasing trend. Not surprisingly, if the weight of a car is increasing, then the mpg is decreasing. On top, I also put here some regression line because this is what we're going to replicate with our regression model that we're going to create.

At first, we're going to convert the data to tensors. We have the data already in `cars.wt.values`. That would be our first data. Then we can convert that to `numpy`. I will show you different ways how to do that actually.

We have the list that we have created before now converted to some array. Then we also need to specify some data type. This is some float data, and we need to reshape it.

Let's take a look at it. Now we have here our data with a certain shape, which is basically the way that PyTorch is expecting. We have 32 observations and 1 column, 1 independent feature. Then we are also going to work with the target variable, which is `y`.

This we are going to extract as a list, and now we can convert that directly to some tensors. I will show you two different ways. One is if the data is already provided in some `numpy` format, then you can use the function `from_numpy`. Another way is if the data is just provided as a list, for example, then you can create a tensor out of it with the known function `torch.tensor`.

Now we have the data in the proper format. This is the `x` tensor, and this is our target feature.

We're going to create now our weights. The weights are randomly initialized. Basically, it's just one value. We have to specify the property `requires_grad`, because this is going to be updated. Then we're also going to create that as a plot.

Similarly, we are going to create our bias. Usually with linear regression, you have a term for the slope and another one for the bias. Here we have the weight and the bias.

Then we can specify the number of epochs. Let's see, the number of epochs is 1000, and we specify one specific learning rate.

The data is fed to the neural network for training. Once all the data is provided to the network, then this is called one epoch. The weights are adapted, and the next iteration starts. This is the next epoch. Just to know what the epochs are, we are saying we want to iterate 1000 times and provide the data 1000 times to the network for training.

We are specifying here that we are iterating over all these epochs with this operation, and then each time we are passing one observation. This is our inner loop.

Now in this inner loop, we're going to calculate our predictions. Let's call them `y_pred`. That is our current observation of the independent feature. So `x` at the position of `i` times our weight tensor plus our bias tensor. This is our forward pass.

Then we are going to calculate our loss. Our loss tensor is going to be `torch.pow`. We're creating the square of the difference of `y_pred`, which we just calculated, minus the true value at this current position `i`.

Now we're having our backward pass in which we are calculating the gradients. Here, all we have to do is call the `backward` function on the loss tensor.

Now we are extracting the losses. This value is basically stored in our losses object `.data` at the first position.

We are going to update the weights and biases. For this, we're going to use the functionality `torch.no_grad`. In this area, we are now deactivating the auto gradient because now we want to make use of the calculated gradients and assign it to the tensors.

Afterwards, we are zeroing out the gradients so that the next iteration can start. You can see also one of the functions that is having some underline at the end. This is basically just saying that this is some in-place operation and that the gradients are directly affected with this. So we don't have to assign anything like `w.grad` equals to something; it's directly some in-place operation.

Then we are close to the end of this loop. At the end of each epoch, we want to see the loss value, and we can print it directly.

Let's run that and see if it's working. It's super fast because it's a very small dataset. You can see here the losses are decreasing. That is actually what we want to see. And we are done.

---

### Foundations and Core Concepts of PyTorch
#### Module 7
#### Linear Regression from Scratch (Coding, Model Evaluation)

So now we are here to check the results. Let's just print with the F-string notation our weight, and that is accessible with our tensor `w` and the function `item`. We also want to show the bias, and that is stored in `b.item`.

These are the parameters that our model has calculated. If we plot again our graph, we can see here that we have our slope. The slope is -5. You can roughly cross-check if that is making sense. It's easier with the bias. The bias is the intersection with the Y-axis, and the Y-axis would be hit at 36.8. You have to consider that this is not showing all the way; it's starting at 1.5. So the 0 would be here. It's making sense.

Now we can also calculate our predictions. For this, we're just using our independent feature as `X`, multiply that with our weights, and to that we are adding the bias. Then we are detaching it from PyTorch and converting it to a `numpy` array.

Let's take a look at `y_pred`. These are our predictions. Now it's always good to have some visualization for it. So we're going to create some scatter plot here. We can use our values which are stored in `X_list` and in `y_list`. That is our true observations. Then if we put on some line plot for the same `x` value `X_list`, and then we're taking our predictions that we have created, and then we're putting a line through it.

Actually, it's good that you see that this is actually not working. We are ending up with some error. The data must be one-dimensional. Let's see what is happening here. So we have `y_pred`. We can check the shape, and yes, it's two-dimensional. This is something that you will end up seeing a lot. We have to always make sure that the dimensions of your objects are right. One way to solve this is to call the `reshape` function, `-1` to remove one dimension. Then you can call it. Now it's working.

This is now our regression line which is actually coming from our deep learning model. That is actually quite cool. If you compare that to the statistical model that we have here, it's basically the same.

Even if it doesn't look like much, you have already created your very first neural network, and you were able to train it correctly, and it's providing the right values. That is actually awesome.

You can just believe me, but in statistics, it's always better to trust some numbers. So we can just make sure to see that we get the same results by creating some real linear regression model.

In here, this is just some `sklearn` stuff that we're doing here. So we are fitting our `numpy` object `X_np`, and we're passing our target variable. With this, we create our regression model, and then we can extract our slope and offset. The slope is stored in the regression coefficient, and the intercept is stored in the property `intercept_`.

Let's see that. So the true values are, for the slope -5.3 and the intercept is 37.28. Actually, what we have here is coming very close to this, as you can see here. So the weights and the bias are pretty much what the statistical linear regression would also provide.

Then the last part, that is just bonus. We don't have to do that because it's requiring some additional package installation. I have shown you this here at the end. So we can create a graph visualization of the graph that was used for this. For this, you can use the package called `GraphViz`, which can be downloaded under this. But there are also some additional things to be considered. If it doesn't work for you, it's definitely not a problem. I also had to do some changes just to make it work.

In the end, I just want to quickly show you that once everything is installed and we have added it to the path variable, then you can call the function `make_dot` from this package `torchviz`, and you get this representation of our graph. This is the first neural network that you have created. We can see here which of the functions are called, which of them are depending on other functions, and so on, all the way back to the very beginning where we have the slope and the bias.

This was our first neural network that we have created. From now on, we are building up on what we have learned here. We are putting more complexity to it. We are going to improve the performance and the way our code looks. But the general structure is going to be pretty much like what we have seen here in this first lecture.

Thank you very much for staying with me in this coding lecture, and see you in the next one.

---

### Foundations and Core Concepts of PyTorch
#### Module 7
#### Model Class (Coding)

Hello and welcome to this coding lecture. In this coding lecture, we are going to work with the Model class. So far, we have created our model directly from scratch, but now I'm going to show you how you can set up your model in a separate class which you then can easily instantiate and use in the further code.

For this, I have opened the linear regression model plus start script which is just providing some code for loading the required packages. I run this first chunk, and then it's also having the data import ready to start. You're going to see this is the same dataset that we have used in the lectures before. Then we're converting these lists and `numpy` arrays to tensors.

Everything is ready. Now we can create our Model class. For this, we have to type the class, and we have to give a name to this class. Let's say we want to call it `LinearRegressionTorch` class. This one has to inherit from the PyTorch submodule `nn.Module`.

This is the general structure. You have to create this class. The class is inheriting from `nn.Module`. Now we have to specify the `__init__` function which is called when an object is instantiated from this class. We can create this `__init__` function. It needs to have the keyword `self` which is referring to the objects inside the class. Then we can specify two parameters. I will call them `input_size` and `output_size`.

In here, when the class is initialized, we're creating an internal object which is called `linear`. This one is a linear layer coming from the `nn.Module`. This linear class is requiring two parameters. One is for the `in_features` and one is for the `out_features`. In here, we're passing the size of the model. We're telling that there's one node going in and one node going out.

Every model size needs to have a forward function which is provided with one variable typically called `X`. Since our model is so simple, we can just follow this proposal here. We are just calling the linear layer that we have created before and storing the results in an object called `out`. We are returning this one. Later on, when we have more complex models, we will set up more layers in the `__init__` function, and then we stack them together in the forward function.

That's already it. This is our Model class. Now we can specify some parameters. Our input dimension is 1, our output dimension is 1, and we are ready to instantiate an object of this class. Here we just create a new object called `model`, which is instantiated from this class with the parameters we have just specified. We can run that, and our Model class is very perfect.

The next one is to specify a loss function. Since we're working with linear regression, the loss type of mean square error seems to be quite appropriate. We will use this specific function and assign it to the object called `loss_fun`.

Then we need to specify the optimizer. Here we can also specify another parameter called `LR` for learning rate. We specify the value for it. Then we need to specify the optimizer. The optimizer is created from the submodule `optim`. We're going to use Stochastic Gradient Descent. When this function is called, it is requiring some parameters. Here we are passing the model parameters which we can extract with `model.parameters`. We also can specify the learning rate. With this, we are ready to create our training loop.

This time we are also going to save the losses and the slope and the bias in individual variables. These ones are lists which are completely empty at the beginning. We also set one parameter, the number of epochs to 1000. Now we can create a loop for it. We are iterating over 1000 epochs.

At first, we are setting the gradients to zero. We're doing this by calling `optimizer.zero_grad`. Then we have the forward pass. We're passing our observations through the model that we have created to get the predictions. This is exactly the line we want to run for this. We're passing `x`, our data that we have set up here, to the model, and we get as a return `y_pred`.

Now we have predictions, and we also have the true values. We are at the point that we can calculate the loss. For this, we can call our loss function, pass in the predictions and the true values, and we will get back the losses which we store in the object called `loss`.

Now we can update the weights by calling `optimizer.step`. That would be it if we wouldn't want to learn more of the model. But right here, we want to see a bit more into the insight on what is happening. We will get the parameters, and I will show you how we will get them. Then we are going to store them to the losses and to the slope and bias parameters. We can iterate through the parameters. In `model.named_parameters`, we have all the parameters stored, and we'll get back the name and the parameter. We're only going to check the parameters that are having a gradient required.

Only in the case that the name is `linear.weight`, we are now going to append our slope, and we're going to use `param.data.numpy`, and we are going to use the very first object. We are doing something very similar for the bias term. In here, we're going to append the current bias value of the current epoch.

Then we're going to store the losses. Very similar, we're taking our losses objects, and we are appending what is stored in `loss.data` as a float. Now we can print the loss for every 1/100 epoch. We're going to print the epoch and the losses.

Let's run that. You can see here it's not working; the losses are not decreasing. They are constant. Actually, there is a bug in my code, and I found out that I have missed one small line, one very important one—the one in which the gradients are calculated. We spoke before about the automatic gradient calculation which is triggered by calling `loss.backward`. This one was missing. Small line, big impact. This is very important to calculate the gradients.

Let's just clear up everything and run everything again. Now this is the expected result. You can see we have losses which are decreasing with the number of epochs. At some point, they are getting quite stable. This is good.

Now we can, for example, visualize the model training by creating a scatter plot in which we are passing in the number of epochs on the X-axis. Then we are going to plot the losses. This is the whole purpose why we have stored this object. You can see this is a typical representation of how the losses develop. You start with very high losses at the beginning, and then the gradient descent is in action. You can see that you're getting closer and closer to a good result. At some point, the returns are diminishing more and more. At some point, you get a very flat line at the end at which point it's not really worth having more calculations. At that point, you could basically stop the training and work with this kind of model.

We can also do that for the bias terms. In a very similar fashion, we create a scatter plot on which we have a range of the 1000 epochs. Then we are going to plot the bias term. You can see the same. We are starting with a very poor result. We are starting basically at around zero, and then we are getting better and better. We are getting at some point to the statistical result of 36 point whatever. We can also do the same for the slopes. Same story. We're starting out with some poor results and then a very steady trend towards a constant value at some point. Here it's looking like there is still some significant slope. We could even think about having more epochs looking at that graph.

We now can also create our predictions again by calling our model object. In here, we're going to access the data and convert them to `numpy`. We need to remove one dimension to make it work. In the end, with the plot we're going to create, we can create our scatterplot with our X values which are stored in that list and our `y_list`. These are the true values. Now we can put on top the predictions, and we want to be able to differentiate them. So we specify a color. You can see it actually is better if we take a line plot in this regard.

Here we are. We have our observations, the blue dots, and this is our final regression line. This looks already quite good. All right. You have learned how to set up the Model class, and how to create the training for this specific model.

---

### Foundations and Core Concepts of PyTorch
#### Module 7
#### Exercise: Learning Rate and Number of Epochs

Now here comes a small exercise. You've seen we have two hyperparameters here. We have the learning rate and the number of epochs. I want you to play around with these values just to see which values are working out quite well. At which point do you get some stable results? Just play with these values. Perform the training, take a look at the results, and try to find some good working parameter combinations on your own. Then I'm going to show you my solution to that. All right. See you there.

---

### Foundations and Core Concepts of PyTorch
#### Module 7
#### Solution: Learning Rate and Number of Epochs

And here comes my solution. How would you basically detect this? In some later lecture, we are going to deal with hyperparameter tuning in more detail. Then we're also going to set up some grid search. We will have some very structured approach of modifying parameters and testing all of them here. Now I just want you to modify the parameters manually just to see what the output is.

What I'll do is, I'll start with some very large learning rate, and then I can run everything including the losses plot. I run everything above, and here you can see this is definitely too large. This would be the loss plot. It should have some decreasing slope, and the losses should be getting smaller and smaller with every epoch. Here we can see there is no learning at all, and at some point the losses are even exploding. So that is not working at all.

Now, I usually just go in orders of magnitude through it. So we just take a value 10 times smaller than this one, and then we run everything again. This looks much better. You can see a value of 0.01 is already providing some good results. Let's just check the boundaries of this. Let's just get smaller and smaller until we see at some point that this is not really working out. You can see, if we are having a value of 0.001, then 1000 epochs are definitely not enough yet. We have not reached the stable plateau. We can still get some improvement with having more epochs.

If we want to go with this learning rate, we would have to increase the epoch. I won't do that now. But this is what you can read out of this graph, for example. Let's assume that the optimum value is somewhere in between 0.01 and 0.001. Let's just try 0.02 for this and run it again. This looks quite good. Here we have the stable result at the end. 1000 epochs are enough for this.

In this specific example, we could also take a look at the statistical results because we know that we have the optimum values of the bias and the slope. This one should be 37.3 and -5.3. This is our baseline. In this specific example, we know these are the true results, and then we can check basically the bias and the slope. Let's see where we end up. You can see with this value, we didn't reach the perfect result yet. Let's just take a look at the slope.

As you can see, with this value we did not get yet the perfect results. Let's just increase the number of epochs. Now it's running a bit longer, and then we can take a look at the losses. They seem to be quite stable for a long time already. I guess the same holds for the bias and for the slope. Let's just check the values. The bias ends up with 37.2. Pretty close to the statistical result. The slope is at -5.3. Let's consider this to be close enough. You could also modify a bit more with the number of epochs just to see what are the limits. Is there maybe a minimum number of epochs that you would need depending on the learning rate?

Yes, that is basically what I wanted you to explore. How these two different parameters are interacting with each other? The smaller the learning rate, the longer the training is taking, and then you need more epochs to get to a stable result. With this, we have reached the end of this lecture. Thank you very much, and see you in the next video.

---

### Foundations and Core Concepts of PyTorch
#### Module 7
#### Batches (101)

Hello and welcome to this lecture on batches. What are batches? Why is it useful to partition your data into batches? It is actually the easiest to understand if we take a look at the problem.

The problem is that often you have datasets that are very large. You have a huge dataset that you would want to use to train your model. It might be that passing the complete dataset at once to the model is just impossible due to the constraints of your computer. In these cases, it's very good if you partition your data and provide smaller bytes to the model. This is what is called batches.

Here you can see the alternative to the left side. Now we are not taking the complete dataset and passing it to the model, but instead we're passing it in batch one, which is just having a fraction of all data. Then we're taking the second batch, and so on all the way until the last data is provided to the model.

Now you might ask, well, it might be a good idea to partition the data into batches, but what is actually the optimal batch size? Let's talk about that. The batch size is the number of simultaneously provided datasets. There are a few points to consider here. The batch size is defining the speed of the model training and the stability of the learning process.

There are some reasons for using smaller batch sizes. Smaller batch sizes are actually noisier and provide lower generalization errors. Like I said earlier, it's easier to pass a small batch size of training data because then you don't run into constraints with your memory, either CPU or GPU.

Just to provide some best practices, what are typical batch sizes? The typical batch size is somewhere in between 1 and 500. Often you can see that the size is just a multiple of 2, and it has shown that a good default, if you don't know anything else, is to use 32.

That was our short course on batch sizes. Thank you very much, and see you in the next lecture.

---

### Foundations and Core Concepts of PyTorch
#### Module 7
#### Batches (Coding)

Hello and welcome to this coding lecture. In the previous video, you have learned about batches, and this is what we are going to implement now.

Here you can see I have opened this linear regression batches start script, which is basically the code from the Model class implementation. We can just check that everything is working by running everything above the line where we plot the losses. We're working with the same dataset, and we create our linear regression model with PyTorch.

What we're going to do now is, we are iterating through our dataset with a defined batch size. For this, we need to define this parameter at first—the batch size. We're going in smaller batches through the data within every epoch. What this technically means is we have the outer loop for iterating over the number of the epochs. Then in every epoch, we are passing the training data bit by bit to the model. This is what we are going to implement now.

The group is looking like this. We are creating a for loop, and the range is from zero to the length of `X`, or we can say `x.shape` at position 0. This is returning the number of rows of our training data. We're iterating over the number of rows of our dataset. The step size is the `BATCH_SIZE`. Every time we're increasing the value by 2. That is already good.

Now, we have some problem with the indentation. Everything that is inside the for loop needs to be indented. All right, but still this cannot work because still we're passing in here the complete training data. This of course is not correct. Here we need to add the specific location, so we're going from position `i` to position `i` plus the `BATCH_SIZE`. At the first run, we're starting with 0, and then the range is going from 0 to 2. Then `i` is jumping with a value of 2, and then this will return the next pair of data.

Now we have the predictions right. We have to do the same for the true values because this is now having the predictions for two observations and not the complete dataset. Then we need to compare that of course also to the same two locations of the true tensor. Again, we're slicing for position `i` to `i` plus the `BATCH_SIZE`. That should be it.

We can run now everything. Let's just clear the output and run everything. Now you can see here due to the indentation, we can correct that maybe that we are having now the epoch 0 a number of times because every time we are passing only a small fraction of the complete dataset. Finally, we have more of these values. This is just due to the fact that we're adding several values per epoch, but that shouldn't care us too much.

The main point in here is that we successfully implemented the `BATCH_SIZE`. We are now passing the data not as a complete chunk to our model training, but instead we are taking smaller bytes and passing them one by one to the model.

That was basically it. Inside our outer loop for the iteration over the epoch, we have now introduced another inner loop for iterating over the batches. Then all we had to do is, we had to pick the right positions of the data that we are passing in the forward pass, and we had to adapt in the loss function the position of the true tensor. That is everything that we needed to implement to make this work and to implement successfully the `BATCH_SIZE`.

That is everything that I wanted you to learn in this lecture. Thank you very much, and see you in the next video.

---

### Foundations and Core Concepts of PyTorch
#### Module 7
#### Datasets and Dataloaders (101)

Hello and welcome to this lecture on Datasets and Dataloaders. Let's start with the question: Why is it a good idea to create these specific classes?

The idea behind is that ideally your model training should be separated from your data processing, because then you can change something at the end of data processing and your modeling part can stay exactly the same as before. In the end, your code is having some increased readability and you have a very high modularity. It's just better code, and it's making your life easier later on.

For this, there are two specific classes implemented in PyTorch, and I will show you how to use them, which are `Dataset` and `Dataloader`. The `Dataset` is an interface to some preloaded datasets that you can use. But mainly we are going to use it as an interface to our own, to our custom datasets.

Here the `Dataset` is storing the samples and the labels, and then the `Dataloader` is taking over. This is some iterable wrapped around `Dataset`, and that is basically loading then the data.

Let's take a look at how you can work with custom datasets. You can just implement your own class. You can see it here at the right-hand side. First of all, you have to load the classes `Dataset` and `Dataloader`. Then you can set up your own `Dataset` class, which is inheriting from `Dataset`. It's important that your `Dataset` class has the implementations for three different functions.

The first one is the `__init__` function. This is the function that is run once during the instantiation process of creating an object. Then you have a `__len__` function, which is just returning the number of samples that are available. Then you have to have some implementation of the `__getitem__` function. This one is loading samples from the dataset, preprocesses them in the way that you have defined, and returns them for a given index.

Here at the right-hand side, you can see our own implementation as we will develop it together in our coding session. Here we are passing in our independent and dependent dataset `X` and `y`. We are creating the internal objects `X` and `Y`. Then the `__len__` function is just returning the length of your dataset. The `__getitem__` function is provided with an index, and then it's returning the independent and dependent feature at the position of the index.

That's it on the `Dataset`. Then the next part is the `Dataloader`. The `Dataloader` basically iterates through the dataset and it's returning batches of your data. We have spoken about batches before. Depending on the `BATCH_SIZE` which was defined, now the `Dataloader` is returning a number of different datasets.

It has a lot of different features which are coming in very handy. For example, you can just set a flag so that all the data are shuffled, and they're not taken from top to bottom but randomly sampled from the dataset. There are also some more advanced sampling strategies. There's a lot to discover with this class as well.

That's it for our introduction into `Dataset` and `Dataloader`. Now in our upcoming coding session, we're going to see how we can actually implement that. Thank you very much for joining, and see you there.

---

### Foundations and Core Concepts of PyTorch
#### Module 7
#### Datasets and Dataloaders (Coding)

Hello and welcome to this coding lecture on Datasets and Dataloaders. You can see here I have loaded this linear regression dataset dataloader start script. This basically reflects the code that we have created with the batches before.

Now what we're going to do is basically after we have created our independent and dependent features, we are going to create a new chunk on which we are going to create at first a `Dataset` and then the `Dataloader`.

For this, we are going to create our `LinearRegressionDataset`. This one inherits from the `Dataset` class. The `Dataset` class needs to be imported, of course. So we need to import it from `torch.utils.data`. From this submodule, we are importing the `Dataset` and the `Dataloader`.

We can run this, maybe all chunks until the one where we're currently creating the code. Now we can create our `Dataset` class. As you have seen before, the `Dataset` class requires some functions. The first one is the `__init__` function in which we are going to pass in the independent feature and the dependent feature. Then the internal objects `x` and `y` are populated with the data that was passed in during the instantiation of this object.

Then we need to set up another function for length. Everything that this function is doing is it's returning the length of the complete dataset. Then we need some implementation of the `__getitem__` function, which is accessing the internal objects with `self`. Then you can pass in some index, and it is returning the independent features at this position of the index and the dependent features.

This is everything that you need to do to set up successfully the `Dataset`. Then we can specify the `Dataloader`, and we can directly assign it to the object that is called `train_loader`. For this, all we have to do is call the `Dataloader` function, and we need to pass some parameters to this. The first one is the `Dataset`, and the `Dataset` is basically created with the `Dataset` class that you can see here.

In here, the objects `X_np`, so the `numpy` version of the independent features, and the `numpy` array of the dependent feature are passed into this `Dataset` class that we have just created, and we pass this to the `Dataloader`. This is the required parameter. We also need to specify another parameter which is the `batch_size`. We just leave it with a size of 2 for now. You can see here there are many more parameters. You can do much more with this. For this introduction, we are just leaving it with the required parameters, with the `Dataset` and with the `batch_size`.

Now we can run this. Now we have some implementation of the `Dataset` and the `Dataloader`. The modeling class is not touched; everything is kept unchanged. We are also going to keep the loss function and the learning rate and the optimizer. The next change is coming in our training groups.

You can see how we have implemented the batch sizes before. We had to make sure that we are picking the exact right positions of the independent and dependent feature. We need to make sure with the slicing that we get the right starting and end position, and we have to make sure to apply this at different points. Actually, one purpose of the `Dataloader` is that it is making our life a bit easier in this regard. We are going to work on this inner loop and also how the data that we have here is modified.

Before we go there, we will just have a small x course. This one is going to check the `train_loader` and what this returns. What we're going to do is basically we're going to make use of the `enumerate` function. In here, we are passing the `train_loader`, and what we get back is some iteration of this training data that we have.

We can put this into a for loop, and then we get back two different returns. One is the position at which we currently are, and the other one is data. We can run it like this. Then we can print, let's just print which batch we have here and then which data we can see.

We can see here the zeroth batch. Then we can see here the data is actually a list of tensors. Let's try to understand this in more detail because here in the data, we have the first object, and that one is having the size of 2—that is the `batch_size`—and then the number of features. Then we have the second object with the same size. What this is actually referring to is our independent feature and the dependent feature. What we could also do is we could also say that we get back here a couple of the independent feature and the dependent feature. Then we could print these two different objects.

This is basically how we have to understand the return values of the enumerated `train_loader`. This is what we're going to make use now in our training loop. First of all, we are going to work on this inner loop definition, and we can make use of what we have just learned. We have `X` and `y`, and we have in `enumerate train_loader`.

Now we have changed that, but that also means we don't need any more these specific slices that we use here with the batch positions. We also have to modify this `y` value because now we are taking the updated `y` value that is defined here. Now we are passing in our independent feature in the forward pass. We get back to the predictions. Then we make use of these predictions together with the true independent features during the loss calculation. The rest is basically staying the same.

Now we can run everything, and we should see that there is some decreasing development of the different losses. It's looking like to be very stable already up to a few 100 epochs. You can see here. It's converging quite nicely. This is then providing the same results that we had before in the coding lecture.

The main takeaway of this coding lecture is basically that you have learned how to set up the `Dataset` class with your custom data. You have seen that you need to implement these three different functions: the `__init__` function in which you pass in the data that you want to process, then the `__len__` function which is returning the length of the object, and then the `__getitem__` function which is returning based on an index which has provided the independent and dependent feature.

The `train_loader` is defining in very much detail how the data is going to be provided during the learning. The two required parameters are here, the `Dataset` that you need to pass in, and the `batch_size`. There is much more that you can explore. You can, for example, enable the shuffling so that the data is sampled from the training data. There is much more to explore here.

Finally, you have seen how you can include this in the training loop. At which positions you have to modify the code to make it work and to pass in the return of the `train_loader` into the training loop. Then the training is performed accordingly, and you're done.

That's it what you should learn about `Dataset` and `Dataloader`. Thank you very much for watching, and see you in the next lecture.

---

### Foundations and Core Concepts of PyTorch
#### Module 7
#### Saving and Loading Models (101)

Hello and welcome to this lecture on model saving and loading. First of all, why do we need that? Well, you know that training some deep learning model might take hours or even days or weeks. You definitely need to find a way to save the model in the end because you cannot afford to just retrain it every time you need it.

For this, there are of course some functions implemented in PyTorch. The first one is `torch.save`. This one is saving your model. Here the first recommendation is actually to only save the `state_dict`.

You can ask yourself why is it not good to save the complete model? Well, it's because if you would save the complete model, which is also possible, then there are some exact directory structures saved. Some paths are saved to the file with the classes. You might end up in a situation where your code easily breaks when you try to load the model on a different computer, for example. Due to these reasons, it's the recommended way to create a new instance of your model once you have trained the model and you want to load it, and then just to load the state dictionary into your instance of the model.

The `state_dict` is basically a dictionary with all the layers and the corresponding parameters. Here only the learnable parameters and registered buffers like batchnorm are saved. At the bottom, you can see how to use it. You can just call `torch.save()`, and then reference to your model. In case it's called `model`, then it's `model.state_dict`, and then you have to provide a `PATH` to save it.

The corresponding loading function is then called `torch.load_state_dict`. This is like the name you would guess, loading the state dictionary. You can see what I have just explained. If you want to load it, you have to create an instance again of your model class, and then you have to load the state dictionary with this functionality.

Since you don't want to retrain the model, you can put the model into evaluation mode with `model.eval`.

Just for completeness, I want to tell you that there is this way of saving the complete model, and then the corresponding loading function would be called `torch.load`. But as I've said before, this is not the recommended way to do.

Good. That's it. In the upcoming coding lecture, we will see how we can easily implement this. Thank you, and see you there.

---

### Foundations and Core Concepts of PyTorch
#### Module 7
#### Saving and Loading Models (Coding)

Hello and welcome to this coding lecture on model saving and loading. You can see here, I've loaded the start script for model saving and loading. You can just jump to the bottom and run everything above this line to set up the model and perform the model training. This is what we're going to start with. We have a model, and it has already trained parameters. Now you're going to find out how we can actually save this model and load this later on.

First of all, we have spoken in the theory lecture about the state dictionary, and I just want to show you this. You can access this by calling the method `state_dict` on the model. You can see here this is an ordered dictionary where you have at first the linear weights and then the linear bias.

If you want to save the model state, then all you have to do is call `torch.save`, and you have to tell what to save. That is actually the `model.state_dict`. The second parameter is the file which you want to use for saving this. You can call it like here `model_state_dict.pth` for PyTorch.

If we run that, you can see that there is a `model_state_dict` saved in the current working folder. Now if you want to load this model, all you have to do is you have to at first create an instance of your Model class. That is our `LinearRegressionTorch` class with the input dimension and the output dimension. This is initialized with some random weights. Now you have to load the `state_dict`.

You have to run `model.load_state_dict`, with `torch.load`. In here, you're passing the file that we have just saved. That would be the opposite direction for loading a saved model. Then you can just check the `model.state_dict`, that everything was loaded correctly.

With this, you could move forward. You could just share the trained model, or just use it at some later point. Just to recap what you have learned: This lecture was focusing on how you can save the model, and how you can later on load it again. In here, you have seen how you can access the state dictionary, and then how you have to set it up so that you're saving this state dictionary in a PyTorch file. Later on, if you want to load the model, you have seen that you have to at first create an instance of your Model class. Then you have to populate the state dictionary with the parameters that were saved before in this specific file.

That's all you have to know about saving and loading the model. With this, we have reached the end of this lecture. Thank you very much, and see you in the next one.

---

### Foundations and Core Concepts of PyTorch
#### Module 7
#### Model Training (101)

Hello and welcome to this lecture on PyTorch model training. There is actually a strong need to have this lecture because the process is, to be honest, quite confusing in PyTorch. We have so many different objects, and it's not really clear which objects are impacted at which point, and by what. Let's speak about it.

What do we have? We have the model, we have the optimizer, we have a loss function, we have a gradient update, then we have the backward function of our loss function, and we have the `optimizer.step`. Then there are the model parameters. So many objects are floating around, and it's not really clear which objects are changed and impacted at which point, and by what. It's really important to understand how everything interacts.

There is a very good article I would like to make yourself aware of. You can find the link down here, and it's really worth reading. It helped me a lot to understand these things because they're not so simple.

Basically, we're having these objects. We're having the model which is based on our submodule `torch.nn`. This one is including the layers, the activation functions, and so on. Then we have the optimizer. Actually, here we have the first issue because the model parameters need to be shared with the optimizer, and we don't know how. Then we have the loss function. That one is used to calculate the losses. All of these objects are basically feeding into the model parameters and the gradients.

You can see the classical training loop. This is a very typical structure that you can see a lot in our coding lectures. We are having an outer loop for iterating over the different epochs. Then we have the inner loop in which we are iterating over the batches. At first, we have the optimization. We are zeroing the gradients out. We will talk about that more in the later lectures. Then we have our forward pass in which we are calculating the predictions. Then we can take these predictions together with the true values and the loss function so that we get, taking all of this into account, the losses.

Then we have the backpropagation and the updating of the weights by calling `optimizer.step`. This is basically the structure that you will see over and over again in the recording.

Now we want to talk about all these little details in here. The first one is `optimizer.zero_grad`. This one is clearing the gradients. For this, we have to know that the optimizer is accumulating the gradients. For every new pass, every new forward pass or backward pass, the gradients need to be deleted.

Then the question is still, why is actually the optimizer holding the gradient and not the model itself? Let's talk about that later.

We have our forward pass. In this step, the predictions are calculated. That one is very straightforward. We're taking data, putting it through the model, and get back some predictions.

Similarly, for the loss calculation, also pretty straightforward. We have the true values, we have our predictions, and we have a loss function. All of that together is returning some actual calculated loss.

Then we have the gradient calculation. In this step, when we call `loss.backward`, this loss function object is calculating the gradients for all the nodes. You can see here some example. We want to calculate the gradient of the `weight1`. That is basically the partial derivative of the loss function with respect to this `weight1`.

You have to know that the gradients are changed directly. They are changed inplace, and implicitly the model layers are used and the tensor gradients are updated.

Then we have the weight update. At this point, you can assume that the gradients are known. Now we have to update the weights. For this, you can use the `step` function of the optimizer. That is the purpose of this function. In this step, the model parameters are updated, although the model was never caught in this step. Also here there is some inplace functionality, and things are changed in the backend, which is not really clear how it's done.

Why is it under weight? What are the advantages and disadvantages? The advantage is that with the way the developers have set up PyTorch, they have created some very flexible and composable API which can be used, not just for deep learning, but also for other computational graphs.

The disadvantage is a bit like what we have touched on in this lecture. The model, the loss, the optimizer—they change the parameter state, and it's not really clear who is the owner at which step.

I hope that this made it a bit clearer, and you're now less confused than at the beginning of this lecture. Thank you, and see you in the next one.

---

### Foundations and Core Concepts of PyTorch
#### Module 7
#### Hyperparameter Tuning (101)

Hello and welcome to this lecture on Hyperparameter tuning. There are some reasons for tuning the parameters.

The first one is covering the training and the inference time. Depending on the parameters you're choosing, you could end up with some very long training time or some very long inference time.

Another reason might be that you just want to improve your results because the hyperparameters are very much impacting the performance.

The last one is touching on the convergence because if you have wrong parameters, it might be that you end up with exploding or vanishing gradients, and your model is just not being able to learn correctly.

Let's assume there is a need to perform hyperparameter tuning. Then the question is how can you actually do that? The intuition is that you want to track multiple combinations of parameters, and then in the end just pick the best one.

There are also certain packages that you can use. Here I have just named some, like `RayTune`, `Optuna`, and `Skorch`. The last one is what we're going to use in our coding lecture.

There are a bunch of different hyperparameters. I tried to categorize them into different classes. The first class would be the network topology. If you want to change the topology of your network, you could change the number of nodes, you could change the layer types, the activation functions—everything that is defined in your models.

Then there are some other network objects like the loss function or the optimizer which you could change or which have some parameters themselves that could be optimized.

During the model training, there are also some other parameters like the learning rate, the `batch_size` we spoke about earlier, or the number of epochs. All of these could be modified and much more.

Let's touch on some of them. Let's recap the batch size. Here you can see the problem. You could pass your dataset completely into a model, which might not be possible. Due to that, it might be a good idea to have a small batch size. Using a small number of datasets and putting these small bytes to the model, or using some larger size and then putting these larger bytes to the model.

Here are some advantages and disadvantages. For the GPU utilization, if you have a GPU, then it's advantageous to use some larger size. Of course, the number of iterations is going up if you have very small batch sizes. Smaller sizes have also shown to improve the stability of your training. You can see already there are some reasons to use smaller values, and other reasons to use larger values. It might be complicated. The best practice here is to use a batch size of 32 to start with.

The next hyperparameter I want to touch on is the number of epochs. You can see you have two different options. You could have a low number of epochs or a very high number of epochs. This of course is impacting your training time because the higher the number of epochs, the longer the training time. On the inference time, that doesn't have an impact, but it does have also an impact on the model performance.

The model performance is poor if your number of epochs is too small. Typically, it's better to have a larger number of epochs. But there is also some limit because at some point it just doesn't make sense to increase the number of epochs anymore. Also, you might end up with some stability problems if you are having a too large number of epochs.

The next point is covering the number of hidden layers. There could be just a very small number of hidden layers. Like you can see here, we have an input layer, a hidden layer, and an output layer. Here is the opposite side where you have a number of different hidden layers. For the ability to learn complex patterns, it's necessary and essential to have many hidden layers. For this, it's better to have more layers. But then of course, it's having an impact on the training time. It takes of course longer if you have more layers, and thus more nodes. Also, it's having the same impact on the inference time. When you deploy your model, it's also taking longer if you have more layers included. There's also this risk of overfitting. If your model is getting too complex, it might be that the model is just memorizing the observations, and this might reduce the generalization of your model.

The next hyperparameter I want to touch on is the nodes within a layer. You could have a hidden layer with just a few nodes, but you could also have the opposite—a hidden layer with many nodes. It's similar to before. We have the same pattern here. If you need to have a model which is able to learn complex patterns, then it's necessary to have more nodes in your hidden layers. If you have more layers, then this is good in the case that you need to have the ability to learn complex patterns. But it has a negative impact on the training time and the inference time because these ones are getting up. Also, having fewer nodes might be beneficial because the risk of overfitting is decreased. This first point and the last point—these are always creating some tension. If your model is too simple for the problem, then it's not able to learn the complex patterns. But the opposite might be that your model is too complex for the problem, and then you are risking to get overfitting.

How can you actually create some structured search? There are different approaches. The first one is called grid search. In this approach, you need to at first define your search space. You need to define a set of parameters with some limiting values, and then you are basically evaluating all possible combinations. In a case where we're saying we want to make some grid search based on the learning rate and the batch size, you can see we have two different values for the learning rate and three for the batch size. Then you can expand the metrics and create all the different runs. Basically, all possible combinations of both parameters. This approach of grid search is very well for checking well-known parameters.

There is also random search. In this approach, you would pick randomly a point from your configuration space, and that might be better for the discovery of well-working parameter combinations.

I want to make you aware of one package that I found to work very well. That is also what we're going to use in our coding section. That is called `scorch`. That's a scikit-learn compatible neural network library that wraps PyTorch. You can see where to find the repo. This can be integrated into your sklearn pipeline. It also provides some grid search functionality, and this is actually what we are going to code now in the upcoming coding lecture. Thank you, and see you there.

---

### Foundations and Core Concepts of PyTorch
#### Module 7
#### Hyperparameter Tuning (Coding)

Hello and welcome to this coding lecture on hyperparameter tuning. I have opened here the start script for the hyperparameter tuning. The sample code that I'm going to provide here is covering the package loading, the data import. You can see here we have set up the `Dataset` class and the `Dataloader` and the modeling class together with the loss function and the learning rate and the optimizer. Many things are already provided.

What we now are going to implement is something that is called grid search. In the theory lecture, you have seen what grid search is meaning. We are going to set up some parameters, which are then combined, and every combination of these parameters is going to be evaluated and compared to each other. Finally, the best parameter combination is used.

For this, I'm going to use a package that is called `scorch`. You can find it here at this GitHub page. This is basically a scikit-learn compatible neural network library that wraps PyTorch. First of all, we have to go to the top of this file and import the necessary functions of this specific package. From `scorch`, we are importing the `NeuralNetRegressor`.

Let's just scroll to the bottom of the script and then run everything above. Now it should look like this. Everything was loaded into the workspace, and now we are ready to set up our model. The first thing is that we're going to make use of this `NeuralNetRegressor`. In here, this is requiring some parameters. The first one is the `LinearRegressionTorch` model class that we have set up. The second one is a parameter, the `max_epochs` that we can set. Then we can also specify the `learning_rate`. We can also specify a parameter called `iterator_train_shuffle`, and we can set that to `True`.

You can also check out the help function to find out what other functionality this is providing. Let's just see if this works. This looks good. Then the next point is to set some parameters. The other parameters that we're going to set are `train_split` equals `False`. We're not splitting our data according to some `train_test_split` ratio. The other parameter that is going to be our `verbose` that we set to 0.

Now we're setting up our dictionary of the parameters that we're going to modify. The first one is the `learning_rate`. Let's just use a smaller number of learning rates because it's just for teaching purposes. We are going to take three different learning rates that are going to be evaluated. We provide this list, and we're going to set up another parameter called `max_epochs`, which is also having values as a list. We're having a range of, let's start with 10 epochs, then have 200, then 500 epochs.

Now we have to set this up in our grid search. For this, we can call the `GridSearchCV` function that we've loaded. What this is needing is at first the estimator, the network that we have set up; then the `param_grid`, this is our dictionary of the parameters that we're going to modify; and then it wants to find out the best parameter combination, so we need to specify some scoring. Since we are working with a regression problem, we can make use of the scoring parameter `R squared`. We are using three different runs for the cross-validation. To see a bit more, we can specify the verbosity to be 2.

The next point is that we need to pass our data to the `fit` method. In here, like usually in scale, we are putting in the independent feature and the dependent feature. Finally, once everything was evaluated, we can put into a print statement the best score that is scored in the property `best_score_` of `gs`. Then we can also specify the `best_params`, the best parameters which are stored in `gs.best_params`.

Now we have set up everything. Let's perform now the grid search. You can see here it's running. You can see which parameters were modified, and how long it took. Then we can see there is a small typo. This one got an unexpected argument, `iteration_train_shuffle`. We can check that out.

You can see here there is a typo. It actually needs to be called `iterator_train_shuffle`. Now it's taking a bit longer, and eventually we're going to have some results. Finally, we are done. You can see here the best score. What is more interesting for us, this is the best parameter combination, so based on these two different parameters that were combined. In total, we had three parameters for learning rate and three for `max_epochs`. Nine different parameter combinations were checked. Out of all these parameter sets, these two parameter combinations are the best ones.

If you have more parameters, you can just add them to this dictionary, and then the larger grid search is performed.

All right. In this lecture, you have learned how you can perform some grid search to find the best parameter combination of your hyperparameter tuning. You have seen how we can implement this with the `scorch` package. You have seen how you have to set up everything to perform this grid search together with cross-validation, and how to extract the best score in the end and how to get the best parameter combination.

With that, we have reached the end of this lecture. Thank you very much, and see you in the next one.