In this module, we will focus on evaluating machine learning models. You will learn about underfitting and overfitting, and how to mitigate these issues. The module will also cover the train-test split method and its importance in model evaluation, along with various resampling techniques to manage imbalanced datasets effectively.
Learning Objectives
Identify and address issues related to underfitting and overfitting.
Implement the train-test split method for model evaluation.
Utilize resampling techniques to handle imbalanced datasets.


Underfitting Overfitting (101)

Hello and welcome to this lecture. In this lecture I will speak about some very important problems that you will often encounter in your data analysis, underfitting and overfitting.
Play video starting at ::14 and follow transcript0:14
Here, you see an example for regression.
Play video starting at ::19 and follow transcript0:19
The relationship between x and y is not linear, but rather following a quadratic shape,
Play video starting at ::25 and follow transcript0:25
so it is not surprising that a linear relationship is not a good fit for the data as you can see at the left subplot. The model is too simple for the given complexity. The green points represent the actual observations, and the blue line represents the model that aims at predicting y for a given x.
Play video starting at ::44 and follow transcript0:44
The vertical difference between actual measurement data and predicted values are pretty high. This is shown temporarily for some points indicated with the red lines. This means there is a high bias of the model. Underfitting is closely related to high bias.
Play video starting at :1:4 and follow transcript1:04
At the right side nearly all points of the blue line are covered by the model. For these specific points the predictions of the model will be nearly perfect. You will hardly find any error indicated as a red line for the data. The model was trained on this is due to the fact that all noise which is related to deriving these measurement points was taken into the model as well. So this model will generalize pretty poorly on new data, and it will not work very well in real life situations.
Play video starting at :1:36 and follow transcript1:36
Now take a look at how underfitting and overfitting are reflected in classification problems. We have two independent variables X1 and X2, which are used to predict class membership. The actual classes are represented as green or red color points.
Play video starting at :1:57 and follow transcript1:57
At the left subplot we see a linear decision boundary. Everything below the blue line will be predicted as green points, everything above the blue line will be predicted as red points. This linear decision boundary leads to a significant misclassification rate. Points are predicted as red points where they should be predicted as green points, and vice versa.
Play video starting at :2:21 and follow transcript2:21
This is an example of underfitting the data, and there is a high bias of the model.
Play video starting at :2:30 and follow transcript2:30
At the right subplot you can see a very complex decision boundary. It is actually much too complex for the given problem. Based on this data the model provides a perfect prediction. Some of the points within the red area are measured as green points. It is rather likely that this is coming from some measurement error than being actual properties of the data.
Play video starting at :2:52 and follow transcript2:52
Measurement errors are some source of noise.
Play video starting at :2:56 and follow transcript2:56
It could be avoided to cover noise in the model. Otherwise, the model will try to predict the noise, which by definition cannot be modeled. So here you can see an example of overfitting, and this model is having a high variance.
Play video starting at :3:15 and follow transcript3:15
These are very good fitted models for the given problems of regression and classification. At the left subplot you can see the result for the regression model. Here a quadratic model fits the database, at the right subplot a curve shape also fits the database.
Play video starting at :3:35 and follow transcript3:35
We have seen the term bias invariances before, but what is it exactly? You can see the dart targets,
Play video starting at :3:42 and follow transcript3:42
four different players aim to throw five dart arrows as best as possible to the target.
Play video starting at :3:48 and follow transcript3:48
The result of player A, you can see at the top left
Play video starting at :3:55 and follow transcript3:55
all arrows are close to each other, so the variance is small, but they all have an offset, which means they have a high bias.
Play video starting at :4:4 and follow transcript4:04
Now it's player B's turn. He throws and all arrows are offseted and not on the target. You could also say his play has a high variance and a high bias.
Play video starting at :4:19 and follow transcript4:19
Now player C throws. He is very well centered, which means the bias is low, but there is not enough stability in his plane, so the variation of the dart arrows is large, which means the variance is high.
Play video starting at :4:34 and follow transcript4:34
Now, player D is next,
Play video starting at :4:36 and follow transcript4:36
and he or she is the actual champion of this tournament. All arrows are centered very well and hitting bull's eye, The plane is having a low variance and a low bias.
Play video starting at :4:50 and follow transcript4:50
So here are our axis. Horizontally we can see Variance axis ranging from Low to High; and vertically we see the Bias axis which is also ranging from Low to High. Clearly the best result is low variance and low bias. But often you can't achieve both at the same time. So the question is how to trade off bias and variance.
Play video starting at :5:18 and follow transcript5:18
So what does bias and variance mean in the scope of machine learning models? In our machine environment,
Play video starting at :5:26 and follow transcript5:26
bias is the difference between prediction and actual values of the training data. It is a measure for how well a model is fitted to the training data. A high bias means that the model is not very well fitted. And the performance measure like R squared is comparably low. But at the same time, usually the variance is low. An example for high bias and low variance is if you have a relatively low R squared for the training data, but the validation or testing data is not behaving much worse. Its R squared is in the same order. You can see that the red arrow indicating the difference in performance
Play video starting at :6:11 and follow transcript6:11
of the model on training and validation data is quite short. Your model might be underfitted.
Play video starting at :6:18 and follow transcript6:18
Let's come to variance.
Play video starting at :6:22 and follow transcript6:22
Variance is the difference between prediction of training data and the prediction of validation data. Here performance on training and validation data is compared, you also want to avoid to develop a model that has low bias and a high variance. In the example you see that the model has a very high R squared for the training data. So it is nearly perfectly fitted to the training data. But when this model is applied to the validation data, its performance drops dramatically, the red arrow is quite large and your model is overfitted.
Play video starting at :6:58 and follow transcript6:58
So typically, if you add more parameters to a model or if you use a more complex model from the start, the model complexity increases which leads to a reduced bias but increased variance. The opposite also holds true, using a less complex model will lower your model complexity, but it will increase your bias and decrease your variance.
Play video starting at :7:23 and follow transcript7:23
If you have a high bias, your model learns very fast. It is easy to understand also due to the low number of parameters that need to be fitted, but the model won't perform very well on complex problems. We will see an underfitting of the model.
Play video starting at :7:40 and follow transcript7:40
At the right plot you can see some Error Metric.
Play video starting at :7:44 and follow transcript7:44
It could be, would mean to get arrow for a regression problem or something else. This is plotted over the training data sets.
Play video starting at :7:52 and follow transcript7:52
If you have a small training dataset, you have a very small error based on your training data. But a very high error based on the test data
Play video starting at :8:2 and follow transcript8:02
with increasing size of the training data. The Error Metric increases on the training data but decreases on the test data, and with very large training data the difference in performance between training and test data diminishes. But the Error Metric will be significantly higher compared to the best performance indicated as red line.
Play video starting at :8:25 and follow transcript8:25
If you have a high variance, the Error Metric based on the training data will be extremely small for a very small dataset. This is because all points can be fitted perfectly to the model. And if you add more and more data to the training, the model cannot incorporate all of it into the modeling. There is more and more risk classification. So the error metric increases, but it will always be below the best performance.
Play video starting at :8:52 and follow transcript8:52
The opposite holds true to the test data with a small training dataset size, Error Metric based on test data is very high. It decreases with more and more training data used for the modeling, but still its error will be significantly above the best performance.
Play video starting at :9:10 and follow transcript9:10
There are some algorithms that have low variance like linear regression, linear discriminate analysis, LDA,
Play video starting at :9:17 and follow transcript9:17
or logistic regression.
Play video starting at :9:20 and follow transcript9:20
Some algorithms that usually are combined to high variance are decision trees, K-Nearest Neighbors or Support Vector Machines.
Play video starting at :9:31 and follow transcript9:31
They train much more parameters, and you need to ensure that they don't overfit, which means they provide very good training performance but very poor validation performance because they might not generalize very well.
Play video starting at :9:50 and follow transcript9:50
So you need to find the right balance. This balance is often called bias variance trade-off. The goal is to have a reasonably low bias and low variance. Minimizing both at the same time is not possible because both have opposite directions.
Play video starting at :10:8 and follow transcript10:08
You can see this in the right graph. An error metric is plotted over the model complexity. With increasing model complexity, your bias decreases, see the red line; but the variance increases, take a look at the green line.
Play video starting at :10:25 and follow transcript10:25
What you want to do, is to minimize the overall error, which is the sum of the variance error plus the bias error. And the overall error is represented by the orange line.
Play video starting at :10:37 and follow transcript10:37
It is something like a quadratic function with a clear minimum. And your task is to find this optimum point of model complexity for which the overall error is minimized.
Play video starting at :10:49 and follow transcript10:49
Linear machine learning algorithms typically have high bias and low variance. Non-linear machine learning algorithms typically have low bias and high variance.
Play video starting at :11:3 and follow transcript11:03
But for these you can hyper tune many parameters to find some optimum.
Play video starting at :11:8 and follow transcript11:08
When we cover these algorithms, we will find out how to do this.
Play video starting at :11:12 and follow transcript11:12
So that's it for this lecturer. Thank you very much, and see you in the next one.

Train Test Split (101)

0:01
Hello and welcome to this lecture. In this lecture I will speak about splitting of data, how this is done, and why this is an important step of the workflow.
Play video starting at ::12 and follow transcript0:12
The method of splitting data is called train, validation, test, split. The idea is to use only a fraction of the data for the training of the model, and some other parts are reserved for validating the model and testing the final model.
Play video starting at ::28 and follow transcript0:28
So training data is data that is sampled used to fit the model,
Play video starting at ::36 and follow transcript0:36
validation data is data that is sampled to evaluate a model, a potential model that is fitted based on training data is fine tuned with validation data. The model occasionally sees the validation data but if it is only happening a few times, then the model won't be able to learn from these occasional events.
Play video starting at ::58 and follow transcript0:58
The model might be affected. But if at all only indirectly
Play video starting at :1:4 and follow transcript1:04
test data is sampled to provide an unbiased evaluation of the final model. This means you trained a model based on training data, and evaluated it based on validation data. And finally, you want to make sure that everything works out. So you perform a last validation with test data. This splitting of train, validation, test is considered the gold standard. The final model is checked with test data only once, and it should have the same distribution as the validation data.
Play video starting at :1:38 and follow transcript1:38
Test data can be used for evaluating competing models in competitions, for example like Hegel. And often you will see that there is only a split of training and validation data, and no test data is held back.
Play video starting at :1:55 and follow transcript1:55
But how do you define the count of data for the three groups? Well, that depends on two different parameters. One, the total number of samples; and two, the actual model. Some models require more training data than others. The validation data has to be big enough to detect differences between different models. And as you know, models have different complexities, and some models only have very few hyperparameters that commit,
Play video starting at :2:26 and follow transcript2:26
and these models will be easier to validate and require less data for validation.
Play video starting at :2:32 and follow transcript2:32
Other models with many hyperparameters will be harder to validate, and thus require more validation data.
Play video starting at :2:42 and follow transcript2:42
In the next lecture, this idea will be shown interactively, and you can check it out yourself, so stay tuned.
Play video starting at :2:50 and follow transcript2:50
Thats it for this lecture. Thank you very much, and see you in the next one.

Foundations and Core Concepts of PyTorch
Module 4
Resampling Techniques (101)

0:01
Hello and welcome to this lecture. In this lecture I will present resampling techniques to you.
Play video starting at ::9 and follow transcript0:09
Why are resampling techniques needed? We have seen that we can split our data into training, validation, and testing data; or more simply only into training and validation data.
Play video starting at ::22 and follow transcript0:22
Well, if you reserve all data for training, there is nothing left for validation.
Play video starting at ::28 and follow transcript0:28
But if you reserve some data for validation, this data cannot be used for training the model anymore.
Play video starting at ::35 and follow transcript0:35
So the underlying problem is a target problem.
Play video starting at ::40 and follow transcript0:40
The best training performance is reached if as much data as possible is used for fitting the model.
Play video starting at ::47 and follow transcript0:47
The same holds for validation. If you want to have stable validation results, as much data as possible should be reserved for validation. Maximizing both is just not possible. That is the main problem which can be overcome with resampling techniques.
Play video starting at :1:6 and follow transcript1:06
The solution is called K-fold Cross-Validation, and it worked quite simple. I will explain it based on the shown example.
Play video starting at :1:15 and follow transcript1:15
So you split your data into training and validation.
Play video starting at :1:20 and follow transcript1:20
There are k different sets. In our example, 5 different partitions and 5 different models are created.
Play video starting at :1:30 and follow transcript1:30
And as you can see in the first partitioning, the 1st 20% of the data, in our case the 1st 100 observations are reserved for validation and the residual 80% is reserved for training.
Play video starting at :1:45 and follow transcript1:45
Then the first model is fitted based on the training data and model performance is checked based on this validation data.
Play video starting at :1:54 and follow transcript1:54
And now this procedure is repeated.
Play video starting at :1:58 and follow transcript1:58
Now, a different chunk of the data is reserved for validation. All observations between 101 and 200 are reserved for validation. And the residual data again is used for training this time of model two. And again, model performance is analyzed based on this set of validation data.
Play video starting at :2:19 and follow transcript2:19
This process is repeated a few times more until we have five models fitted and five different model performance results.
Play video starting at :2:28 and follow transcript2:28
This is then called five-fold Cross-Validation.
Play video starting at :2:32 and follow transcript2:32
Typically something like 10-fold Cross-Validation is used.
Play video starting at :2:37 and follow transcript2:37
You also would not use data and sequences like shown here because this would assume that, for example the target variable is randomly distributed over all data.
Play video starting at :2:48 and follow transcript2:48
I only did this here to make my point, and for having a simpler to understand visualization. In reality data is chosen rather randomly.
Play video starting at :2:59 and follow transcript2:59
So what comes next? Assume we have chosen five-fold Cross-Validation, and received the shown R2 results for each of these models based on validation data. Our results are taken and averaged, and the overall model performance is 0.27 in our example.
Play video starting at :3:20 and follow transcript3:20
The best and most stable results are achieved if you create as many folds as there are data points. You always hold one observation back for validation, and create the model based on all data except for this one observation, this is called Leave One Out Cross Validation. So why not always use this approach?
Play video starting at :3:42 and follow transcript3:42
The problem is that this is computationally very costly. While we create one model in our classical train, validation, split scenario, we now create as many models as there are observations.
Play video starting at :3:56 and follow transcript3:56
This is not realistic even if you use very simple models. I told you that ten-fold cross-validation is quite common. The reason is that this creation of 10 folds is often a good compromise between model prediction, performance, and computational effort.
Play video starting at :4:15 and follow transcript4:15
There are some advantages of cross-validation versus the classical train, validation, split. First, it provides better and more stable results. It is also very helpful if you want to compare different machine learning algorithms. You can also use it for checking the impact of different hypertuning parameters on the quality of the model.
Play video starting at :4:37 and follow transcript4:37
The disadvantage is that it is computationally very expensive because many different models are created.
Play video starting at :4:46 and follow transcript4:46
That's it for this lecture. Thank you very much, and see you in the next one.