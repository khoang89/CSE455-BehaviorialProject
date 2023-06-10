# CSE 455 Project
## Intro
Computer Vision is a really powerful tool that can be applied to many problems in society, ranging from fields such as medicine to law. By teaching a neural network how to identify a problem visually, one can reduce the manpower required in many industries. My motivation is to see if computer vision to identify "reckless behaviors", such as drivers who don't wear seatbelts or people who litter. Such infractions may not be worth pursuing by authorities, but an automated system could be a way to curtail such behaviors without requiring additional manpower and funding. 

I chose a simple problem to solve first, one that had relevence when the pandemic was at its height. The problem I wanted to solve was whether a computer system can identify whether someone is wearing a face mask or not. Because of that experience, facial recognition software had to adapt to detecting when a user is wearing a face mask and change their recognition criterias accordingly. However, I later encountered some issues that I would talk about later. As a result, I tried to switch over and use my same model on another dataset with a similar theme. I wanted to find one that had drivers who wore/didn't wear seatbelts, as this could be a good technology for traffice cameras. However, the closest dataset I found to that theme composed of images of drivers in various behaviorial states, beyond the scope of just wearing a seatbelt. I found that my code can be easily rereused without the need for modifications to the training method and only the data import part needs to be slightly modified (if the data isn't all together and separted in folders by class).

## Approach
In order to solve this classification problem, at first I planned to use a mixture of semantic segmentation and object detection on a dataset of images of people with facemasks, without facemasks, and wearing them incorrectly. However, this required too much manpower to do the labeling. Hence, I decided to use the traditional transfer learning route, which has three benefits. First, transfer learning allows me to try out different already optimized neural networks. Secondly, because these pre-trained networks have been trained on a large assortment of items, basing my model's off them would shorten training time and allow my model to specialize in the specific area I want while still having good general performance. Lastly, my code can be easily reused for a different usecase with equally as good performance (generalizability of this pre-trained networks) as later demonstrated in this final project.

I used Resnet-18 with mini-batch Stochasistic Gradient Descent and holdout cross-validation. 
- [ResNet18](https://arxiv.org/abs/1512.03385) is a pre-trained model that won 1st place at the ImageNet Challenge (ILSVRC) 2015. It was a deep neural network model developed by a [team of researchers at Microsoft Research] (https://blogs.microsoft.com/ai/microsoft-researchers-win-imagenet-computer-vision-challenge/) that allows for multi-layer networks where neural depth does have performance gains for a model's accuracy. This is done by the use of residual learning and residual connections being established between these layers. Resnet is a commonly used pre-trained model. I chose the 18 layer variant for less complexity and faster processing while not suffering on performance. My Resnet model is initialized with n-nodes, where n is the amount of classes a trained dataset has (3 for the mask dataset and 10 for the Statefarm dataset).
* Stochastic Gradient Descent is used as our optimizer. We used the mini-batch variant approach as it is typically considered to be the best gradient descent variant because of its high paremeter updates frequency while being more computationally efficient than stochasistc gradient descent. 
+ Holdout is used as it is a simplistic cross-validation approach that doesn't require modifying our testing, validation, and training datasets every iteration. It has a downside of not being able to guarantee that our training dataset represents the entire dataset, compared to more advanced methods such as k-fold cross validation. However, with a large dataset such as ours and the fact that there are a small number of possible classifications for each image, our training dataset will be representative of the entire dataset. We want to use validation to get an unbiased evaluation of our current model fit per epoch, to determine whether our training model is getting better or worse. A 80-10-10 training/validation-test split is used as standard in industry.
### Parameters
- Learning Rate: .05
* Momentum: .9
* Epochs: 10
- Train/Validation/Test Ratio: 80-10-10
- 
## Dataset

## Results

## Discussion
