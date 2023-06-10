# CSE 455 Project
This project was worked on by Kevin Hoang for CSE 455: Computer Vision at UW for Spring Quarter 2023.
## Intro
Computer Vision is a really powerful tool that can be applied to many problems in society, ranging from fields such as medicine to law. By teaching a neural network how to identify a problem visually, one can reduce the manpower required in many industries. My motivation is to see if computer vision to identify "reckless behaviors", such as drivers who don't wear seatbelts or people who litter. Such infractions may not be worth pursuing by authorities, but an automated system could be a way to curtail such behaviors without requiring additional manpower and funding. 

I chose a simple problem to solve first, one that had relevence when the pandemic was at its height. The problem I wanted to solve was whether a computer system can identify whether someone is wearing a face mask or not. Because of that experience, facial recognition software had to adapt to detecting when a user is wearing a face mask and change their recognition criterias accordingly. However, I later encountered some issues that I would talk about later. As a result, I tried to switch over and use my same model on another dataset with a similar theme. I wanted to find one that had drivers who wore/didn't wear seatbelts, as this could be a good technology for traffice cameras. However, the closest dataset I found to that theme composed of images of drivers in various behaviorial states, beyond the scope of just wearing a seatbelt. I found that my code can be easily rereused without the need for modifications to the training method and only the data import part needs to be slightly modified (if the data isn't all together and separted in folders by class).

## Approach
In order to solve this classification problem, at first I planned to use a mixture of semantic segmentation and object detection on a dataset of images of people with facemasks, without facemasks, and wearing them incorrectly. However, this required too much manpower to do the labeling. Hence, I decided to use the traditional transfer learning route, which has three benefits. First, transfer learning allows me to try out different already optimized neural networks. Secondly, because these pre-trained networks have been trained on a large assortment of items, basing my model's off them would shorten training time and allow my model to specialize in the specific area I want while still having good general performance. Lastly, my code can be easily reused for a different usecase with equally as good performance (generalizability of this pre-trained networks) as later demonstrated in this final project.

I used Resnet-18 with mini-batch Stochasistic Gradient Descent and holdout cross-validation. 
- [ResNet18](https://arxiv.org/abs/1512.03385) is a pre-trained model that won 1st place at the ImageNet Challenge (ILSVRC) 2015. It was a deep neural network model developed by a [team of researchers at Microsoft Research](https://blogs.microsoft.com/ai/microsoft-researchers-win-imagenet-computer-vision-challenge/) that allows for multi-layer networks where neural depth does have performance gains for a model's accuracy. This is done by the use of residual learning and residual connections being established between these layers. Resnet is a commonly used pre-trained model. I chose the 18 layer variant for less complexity and faster processing while not suffering on performance. My Resnet model is initialized with n-nodes, where n is the amount of classes a trained dataset has (3 for the mask dataset and 10 for the Statefarm dataset).
* Stochastic Gradient Descent is used as our optimizer. We used the mini-batch variant approach as it is typically considered to be the best gradient descent variant because of its high paremeter updates frequency while being more computationally efficient than stochasistc gradient descent. 
+ Holdout is used as it is a simplistic cross-validation approach that doesn't require modifying our testing, validation, and training datasets every iteration. It has a downside of not being able to guarantee that our training dataset represents the entire dataset, compared to more advanced methods such as k-fold cross validation. However, with a large dataset such as ours and the fact that there are a small number of possible classifications for each image, our training dataset will be representative of the entire dataset. We want to use validation to get an unbiased evaluation of our current model fit per epoch, to determine whether our training model is getting better or worse. A 80-10-10 training/validation-test split is used as standard in industry.

Originally, I wrote a training method that was gradient descent where the entire dataset was trained and then parameters were updated once during each epoch, rather than my minibatch approach. For the mask dataset, it told me that the accuracy of the starting epochs were 95 percent with a low loss. During subsequent iterations, my accuracy and loss didn't improve (both validaiton and training). Hence, I decided to use the same training method on a different larger more complicated dataset (the StateFarm one) where the performance was horrible. Across epochs, my accuracy and loss stayed relatively the same. Hence, I knew my code had flaws and I rewrote it. After making sure it worked for the StateFarm dataset, I applied my code to the Mask dataset too.

### Parameters
- Learning Rate: .05
* Momentum: .9
* Epochs: 10
- Train/Validation/Test Ratio: 80-10-10
## Dataset
We use two datasets on Kaggle. The first is the [Face Mask Detection](https://www.kaggle.com/datasets/vijaykumar1799/face-mask-detection) dataset which contains 3 classes (with mask, without a mask, and wearing mask incorrectly). This dataset contains 8982 images split evenly between the three classes. The images from this dataset were sourced from two other mask datasets([1](https://www.kaggle.com/datasets/ashishjangra27/face-mask-12k-images-dataset) and [2](https://www.kaggle.com/datasets/andrewmvd/face-mask-detection)) so that no samples are noisy and the classes are evenly distributed. 

Our second dataset is the [State Farm Distracted Driver Detection](https://www.kaggle.com/c/state-farm-distracted-driver-detection) which contains ten classes ( safe driving, texting - right, talking on the phone - right, texting - left, talking on the phone - left, operating the radio, drinking, reaching behind, hair and makeup, and talking to passenger). I didn't use their complete dataset, instead I used a subsection. There are 22,424 photos in the training folder separated into different directories based on class (just like the Face Mask data). I used these images as my overall dataset so that I won't have to cross reference their .csv file for class types.

Each image is resized to 224 pixels and normalized by the same mean (0.485, 0.456, 0.406) and std values (0.229, 0.224, 0.225) that to be of the same format of the images ResNet was originally trained on. Each image has a 50 percent chance of being flipped horizontally so to prevent overfitting.  As stated before, the images in both datasets are then split into training, testing, and validation sets once.

## Results & Discussion
Let's first take a look at the performance of our methodology on the mask dataset. The best trained model provided a performance accuracy of 97.88 percent on the testing dataset. Let's see if our model can label a group of images from the test dataset correctly. We'll find that our labels are correct!

![Mask Demonstration](/figures/Mask_demonstration.png)

Below are the training and validation loss plots. We cannot overlay them directly to create a Cross-Entropy plot as training loss is calculated per minibatch while validation loss is calculated per epoch. 

![Mask Training Loss](/figures/Mask_ResNet18_SGD_Training_Loss.png)

![Mask Validation Loss](/figures/Mask_ResNet18_SGD_Validation_Loss.png)

Below are the training and validation accuracy plots. Once again, we cannot overlay them directly to create a Cross-Entropy plot as training loss is calculated per minibatch while validation loss is calculated per epoch. 

![Mask Training Accuracy](/figures/Mask_ResNet18_SGD_Training_Accuracy.png)

![Mask Validation Accuracy](/figures/Mask_ResNet18_SGD_Validation_Accuracy.png)

We see that our accuracy increases and converges at close to 100 percent accuracy and that our losses are reduced significantly. We see that we converge around 10 epoch at high accuracy with a significant reduction in loss in our validation data.

Let's use our exact code and methodology on the StateFarm dataset. The best trained model provided a performance accuracy of 98.79 percent on the testing dataset. Let's see if our model can label a group of images from the test dataset correctly. Our labels are correct!

![StateFarm Demonstration](/figures/StateFarm_demonstration.png)

Below are the training and validation loss plots. 

![StateFarm Training Loss](/figures/StateFarm_ResNet18_SGD_Training_Loss.png)

![StateFarm Validation Loss](/figures/StateFarm_ResNet18_SGD_Validation_Loss.png)

Below are the training and validation accuracy plots. 

![StateFarm Training Accuracy](/figures/StateFarm_ResNet18_SGD_Training_Accuracy.png)

![StateFarm Validation Accuracy](/figures/StateFarm_ResNet18_SGD_Validation_Accuracy.png)

We see the same performative behavior across datasets. Our model is pretty good!

## Conclusion
We found that our methodology can be applied to different "behavorial" image classification problems with ease. We demonstrated the power of transfer learning and how powerful it is in predicting human behavor and reducing the need for human oversight. However, the ease of access of such technology should be a worry to us all. As computer scientists, we still have a duty to ensure our work is ethical and does not cause harm. While a computer vision program that identifies problematic behavior such as littering may sound fine, it is a slippery slope. Authortarian governments can use the same principles demonstrated here alongside their pre-existing public surveillence infrastructures to crack down on dissidents, journalists, and protestors. Or, corporations could use this technology to monitor their workers' productivity at all times (corporations such as [Amazon](https://www.theverge.com/2019/4/25/18516004/amazon-warehouse-fulfillment-centers-productivity-firing-terminations) already track productivty). This technology really has the power to harm the lives of people (through arrests and job terminations) if used in a wrong and unfair manner.

One thing that would be interesting for a further line of research is to see the performance of our system on noisy images. The images in this datasets were all cleaned and showed only one subject. Would our model be able to identify multiple people with different states of wearing or not wearing masks? Furthermore, does our model have racial biases? Current facial recognition technologies suffer from [racial biases ](https://sitn.hms.harvard.edu/flash/2020/racial-discrimination-in-face-recognition-technology/) likely due to not-equally diverse datasets. Were the datasets that this model was trained on diverse? No such data is collected or mentioned in the dataset, so we would have to classify each image based on race to see if there is an equally racially diverse dataset. Then, we have to see if there are performance disparities between races. This is important, as previously mentioned, if this technology is used for employment and the justice system, biases and failuires would harm people's lives.
