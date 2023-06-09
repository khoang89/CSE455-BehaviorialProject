# CSE 455 Project
## Intro
Computer Vision is a really powerful tool that can be applied to many problems in society, ranging from fields such as medicine to law. By teaching a neural network how to identify a problem visually, one can reduce the manpower required in many industries. My motivation is to see if computer vision to identify "reckless behaviors", such as drivers who don't wear seatbelts or people who litter. Such infractions may not be worth pursuing by authorities, but an automated system could be a way to curtail such behaviors without requiring additional manpower and funding. 

I chose a simple problem to solve first, one that had relevence when the pandemic was at its height. The problem I wanted to solve was whether a computer system can identify whether someone is wearing a face mask or not. Because of that experience, facial recognition software had to adapt to detecting when a user is wearing a face mask and change their recognition criterias accordingly. However, I later encountered some issues that I would talk about later. As a result, I tried to switch over and use my same model on another dataset with a similar theme. I wanted to find one that had drivers who wore/didn't wear seatbelts, as this could be a good technology for traffice cameras. However, the closest dataset I found to that theme composed of images of drivers in various behaviorial states, beyond the scope of just wearing a seatbelt.

## Approach
In order to solve this problem, at first I planned to use a mixture of semantic segmentation and object detection on a dataset of images of people with facemasks, without facemasks, and wearing them incorrectly. However, this required too much manpower to do the labeling. Hence, I decided to use the traditional transfer learning route, which has two benefits. First, transfer learning allows me to try out different already optimized neural networks. Secondly, because these pre-trained networks have been trained on a large assortment of items, basing my model's off them would shorten training time and allow my model to specialize in the specific area I want while still having good general performance.

I planned to use Resnet-18

I found two data sets that met this requirement and will try to test my model on both datasets. One dataset is more cleaned out, which will more likely be the one I use if I only chose to build my model and test it on one dataset.

## Previous Work

## Dataset

## Results

## Discussion
