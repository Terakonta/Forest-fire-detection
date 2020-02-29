# WildFire-Project

## Introduction
My team and I are undergraduate students at the University of Toronto. We joined LearnAI which is a society aiming to help students who can't take AI related courses, either because they are in lower years or aren't study computer science, and give them an opportunity to explore this fascinating field. After a semester of workshops, this is the first time we worked on an ML project.

## Problem
Massive wildfires are destroying homes and habitats aroudn the world. With wildfires, early detection and prevention becomes key to containing the fire.
Most recently Australia's bush fire has been estimated to have taken 1.25 billion animal life. The late 2018 Woolsley fire in California has detsroyed more than 1500 building.

## Solution
States like California and Nevada has set up surveillance cameras along wildfire hotspots. These cameras are monitored by volunteers during peak season. However we believe that these can be automated by the help of machine learning algorithms.

What this project aims to do is to detect fires from surveillance footage using convolutional neural networks. 

## Data
The dataset for this project was taken from AlertWildfire's webpage for livestream footage of firehotspots, and their youtube channel for videos of past fires.

Not fire/ livestrem: http://www.alertwildfire.org/

Fire feeds: https://www.youtube.com/user/nvseismolab/videos?disable_polymer=1

## Model
We used Convolutional Neural Networks on images taken from the videos. For our model, we had to first extract frames from the videos, and crop them to get rid of labels and coordinates and turn them into black and white. 

## Results
So far we have been able to reach a 94% testing accuracy with our model. We are currently working on imporving our accuracy for early detection as well as using different models to allow for live detection (improve detection speed). 
