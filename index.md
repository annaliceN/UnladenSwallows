## Team Unladen Swallows
#### Donna Albee (albeed), Sam Lewis (sameul26), Annalice Ni (anni00), Eric Yoon (yoone2)

### Context:

<iframe width="560" height="315" src="https://www.youtube.com/embed/liIlW-ovx0Y" title="YouTube video player"
    frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
    allowfullscreen></iframe>

### Introduction

####  The Problem

What are birds? 

Seemingly everywhere, birds can be found at almost any time of the day. Their tweets can wake you up at the crack of dawn, they can be found assailing innocent passersby for food around noon and seen crowing ominously in the trees at dusk. Yet how do you identify birds? What is the difference between a crow and a raven? Do birds even exist?

These deep philosophical questions are what prompted us to work on classifying different bird species for our final project. Through the Birds! Kaggle Competition, our goal was to train a model that would classify birds in an image. This would allow us to answer important questions like

> How do you tell the difference between an African and European swallow?

And many more! 

#### Datasets and Data Augmentation

For our dataset, we used the Bird Dataset provided by the Birds! Classification Kaggle Competition by Joseph Redmond. This dataset has approximately 50,000 images in it and 555 unique classes of birds. When processing images, we used both 224 x 224 and 128 x 128 images, as well as random modifications to images to diversify our training set. The transformations included cropping images and flipping images.
