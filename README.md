<!--
*** Thanks for checking out this README Template. If you have a suggestion that would
*** make this better, please fork the tinyml-mapping-backlight and create a pull request or simply open
*** an issue with the tag "suggest".
*** Thanks again! Now go create something AMAZING! :D
***
***
***
*** To avoid retyping too much info. Do a search and replace for the following:
*** fullmakeralchemist, tinyml-mapping-backlight, twitter_handle
-->

<!--#     The TensorFlow Microcontroller Challenge    -->
   <h1>Hands Spelling Recognition with Object detection</h1>

<!-- PROJECT LOGO -->
<!--
<br />
<p align="center">
  <a href="https://github.com/fullmakeralchemist/tinyml-mapping-backlight">
    <img src="assets/logo.png" alt="Logo" width="720">
  </a>
  <br />
  -->

  <img src="https://img.shields.io/github/languages/top/fullmakeralchemist/tinyml-mapping-backlight?style=for-the-badge" alt="License" height="25">
  <img src="https://img.shields.io/github/repo-size/fullmakeralchemist/tinyml-mapping-backlight?style=for-the-badge" alt="GitHub repo size" height="25">
  <img src="https://img.shields.io/github/last-commit/fullmakeralchemist/tinyml-mapping-backlight?style=for-the-badge" alt="GitHub last commit" height="25">
  <img src="https://img.shields.io/github/license/fullmakeralchemist/tinyml-mapping-backlight?style=for-the-badge" alt="License" height="25">

  <a href="https://www.linkedin.com/in/fullmakeralchemist/">
    <img src="https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555" alt="LinkedIn" height="25">
  </a>
  <a href="https://twitter.com/makeralchemist/">
    <img src="https://img.shields.io/twitter/follow/makeralchemist?label=Twitter&logo=twitter&style=for-the-badge" alt="Twitter" height="25">
  </a>
  <!--
   <h3 align="center">Tiny ML in Mapping Dance, Visual Arts and interactive museums</h3>
  <p align="center">
    Because Art Inspired Technology, Technology Inspired Art
    <br />
    <a href="https://experiments.withgoogle.com/mapping-dance"><strong>View the project»</strong></a>
    <br />
  </p>
  <p align="center">
  <a href="https://experiments.withgoogle.com/mapping-dance">
    <img src="assets/TFChallengeWinners.png" alt="Logo" width="720">
  </a>
  </p>
  <br />
</p>
<br />
-->

<!-- TABLE OF CONTENTS -->
## Table of Contents

* [About the Project](#about-the-project)
  * [Motivation](#motivation)
  * [Built With](#built-with)
* [Getting Started](#getting-started)
  * [Prerequisites](#prerequisites)
  * [Installation](#installation)
* [Usage](#usage)
  * [Data Exploration](#data-exploration)
  * [Model Training](#model-training)
  * [Mapping and lightning Script Running](#mapping-and-lightning-script-running)
  * [Perform the Model](#perform-the-model)
* [Kinetic Sculpture](#kinetic-sculpture)
* [Challenges I ran into and What I learned](#challenges-i-ran-into-and-what-i-learned)
* [Observations about the project](#observations-about-the-project)
* [Accomplishments that I'm proud of](#accomplishments-that-im-proud-of)
* [What's next for Tiny ML in Mapping Dance, Visual Arts and interactive museums](##whats-next-for-tiny-ml-in-mapping-dance-visual-arts-and-interactive-museums)
* [License](#license)
* [Contact](#contact)

<!-- ABOUT THE PROJECT -->
## About The Project

<!-- [![Tiny ML in Mapping Dance](https://i9.ytimg.com/vi/3YUVTDTo-Zk/mq1.jpg?sqp=CNTs2IcG&rs=AOn4CLBiPsvQ2bGNVZvn_j-nJXj8d81hLA)](https://www.youtube.com/watch?v=3YUVTDTo-Zk) -->

The goal of this project is to detect and translate Sign Language (ASL) fingerspelling into text. You will create a model trained on the images dataset, custom created specifically to try different approaches. This may help move sign language recognition forward, making AI more accessible for the Deaf and Hard of Hearing community.

This project makes use of a machine learning platform that simplifies the process capable of identifying associated gesture recognition through images. This allows the user to work with it on custom needs.

### Motivation

Some facts:

Voice-enabled assistants open the world of useful and sometimes life-changing features of modern devices. These revolutionary AI solutions include automated speech recognition (ASR) and machine translation. **Unfortunately, these technologies are often not accessible to the more than 70 million Deaf people around the world who use sign language to communicate, nor to the 1.5+ billion people affected by hearing loss globally.**

##### Technology can be an element for good, but only when everyone is included.

But sign language recognition AI for text entry lags far behind voice-to-text or even gesture-based typing, as robust datasets didn't previously exist.

Technology that understands sign language fits squarely with AI solutions and makes it universally accessible and useful. AI principles also support this idea and encourage projects that empower people, widely benefit current and future generations, and work for the common good. This project can be scaled, and support individual user experience needs while interacting with technology.

### Built With

With a lot of love 💖, motivation to help others 💪🏼 and [Python](https://www.python.org/) 🐍, using:

* [Roboflow](https://app.roboflow.com/)
* [Google Colab](https://colab.research.google.com/) <img src="https://colab.research.google.com/img/favicon.ico" width="15"> (with its wonderful GPUs)
* Laptop with webcam
* [Streamlit](https://streamlit.io/)


<!-- GETTING STARTED -->
## Getting Started

Object detection is a groundbreaking computer vision task that has a ton of applications across various industries. It goes beyond traditional image classification, where a model assigns a single label to an entire image, to identify and locate multiple objects within an image, often accompanied by bounding boxes outlining their positions.

When working on custom models for object detection or other machine learning tasks, one of the challenges that researchers and developers may encounter is the lack of suitable databases or datasets. Overcoming these challenges often requires creativity and resourcefulness so this post will focus on how to create your custom database.

<p align="center">
<img src="media/dog.png" width="60%">
</p>

## Prerequisites

This is short list things you need to use the guide. 

* Python
* Git

## Part 1: Introduction and Setup for Roboflow
Welcome to Part 1 of our three-part tutorial series on Building Your Own Real-Time Object Detection App: Roboflow(YOLOv8) and Streamlit. In this series, we will walk you through the process of building an end-to-end object detection app that can identify objects from a photo. This web app was built only for images because we are using [share.streamlit.io](http://share.streamlit.io/) this is the Streamlit project hub where you can post your Streamlit projects free and it has a limit of 1 GB memory space for the app, there is a few libraries that cover a lot of that space so in another post or series I’ll add more about video and webcam functions to complement this app.

In Part 1, we will introduce the project, give you a demo of the app in action, and explain why I chose Roboflow and Streamlit for this project. We will also guide you through the setup process, including installing dependencies and creating the necessary files and directories.

By the end of this series, you will have the skills to build your own object detection app. So, let’s dive in!

### Demo of the Object Detection App
This is the [web app](https://objectdetection-eduardo.streamlit.app/) demo from the project that we are going to create and build together in the Streamlit share cloud. The app Object Detection will Upload an image on the WebApp and show detected objects.

### Object Detection
Object detection is a computer vision solution that identifies instances of objects in visual media. Object detection scripts draw a bounding box around an instance of a detected object, paired with a label to represent the contents of the box. For example, a person in an image might be labeled “person” and a car might be labeled “vehicle”.

### What is YOLOv8?
YOLOv8 is the newest state-of-the-art YOLO model that can be used for object detection, image classification, and instance segmentation tasks. YOLOv8 was developed by [Ultralytics](https://ultralytics.com/?ref=blog.roboflow.com), this model is used in Roboflow.

### Why Should I Use YOLOv8?
Here are a few main reasons why you should consider using YOLOv8 for your next computer vision project:

YOLOv8 has a high rate of accuracy measured by COCO and Roboflow 100.
YOLOv8 comes with a lot of developer-convenience features,an a well-structured Python package.
The labeling tool is easy to use and you don’t need to install a tool for that.
And last but not least is not difficult to run it also is faster than use a notebook with TensorFlow. In my case it takes 3 hours to train the model in Google Colab but with Roboflow it took me a few minutes.
### Why Streamlit is a Good Choice for Building a ML App
[Streamlit](https://docs.streamlit.io/) makes it easy to build web-based user interfaces for machine learning applications, enabling data scientists and developers to share their work with non-technical stakeholders.

Streamlit is an open-source framework that simplifies the process of building web applications in Python. And it has it’s own project cloud that makes really easy deploy your project.

### Project Setup: Installing Dependencies and Creating Required Files and Directories
Before diving into the project, make sure you have the following dependencies installed on your system. In my case I’m a Windows user so everything in this tutorial is working for July 2023 in Windows 11.

For this project I have Python 3.11 but in Streamlit cloud only has the version 3.8 to 3.11 so I recommend using that range of versions and the Python packages that we will use will be PyTorch, Ultralytics and Streamlit. We can install these packages using pip into a separate virtual environment.

### Creating Virtual Environment
When working on a Python project, it’s important to keep your dependencies separate from your global Python environment to prevent conflicts between different projects, especially with Pytorch.

Make sure you already have installed Python, VS code(or other IDE) and Git. Follow the next steps:

Create a new virtual environment by running the following command in the terminal after venv you can name as you wish your environment:
```
python -m venv env
```
Then activate the enviroment:
```
env\Scripts\activate
```
The first step is getting our data set (Images folder). In this case I recommend having at least 200 images. While the more pictures you have, the better your model becomes but don’t use pictures nearly identicals. I’m using 4 different sign hand posture so taking 50 photos with any device can take a lot of time so let’s create an environment only for the script that will take photos with our web cam. In this environment we only need to install OpenCV. So run in your terminal:

```
pip install opencv-python
```
Now you can run the following script, basically you can modify the labels, these labels will be used to create folders and will take the number of images that you declared. After finishing with the first label it will continue with the next one until it finishes the labels list. And will display a window that shows what is capturing. Also you can modify the time between each shot and time between the labels capture. Start taking pictures:

```
link codigo
```
At this point we will have the amount of images that we need but the name of each picture is random so we have to rename it to make it easier to identify each image. The next code will rename each image in just one folder so run the code for each folder in your project.

link codigo

### Create a project with Roboflow
Building a custom dataset can be a painful process. It might take dozens or even hundreds of hours to collect images, label them, and export them in the proper format. Fortunately, Roboflow makes this process straightforward. If you only have images, you can label them in [Roboflow Annotate](https://docs.roboflow.com/annotate?ref=blog.roboflow.com). (When starting from scratch, consider [annotating large batches of images via API](https://docs.roboflow.com/annotate/annotate-api?ref=blog.roboflow.com) or use the [model-assisted labeling](https://blog.roboflow.com/announcing-label-assist/) tool to speed things up.)

Before you start, you need to create a Roboflow [account](https://app.roboflow.com/login?ref=blog.roboflow.com). Once you do that, you can create a new project in the Roboflow dashboard.

<p align="center">
<img src="media/1.png" width="60%">
</p>

Keep in mind to choose the right project type. In this case choose, Object Detection.

<p align="center">
<img src="media/2.png" width="60%">
</p>

### Upload your images
Add data to your newly created project. You can do it through the [web interface](https://docs.roboflow.com/adding-data/object-detection?ref=blog.roboflow.com). If you don’t have a dataset, you can grab one from [Roboflow Universe](https://universe.roboflow.com/?ref=blog.roboflow.com).

If you drag and drop a directory with a data set in a supported format, the Roboflow dashboard will automatically read the images and annotations together. To create a data set with annotations locally in Windows check [this post](https://medium.com/@lalodatos/label-your-images-with-labelimg-in-windows-for-object-detection-models-1b0a66f00a7b).

<p align="center">
<img src="media/3.png" width="60%">
</p>

<p align="center">
<img src="media/4.png" width="60%">
</p>

After all images uploaded you can click Save and Continue.

<p align="center">
<img src="media/5.png" width="60%">
</p>

Then it will appear the pop-up window and you can Click only in Assing Images, in this part if you are working with a Team you can invite them to add images or labeling.

<p align="center">
<img src="media/6.png" width="60%">
</p>
Then we need to click Start Annotating in case you upload images only to use the label tool from Roboflow.

<p align="center">
<img src="media/7.png" width="60%">
</p>

### Label your images
Use the tool to select the element with the classes that you are going to use in your model. And repeat the same process for all the images.

<p align="center">
<img src="media/8.png" width="60%">
</p>

After you finish labeling all the images click the back button highlighted in red in the image below.

<p align="center">
<img src="media/9.png" width="60%">
</p>

Now we can add all the images to the Dataset with the button Add n Image to the Dataset.

<p align="center">
<img src="media/10.png" width="60%">
</p>

Now will appear the option to Add Images you can choose different options I recommend using the default option.

<p align="center">
<img src="media/11.png" width="60%">
</p>

After loading our images to the database another window will appear. You need to make sure that there are no UNASSIGNED images and the Dataset is ready, once you have it similar as the image below you can Click Generate New Version.

<p align="center">
<img src="media/12.png" width="60%">
</p>

When we Generate a New Version we can use some tools to prepare the data and experiment with them. Go to option 3.

<p align="center">
<img src="media/13.png" width="60%">
</p>

In this option we can apply transformations in all the images, so make sure to configure this depending on your project. Maybe you are using a camera in Raspberry Pi or maybe you want to use images with a specific format. For my project this configuration is perfect.

<p align="center">
<img src="media/14.png" width="60%">
</p>

Option 4 is an amazing tool because you can generate extra versions from your images that can duplicate or triplicate in the free version of the dataset. Let’s see the options.

<p align="center">
<img src="media/15.png" width="60%">
</p>

For this project I’ll use flip horizontal, try to experiment with it, and depending on your project you can choose the options that you need.

<p align="center">
<img src="media/16.png" width="60%">
</p>

After you choose an Augmentation you will see extra options. For my project I only need the Horizontal. Try to check what is best for your custom project. After that click Apply

<p align="center">
<img src="media/17.png" width="60%">
</p>

Then click continue to step 5 and last.

<p align="center">
<img src="media/18.png" width="60%">
</p>

Select the Maximun Version and then click Generate and is ready to go.

<p align="center">
<img src="media/19.png" width="60%">
</p>

After this will appear the next page:

<p align="center">
<img src="media/20.png" width="60%">
</p>

Congratulations now you have an Image Dataset ready to train a model.

## Train YOLOv8 on a custom dataset
In this section, we will dive deeper into the YOLOv8 object detection model and explore how to train it .

There are a wide range of open-source object detection models available. A popular choice is models in the YOLO (You Only Look Once) family, which continue to represent the state-of-the-art in object detection tasks.

Once you have a labeled dataset, and you have made your augmentations, it is time to start training an object detection model. Training involves showing instances of your labeled data to a model in batches and iteratively improving the way the model is mapping images to predictions.

As with labeling, you can take two approaches to training and inferring with object detection models train and deploy yourself, or use training and inference services like Roboflow Train and Roboflow Deploy. Both of which are free for Public plans.

In [Upload your images](#label-your-images) we finished the Versions tool from our Roboflow project now is time to train the model. We have to choose the option Custom Train using YOLOv5 and then Get Snippet.

<p align="center">
<img src="media/21.png" width="60%">
</p>

A pop up copy the lines or save it we need the api_key to modify the notebook, will open a notebook in Google Colab after clicking Copy Snippet. Is a repository make sure to create a copy to save the changes first.

<p align="center">
<img src="media/22.png" width="60%">
</p>

When you open the notebook it is necessary to run all to set up the Colab session. There are a few cells that you can avoid but check it first.

<p align="center">
<img src="media/23.png" width="60%">
</p>

If we remember we have the api_key and extra information about our data set we will use it in the Step 5: Exporting dataset from the Notebook we will find a code cell and we need to replace with the copied lines from Roboflow after that we can run everything without modifying anything else.

<p align="center">
<img src="media/24.png" width="60%">
</p>


### Deploy model on Roboflow
Once you have finished training the YOLOv8 model, you’ll have a set of trained weights ready for use. These weights will be in the /runs/detect/train/weights/best.pt folder of your project. You need to download the filebest.pt to use it in the Streamlit app.

<p align="center">
<img src="media/25.png" width="60%">
</p>

<!-- USAGE EXAMPLES -->
## Creating a Streamlit WebApp for Image Object Detection with a Roboflow model
Streamlitis an open-source app framework for Machine Learning and Data Science teams. Create beautiful web apps in minutes. Streamlit apps are Python scripts that run from top to bottom. Every time a user opens a browser tab pointing to your app, the script is re-executed. As the script executes, Streamlit draws its output live in a browser.

[Create an app](https://docs.streamlit.io/library/get-started/create-an-app) using Streamlit’s core features to fetch and cache data, draw charts, plot information on a map, and use interactive widgets, like a slider, to filter results.

Let’s prepare the virtual environment for the Streamlit app. First let’s create a virtual environment and once created then activate it (Windows).

```
python -m venv env
env\Scripts\activate
```

Then we have to install PyTorch, Ultralytics and Streamlit. Try to install in the next order.

```
pip install torch
pip install ultralytics
pip install streamlit
```

After this we are ready to try the hello world in Streamlit to check that everything is installed correctly. Create a file called app.py and put the next code lines using your favorite IDE:

```
import streamlit as st
st.write("Hello, World!")
```

Then run it from the terminal in cmd and if everything works fine will open the browser.

```
streamlit run app.py
```

Then to create a tool to upload our pictures and use the model we need to open the code editor and let’s get started by replacing the previous file and creating a new one named app.py. But we also need a folder called weights and for the moment is everything. Now let’s go to the next step.

## Create a Uploading an Image On Streamlit WebApp
We’ll use Streamlit to allow users to upload an image. After successfully uploading an image, is ready to run object detection on the uploaded image using YOLOv8. This step involves loading the YOLOv8 model and passing the uploaded image through the model to identify the objects present in the image.

We will also visualize the output of the model with the identified objects highlighted in the image. Let’s go into the code.

In Part 2 of this series, we have discussed how to download a pre-trained weight file of the Yolov8 model. downloaded the best.pt file and saved it inside our weights directory. We will use the same weight file. In the created file with the name app.pywrite the following lines of code:




## Challenges I ran into and What I learned

One of the main challenges was to create a model in Tensorflow without having much knowledge about Machine Learning, my major area of studies is hydrology, and irrigation. That's why I got the Tiny ML book from Pete Warden and Daniel Situnayake and then I found the Tiny Ml Workshop resource may result in the most suitable code resource to tackle this area of necessity. 

The second main challenge (once I learned about creating the model on Arduino) was to create bluetooth connections without having much knowledge about the BLE library, I used before for other irrigation projects WiFi connection using Firebase real-time database and MQTT. This is why I found that Mosquitto and Paho on Raspberry Pi may result in the most suitable technologies to tackle wireless connection, so it could get control in real-time. 

Finally, this is the first time I use a trained model using tflite on an Arduino with Raspberry Pi. In the end, I learned that whenever you may think that you found no way out, the motivation may help you to find alternative solutions with new technologies.

**About the Bluetooth attempt here is the script and the sketch to get data through a [bluetooth connection](https://github.com/fullmakeralchemist/tinyml-mapping-backlight/tree/master/Bluetooth_attempt) follow the instructions**

## Observations about the project

The Bluetooth connection has a limited number of devices to connect on the Raspberry Pi, it only allows 7 devices using bluetooth. In windows is 10 devices so with a lot of dancers it will be difficult using bluetooth. 

Training the moves could be hard doing more than 20 repetitions of a movement, also I realize recording the moves, that is necessary to be really precise doing the movements, the difference in each repetition affects the model precision. 

## Accomplishments that I'm proud of

- Building a custom script to just change a few variables
- Sending data using the BLE library to my laptop using Python
- Sending messages to a ESP8266 board using MQTT
- Learning new technologies in a record time
- Start creating a tool that will help others

## What's next for Tiny ML in Mapping Dance, Visual Arts and interactive museums

- Develop own embedded device for the model deployment (which should already include a accelerometer, gyroscope and a wifi connection)
- Improve user data acquisition through the accelerometer and gyroscope.
- Add Bluetooth recording of accelerometer and gyroscope (you can find a file using python as a receiver of the information and the INO file that sends the data through the BLE library).
- Implement in a dance presentation or a museum (also in my house in holidays)
- Add kinetic sculptures with servo motors to add an effect like the matilda movie.
- Test prototype with a dancer.
- Add the MadMapper API to add more visual effects with the animations.

<center>
<img src="assets/matilda.gif" width="60%">
</center>

## License


<!-- CONTACT -->
## Contact

Eduardo Padron - [@makeralchemist](https://twitter.com/makeralchemist) - 

Project Link: [https://github.com/fullmakeralchemist/tinyml-mapping-backlight](https://github.com/fullmakeralchemist/tinyml-mapping-backlight)

IF YOU THINK THAT YOU CAN HELP ME TO HELP OTHERS, PLEASE DO NOT HESITATE TO CONTACT ME.

<!-- ACKNOWLEDGEMENTS -->
## Acknowledgements

* Icons made by [Flat Icons](https://www.flaticon.com/authors/flat-icons) from [www.flaticon.com](https://www.flaticon.com/)
* Images and gifs made by [Canva](https://www.canva.com/) 
* Thanks to the Tensorflow team Arduino and Raspberry Pi for developing such an incredible technology