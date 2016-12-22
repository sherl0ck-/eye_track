Overview
========

This repository contains the files, work and final report for the NYU Foundations of Machine Learning graduate class for Fall 2016. 

We sought to expand on previous work in gaze tracking by testing traditional regression models against the collected data set consisting of 50000 image screen-coordinate pairs. 

With respect to the folders, 

* **datacollection** contains the application built in Objective-C used for datacollection
* **preprocess** contains the OpenCV application built to preprocess the images
* **report.pdf** is the final report written for the class
* Moreover **pipeline.py** was used to train the models, while **data-final50000.txt** contains the final data. Lastly, the matlab file contains the image saliency map production. Note that you will need SaliencyToolbox (couple of Mb, download at www.saliencytoolbox.net/) by Walter.

In order to run the models run
```bash
python pipeline.py data-final50000.txt 1 1
python pipeline.py data-final50000.txt 0 1
```
in the terminal. 

Any feedback is greatly appreciated.

Frederik Jensen and Jovan Jovancevic
