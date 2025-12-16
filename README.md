# Physics 188 Final Project: Classifying Aurora Structures with Machine Learning Methods

Authors: Claire Chen, Felix Chou, Ishaan Iyer, Pranathi Kolla, Melody Wu, Sean Xu

### Data Used

Data: https://drive.google.com/drive/folders/15OIvDkIMuZKDP-1Dc4Lxhgs2D9TJoHJA?usp=drive_link 

Dataset (filtered and tagged images) index: https://docs.google.com/spreadsheets/d/1-Cyp22nBqxQUManQtD_KFQZszfEfuLIicsKJZnjpQpY/edit?usp=drive_link 

### Code Breakdown 

Download_and_Class is used to first, load tha dataset into usable format, and then run a binary filter
based on color and lightness to classify images as containing an aurora or not.

Aurora_Classification_with_Grayscale converts images to grayscale and uses PyTorch to create a CNN that can
classify the shape/content of the image.
