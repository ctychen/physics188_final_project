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

### Notes


- **Aurora_Dates.ipynb**: Finds dates with high geomagnetic activity and thus high likelihood of auroras being observable, using a set of GM activity metrics from the OMNIWeb database. Also filters these times and dates by sunset / sunrise time to maximize the likelihood that observatories are up and recording, and actually able to see auroras in a dark sky, to minimize the risk of us downloading unusable data.

- **Aurora Dataset Loading.ipynb**: This code is heavily based on the code provided to us by Dr. Harding, Alex Toohey, Tommy Duong, and Sabrina Nazarzai on a research project using the pyaurorax package. The part of the code that downloads the data is essentially the same, with a few changes to variables and flow. The code for the multiple day downloading was also based on their code with changes to the logic and the specified parameters to allow for more control over data downloading. The saving of the frames as pdfs is original code.

- **(final)Download_and_Class.ipynb**: Combining our dataset downloading and image classification code into a single pipeline - given dates and times of interest, pull data from as many observatories as possible, and see what this looks like in terms of containing interesting aurora features, using the binary classifier to get a quick look at what colors and features might be present. We ran this code to pull all of the data used in building our dataset. 

- **Initial_binaryClassification.ipynb**: Implements a binary classifier for images using kmeans, including some processing of the input images, and identifies cluster hues that correspond to an image containing an aurora or not. 

- **Aurora_Classification_with_Grayscale.ipynb**: Implements a CNN using pytorch, which then trains on the dataset comprised of observatory images we pulled and tagged. The model converts images to grayscale for training and testing. Training is done with a pretrained ResNet-18 model. 
