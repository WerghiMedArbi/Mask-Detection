# Mask-Detection
𝐑𝐞𝐚𝐥-𝐓𝐢𝐦𝐞 𝐅𝐚𝐜𝐞 𝐦𝐚𝐬𝐤 𝐝𝐞𝐭𝐞𝐜𝐭𝐢𝐨𝐧  

### Libraries used

-KERAS

-NUMPY


-OPENCV


-SKLEARN



## System Overview

It detects human faces with 𝐦𝐚𝐬𝐤 𝐨𝐫 𝐧𝐨-𝐦𝐚𝐬𝐤 even in crowd in real time with live count status and notifies user (officer) if danger.



## Quick-Start
**Step 1:**
```
git clone https://github.com/WerghiMedArbi/Mask-Detection
```


**Step 2: Install requirements.**
```
pip install opencv-python
pip install np-utils
```
**Step 3: Run Droidcam on your pc and phone**
put the IP Cam Access that you got from your phone in the application on your pc
Optional: use the default cam of your laptop using this  ```source=cv2.VideoCapture(0)```

**Step 4: Run preprocessing+convolutional_neural_networks.ipynb**
```
python preprocessing+convolutional_neural_networks.ipynb
```

Download the dataset from ``` kaggle.com ```
Change the path in the code.

**Step 5: Run detectorwerghi.py**
```
python detectorwerghi.py
```

## Haarcascades 
a machine learning-based approach where a lot of positive and negative images are used to train the classifier. Positive images – These images contain the images which we want our classifier to identify. Negative Images – Images of everything else, which do not contain the object we want to detect.

``` https://github.com/kipr/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml ```


## TO DO

add an alert system
