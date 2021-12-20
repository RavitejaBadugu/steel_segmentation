# steel_segmentation
The data is taken from https://www.kaggle.com/c/severstal-steel-defect-detection competition.

The metric used is Dice coefficient. The data has images which can be part of no classes, 1 class or multiple classes.

My private score is 0.84898.
* submission image
![Alt text](https://github.com/RavitejaBadugu/steel_segmentation/blob/main/steel_images/Screenshot%202021-12-20%20202932.png)

**I used dvc for data versioing and model tracking.**

**segmentation_models library is used for training model because it contains pre-trained models as encoder.**

**I also used tensorflow serving for model deployment as Restapi.**

**For user interactive I used Streamlit and backend operations like preprocessing images and predicting images are written in fastapi and for model deployment instead of loading 
model everytime we get the data i.e) to decrease the overload in fastapi. I used tensorflow serving. So I can deploy the models any where and use the endpoint in fastapi 
application to get prediction.**

**Created a containerized application. This application can be used later to deploy in heroku or in any cloud platform.**


Streamlit api interface
![Alt text](https://github.com/RavitejaBadugu/steel_segmentation/blob/main/steel_images/Screenshot%202021-09-15%20185602.png?raw=true "Title")

predictions after uploading image

![Alt text](https://github.com/RavitejaBadugu/steel_segmentation/blob/main/steel_images/Screenshot%202021-09-15%20185737.png?raw=true "Title")
