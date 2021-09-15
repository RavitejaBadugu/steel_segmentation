# steel_segmentation
The data is taken from https://www.kaggle.com/c/severstal-steel-defect-detection competition.

The metric used is Dice coefficient. The data has images which can be part of no classes, 1 class or multiple classes.

I used dvc for data versioing and model tracking.
segmentation_models library is used for training model.
I also used tensorflow serving for model deployment as Restapi.
For user interactive I used Streamlit and backend operations like preprocessing images and predicting images as written in fastapi and for model deployment instead of loading 
model everytime we get the data i.e) to decrease the overload in fastapi. I used tensorflow serving. So I can deploy the models any where and use the endpoint in fastapi 
application to get prediction. I created Docker-Compose and combined all three and deployed. This docker-compose can be used later to deploy in heroku or in any cloud platform.
