# Author Ibrahim Koné 
# New York City Taxi Fare Prediction Kaggle Web App

[Download Dataset Here](https://www.kaggle.com/c/new-york-city-taxi-fare-prediction)

Do you remember the days before Uber, Lyft, or Gett? Standing in the street trying to hail a taxi waiting for the moment a free cab might drive by and spot you? These days that world seems so far away. And you might often wonder: how do these apps work.

<br />

## How to use it

```bash
$ # Get the code
$ git clone https://github.com/Ibmaria/New-York-City-Taxi-Fare-Prediction.git
$ cd New-York-City-Taxi-Fare-Prediction
$
$ # Virtualenv modules installation (Unix based systems)
$ virtualenv env
$ source env/bin/activate
$
$ # Virtualenv modules installation (Windows based systems)
$ # virtualenv env
$ # .\env\Scripts\activate
$
$ # Install modules - SQLite Storage
$ pip3 install -r requirements.txt or pip install -r requirements.txt
$
$ # Create tables
$ python manage.py makemigrations
$ python manage.py migrate
$
$ # Start the application (development mode)
$ python manage.py runserver # default port 8000
$
$ # Start the app - custom port
$ # python manage.py runserver 0.0.0.0:<your_port>
$
$ # Access the web app in browser: http://127.0.0.1:8000/projets/taxi/
```





<br />

## App Screenshot
![App ](https://github.com/Ibmaria/New-York-City-Taxi-Fare-Prediction/blob/master/captureapp.PNG)

## Download Video App Here
![App Video](https://github.com/Ibmaria/New-York-City-Taxi-Fare-Prediction/blob/master/videoapp.gif)


## Codebase structure

The project is coded using a simple and intuitive structure presented below:

```bash
< PROJECT ROOT >
   |
   |-- taxi/                              
   |--taxidriver/
   |--Docker-version
   |--classify_image.py                        
   |--Dockerfile              
   |--requirements.txt
   |--manage.py
   |--pima_dataset_lgbm_final.txt
   |-- ************************************************************************
```

<br />





