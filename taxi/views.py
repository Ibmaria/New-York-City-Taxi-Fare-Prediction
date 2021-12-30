from django.shortcuts import render
#from keras.models import load_model
#from tensorflow.keras.models import load_model
#import keras
from django.http import JsonResponse
import os
from .helper import *
import pandas as pd
import datetime as dt
import pickle
import numpy as np
import lightgbm as lgbm
from pickle import load
# example of a standardization
from numpy import asarray
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
# define data

from numpy import asarray
import joblib



def index_taxi(request):
    context ={}
    return render (request,"taxi/taxi.html",context)
def index_diabetes(request):
	context={}
	return render (request,"taxi/diabetes.html",context)
def index_wiscon(request):
	context={}
	return render (request,"taxi/index_wiscon.html",context)
def index_blood(request):
	context={}
	return render (request,"taxi/index_blood.html",context)
def predict_cancer(request):
	filename = os.path.join(os.getcwd(),'winsconsin_cancer_dataset.txt')
	model_cancer= lgbm.Booster(model_file=filename)
	radius_mean = request.POST.get("radius_mean")
	radius_mean=float(radius_mean)
	
	texture_mean = request.POST.get("texture_mean")
	texture_mean=float(texture_mean)
	
	perimeter_mean = request.POST.get("perimeter_mean")
	perimeter_mean=float(perimeter_mean)
	
	area_mean = request.POST.get("area_mean")
	area_mean=float(area_mean)
	
	smoothness_mean = request.POST.get("smoothness_mean")
	smoothness_mean=float(smoothness_mean)
	
	compactness_mean = request.POST.get("compactness_mean")
	compactness_mean=float(compactness_mean)
	
	concavity_mean= request.POST.get("concavity_mean")
	concavity_mean=float(concavity_mean)
	
	concavepoints_mean= request.POST.get("concavepoints_mean")
	concavepoints_mean=float(concavepoints_mean)
	
	symmetry_mean= request.POST.get("symmetry_mean")
	symmetry_mean=float(symmetry_mean)
	
	fractal_dimension_mean= request.POST.get("fractal_dimension_mean")
	fractal_dimension_mean=float(fractal_dimension_mean)
	
	radius_se= request.POST.get("radius_se")
	radius_se=float(radius_se)

	texture_se = request.POST.get("texture_se")
	texture_se =float(texture_se )
	
	perimeter_se= request.POST.get("perimeter_se")
	perimeter_se=float(perimeter_se)
	
	area_se= request.POST.get("area_se")
	area_se=float(area_se)
	
	smoothness_se= request.POST.get("smoothness_se")
	smoothness_se=float(smoothness_se)
	
	compactness_se= request.POST.get("compactness_se")
	compactness_se=float(compactness_se)
	
	concavity_se= request.POST.get("concavity_se")
	concavity_se=float(concavity_se)
	
	concavepoints_se= request.POST.get("concavepoints_se")
	concavepoints_se=float(concavepoints_se)
	
	symmetry_se = request.POST.get("symmetry_se")
	symmetry_se =float(symmetry_se )
	
	fractal_dimension_se= request.POST.get("fractal_dimension_se")
	fractal_dimension_se=float(fractal_dimension_se)
	
	radius_worst= request.POST.get("radius_worst")
	radius_worst=float(radius_worst)
	
	texture_worst= request.POST.get("texture_worst")
	texture_worst=float(texture_worst)
	
	perimeter_worst= request.POST.get("perimeter_worst")
	perimeter_worst=float(perimeter_worst)
	  
	area_worst= request.POST.get("area_worst")
	area_worst=float(area_worst)
	
	smoothness_worst = request.POST.get("smoothness_worst")
	smoothness_worst =float(smoothness_worst )
	compactness_worst = request.POST.get("compactness_worst")
	compactness_worst =float(compactness_worst )
	
	concavity_worst = request.POST.get("concavity_worst")
	concavity_worst =float(concavity_worst )
	
	concavepoints_worst= request.POST.get("concavepoints_worst")
	concavepoints_worst=float(concavepoints_worst)
	
	symmetry_worst= request.POST.get("symmetry_worst")
	symmetry_worst=float(symmetry_worst)
	
	fractal_dimension_worst= request.POST.get("fractal_dimension_worst")
	fractal_dimension_worst=float(fractal_dimension_worst)
	
	X_test=[perimeter_worst,concavepoints_worst,concavepoints_mean,concavity_mean,smoothness_worst]
	res = model_cancer.predict([X_test])[0]
	res =float(res)
	if res >0.5:
		resultat= 'Malign'
	else:
		resultat = 'Benign'
	print(res)
	
	return JsonResponse({'result': resultat})





def predict_blood(request):
	
	filename= os.path.join(os.getcwd(),'blood_donation_with_gbm.pkl')
	model_blood= joblib.load(filename)


	Months_since_Last_Donation =request.POST.get("Months_since_Last_Donation")
	Months_since_Last_Donation=float(Months_since_Last_Donation)
	
	Number_donations = request.POST.get("Number_donations")
	Number_donations=float(Number_donations)
	
	Total_Volume_Donated = request.POST.get("Total_Volume_Donated")
	Total_Volume_Donated=float(Total_Volume_Donated)
	
	
	Months_since_First_Donation= request.POST.get("Months_since_First_Donation")
	Months_since_First_Donation=float(Months_since_First_Donation)
	
	last_donation_mean= 9.439236
	last_donation_std= 8.175454
	Months_since_Last_Donation =(Months_since_Last_Donation-last_donation_mean)/last_donation_std
	
	numb_donation_mean=5.427083
	numb_donation_std=5.740010
	Number_donations = (Number_donations-numb_donation_mean)/numb_donation_std
	
	first_donation_mean=34.050347
	first_donation_std=24.227672
	Months_since_First_Donation= (Months_since_First_Donation-first_donation_mean)/first_donation_std
	
	X_test = [[Months_since_Last_Donation,Number_donations,Months_since_First_Donation]]
	
	res= list(model_blood.predict(X_test))[0]
	res =int(res)
	if res == 1:
		result ='Yes will come back'
	else:
		result='No'

	return JsonResponse({'result': result})

def predict_diabetes(request):
	filename1 = os.path.join(os.getcwd(),'pima_dataset_lgbm_final.txt')
	model_diabete= lgbm.Booster(model_file=filename1)
	print("model")

	Pregnancies = request.POST.get("Pregnancies")
	Pregnancies=float(Pregnancies)
	print(Pregnancies)
	Glucose = request.POST.get("Glucose")
	Glucose=float(Glucose)
	print(Glucose)
	BloodPressure = request.POST.get("BloodPressure")
	BloodPressure=float(BloodPressure)
	SkinThickness= request.POST.get("SkinThickness")
	SkinThickness=float(SkinThickness)
	Insulin = request.POST.get("Insulin")
	Insulin=float(Insulin)
	BMI= request.POST.get("BMI")
	BMI=float(BMI)
	DiabetesPedigreeFunction = request.POST.get("DiabetesPedigreeFunction")
	DiabetesPedigreeFunction=float(DiabetesPedigreeFunction)
	print('DiabetesPedigreeFunction',DiabetesPedigreeFunction)
	Age = request.POST.get("Age")
	Age=float(Age)
	test=[Age,DiabetesPedigreeFunction,BloodPressure,BMI,Insulin,Pregnancies,Glucose,SkinThickness]
	#test=[Age,DiabetesPedigreeFunction, BloodPressure, BMI ,Pregnancies, Insulin ,Glucose, SkinThickness]
	res = model_diabete.predict([test])[0]
	if res >0.5:
		resultat= 'Has Diabete'
	else:
		resultat = "No Diabete"
	
	print(resultat)
	return JsonResponse({'result': resultat})



def predict_fare(request):
	filename = os.path.join(os.getcwd(),'taxi_model_new_york.txt')
	model_taxi= lgbm.Booster(model_file=filename)
	pickup_datetime = request.POST.get("pickup_datetime")
	
	if pickup_datetime == 'now':
		pickup_datetime= dt.datetime.now()
		
	else:
		pickup_datetime=dt.datetime.strptime(pickup_datetime, '%Y-%m-%d %H:%M:%S UTC')
	
	passenger_count = request.POST.get("passenger_count")
	passenger_count= int(passenger_count)
	pickup_latitude = request.POST.get("pickup_latitude")
	pickup_latitude=float(pickup_latitude)
	pickup_longitude = request.POST.get("pickup_longitude")
	pickup_longitude=float(pickup_longitude)
	dropoff_latitude = request.POST.get("dropoff_latitude")
	dropoff_latitude = float(dropoff_latitude)
	dropoff_longitude= request.POST.get("dropoff_longitude")
	dropoff_longitude = float(dropoff_longitude)
	pickup_year = pickup_datetime.year
	month = pickup_datetime.month
	months=['January','February','March','April','May','June','July','August','September','October','November','December']
	if month == 1:
		January=1
	else:
		January=0
	if month == 2:
		February=1
	else:
		February=0
	if month == 3:
		March=1
	else:
		March=0
	if month == 4:
		April=1
	else:
		April=0
	if month == 5:
		May=1
	else:
		May=0
	if month == 6:
		June=1
	else:
		June=0
	if month == 7:
		July=1
	else:
		July=0
	if month == 8:
		August=1
	else:
		August=0
	if month == 9:
		September=1
	else:
		September=0
	if month == 10:
		October=1
	else:
		October=0
	if month == 11:
		November=1
	else:
		November=0
	if month== 12:
		December=1
	else:
		December=0
	day_of_week= pickup_datetime.weekday()
	weeks=['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
	if day_of_week == 0:
		Monday=1
	else:
		Monday=0
	if day_of_week == 1:
		Tuesday=1
	else:
		Tuesday=0
	if day_of_week == 2:
		Wednesday=1
	else:
		Wednesday=0
	if day_of_week == 3:
		Thursday=1
	else:
		Thursday=0
	if day_of_week == 4:
		Friday=1
	else:
		Friday=0
	if day_of_week == 5:
		Saturday=1
	else:
		Saturday=0
	if day_of_week == 6:
		Sunday=1
	else:
		Sunday=0

	if (day_of_week == 5 or day_of_week == 6):
		is_weekend=1
		is_weekday=0
	else:
		is_weekend=0
		is_weekday =1
	hour= pickup_datetime.hour
	if hour == 0:
		hour1=1
	else:
		hour1=0
	if hour == 1:
		hour2=1
	else:
		hour2=0
	if hour == 2:
		hour3=1
	else:
		hour3=0
	if hour == 3:
		hour4=1
	else:
		hour4=0
	if hour == 4:
		hour5=1
	else:
		hour5=0
	if hour == 5:
		hour6=1
	else:
		hour6=0
	if hour == 6:
		hour7=1
	else:
		hour7=0
	if hour == 7:
		hour8=1
	else:
		hour8=0
	if hour == 8:
		hour9=1
	else:
		hour9=0
	if hour== 9:
		hour10=1
	else:
		hour10=0
	if hour == 10:
		hour11=1
	else:
		hour11=0
	if hour== 11:
		hour12=1
	else:
		hour12=0
	if hour == 12:
		hour13=1
	else:
		hour13=0
	if hour == 13:
		hour14=1
	else:
		hour14=0
	if hour == 14:
		hour15=1
	else:
		hour15=0
	if hour == 15:
		hour16=1
	else:
		hour16=0
	if hour == 16:
		hour17=1
	else:
		hour17=0
	if hour  == 17:
		hour18=1
	else:
		hour18=0
	if hour== 18:
		hour19=1
	else:
		hour19=0
	if hour == 19:
		hour20=1
	else:
		hour20=0
	if hour == 20:
		hour21=1
	else:
		hour21=0
	if hour == 21:
		hour22=1
	else:
		hour22=0
	if hour == 22:
		hour23=1
	else:
		hour23=0
	if hour== 23:
		hour24=1
	else:
		hour24=0

	

	if (hour >= 6 and hour <= 9) or (hour >= 16 and hour <= 20):
		peak_hour =1
	else:
		peak_hour=0
	trip_distance = distance_trip(pickup_latitude,pickup_longitude, dropoff_latitude,dropoff_longitude)

	airports = {'JFK': (-73.78,40.643),'LGA': (-73.87, 40.77),'EWR' : (-74.18, 40.69),'MNT':(-73.97,40.7831),'Cenpark':(-73.96,40.77)}
	pickup_dist_JFK = distance_trip(pickup_latitude,pickup_longitude,airports['JFK'][1],airports['JFK'][0])
	
	pickup_dist_LGA = distance_trip(pickup_latitude,pickup_longitude,airports['LGA'][1],airports['LGA'][0])
	
	pickup_dist_EWR = distance_trip(pickup_latitude,pickup_longitude,airports['EWR'][1],airports['EWR'][0])
	
	pickup_dist_MNT = distance_trip(pickup_latitude,pickup_longitude,airports['MNT'][1],airports['MNT'][0])
	
	pickup_dist_Cenpark = distance_trip(pickup_latitude,pickup_longitude,airports['Cenpark'][1],airports['Cenpark'][0])

	dropoff_dist_JFK = distance_trip(dropoff_latitude,dropoff_longitude,airports['JFK'][1],airports['JFK'][0])
	
	dropoff_dist_LGA = distance_trip(dropoff_latitude,dropoff_longitude,airports['LGA'][1],airports['LGA'][0])
	
	dropoff_dist_EWR = distance_trip(dropoff_latitude,dropoff_longitude,airports['EWR'][1],airports['EWR'][0])
	
	dropoff_dist_MNT = distance_trip(dropoff_latitude,dropoff_longitude,airports['MNT'][1],airports['MNT'][0])
	
	dropoff_dist_Cenpark = distance_trip(dropoff_latitude,dropoff_longitude,airports['Cenpark'][1],airports['Cenpark'][0])
	
	direction= direction_angle(pickup_latitude, pickup_longitude,dropoff_latitude,dropoff_longitude)
	
	pickup_latitude = np.deg2rad(pickup_latitude)
	pickup_longitude = np.deg2rad(pickup_longitude)
	dropoff_latitude = np.deg2rad(dropoff_latitude)
	dropoff_longitude = np.deg2rad(dropoff_longitude)
	direction = np.deg2rad(direction)
	list_test=[pickup_longitude,pickup_latitude,dropoff_longitude,dropoff_latitude,passenger_count,pickup_dist_JFK,
		dropoff_dist_JFK,pickup_dist_LGA,dropoff_dist_LGA,pickup_dist_EWR,dropoff_dist_EWR,
		pickup_dist_MNT	,dropoff_dist_MNT,pickup_dist_Cenpark,dropoff_dist_Cenpark,
			trip_distance,direction,peak_hour,	is_weekend,	is_weekday,Friday,Monday,Saturday,Sunday,
			Thursday,Tuesday,Wednesday,April,August,December,February,January,	July,
				June,March,May,November,October,September,hour1,hour10,hour11,hour12,hour13,hour14,hour15,
					hour16,hour17,hour18,hour19,hour2,hour20,hour21,hour22	,hour23	,hour24	,hour3,
					hour4,hour5,hour6,hour7,hour8,hour9]
	res = model_taxi.predict([list_test])[0]
	res =int(float(res))
	
	


	
	
	return JsonResponse({'result': res})
    

    

    
    