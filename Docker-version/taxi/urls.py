from .import views
from django.urls import path

urlpatterns = [
	path('taxi/', views.index_taxi, name='taxi'),
	path('diabetes/', views.index_diabetes, name='diabetes_home'),
	path('blood/', views.index_blood, name='blood_home'),
	path('wisconsin_cancer/', views.index_wiscon, name='wiscon_home'),

	path('predict_fare/', views.predict_fare, name='predict_fare'),
	path('predict_diabetes/', views.predict_diabetes, name='predict_diabetes'),
	path('predict_blood/', views.predict_blood, name='predict_blood'),
	path('predict_cancer/', views.predict_cancer, name='predict_cancer'),
  ]
 