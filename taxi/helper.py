import numpy as np

def distance_trip(latitude1,longitude1,latitude2,longitude2):
  r= 6373 # earth radius
  latitude1 = np.deg2rad(latitude1)
  longitude1 = np.deg2rad(longitude1)
  latitude2 = np.deg2rad(latitude2)
  longitude2= np.deg2rad(longitude2)
  dlat = latitude2 - latitude1
  dlon = longitude2 - longitude1
  a = np.sin(dlat/2)**2 + np.cos(latitude1) * np.cos(latitude2) * np.sin(dlon/2)**2
  c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
  distance = r*c
  return distance
def direction_angle(latitude1,longitude1,latitude2,longitude2):
  dlon = longitude2 - longitude1
  x = np.cos(latitude2)* np.sin(dlon)
  y= np.cos(latitude1)* np.sin(latitude2) - np.sin(latitude1)*np.cos(latitude2) * np.cos(dlon)
  beta_en_radians = np.arctan2(x,y)
  beta_en_degres = np.rad2deg(beta_en_radians)
  return beta_en_degres
