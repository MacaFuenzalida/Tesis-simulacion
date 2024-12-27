import numpy as np
import matplotlib.pyplot as plt

### -------- Parámetros --------

## --- Constantes ---
sigma = 5.67*10**(-8)  # Stefan-Boltzmann constant W / m^2*K^4
V = 1                 # volume of the parallelepiped m^3 
A = 1                 # Transfer Area m^2
A_b = 1               # Basal Green Area m^2
L = 1                 # Characteristic length m
Gr = 709.84           # effective incident solar radiation W / m^2

## --- Soil Proporties ---
e_soil = 0.9          # emissivity
a_soil = 0.09         # albedo
alpha_soil = 0.01     # absorbance

## --- Air Proporties ---
rho = 1.184           # density kg / m^3
mu  = 0.066564        # viscosity Kg / m*h
cp  = 1007            # specific heat J / kg*K
tau = 0.9             # Transmittance 
alpha_air = 0.1       # aborbance
rho_air   = 0         # reflectance
Pr   = 0.7296         # Prandtl Number
v_air = 10*1000/3600  # velocity air m/s
k_air = 0.02551       # Conductivity W / m*K

## ----- Green Area -----
# ET_0





