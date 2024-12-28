import numpy as np
import matplotlib.pyplot as plt
from math import sqrt

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
rho       = 1.184          # density kg / m^3
mu        = 0.066564       # viscosity Kg / m*h
cp        = 1007           # specific heat J / kg*K
tau       = 0.9            # Transmittance 
alpha_air = 0.1            # aborbance
rho_air   = 0              # reflectance
Pr        = 0.7296         # Prandtl Number
v_air     = 10*1000/3600   # velocity air m/s
k_air     = 0.02551        # Conductivity W / m*K

## ----- Green Area -----
# ET_0
Rn     = Gr*3600/10**(6) # Radiación Neta  MJ / m^2*h
G      = 0.1*Rn           # Densidad flujo calor del suelo  MJ / m^2*h
Delta  = 0.1447326371     # ** (T) pendiente de la curva de presión de saturación de vapor  kPa / °C
gamma  = 0.000665         # constante psicrométrica  kPa / °C
e_0    = 2.338281271      # ** (T) presión de saturación de vapor a temperatura del aire T kPa
e_a    = 1.286054699      # promedio horario de la presión real de vapor  kPa
u_2    = v_air            # promedio horario de la velocidad del viento  m/s

## ----- Pasos -----
dt = 0.1  
dx = 2*v_air*dt
dy = (4*k_air*dt)**0.5

# --- Vectores y matrices ---
x = np.arange(0, 100/dx, dx)  # Vector posición en el largo del río [m]
t = np.arange(0, 1800 + dt, dt)  # Vector tiempo [s]
y = np.arange(0, 100/dx, dy)  # Vector posición en el ancho del río [m]

A = np.zeros((len(x), len(y), len(t)))  # Matriz de ceros (mallado)

# --- Condición inicial ---
for i in range(len(x)):
    for j in range(len(y)):
        A[i, j, 1] = 298   #K

# --- Completar la matriz A con diferencias finitas ---
#for k in range(len(t) - 1):  # Iterar en el tiempo
#    for j in range(len(y)):
#        for i in range(1, len(x)):  # Empezar desde i=1 para evitar problemas con i-1
#            A[i, j, k + 1] = A[i, j, k] - 
