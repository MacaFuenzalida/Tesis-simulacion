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

h_out     = 0.015292       #  W / m^2 * K

## ----- Temperatura -----
T_soil = 297.15            # K
T_air  = 298.15            # K


## ----- Green Area -----
# ET_0
Rn     = Gr*3600/10**(6)  # Radiación Neta  MJ / m^2*h
G      = 0.1*Rn           # Densidad flujo calor del suelo  MJ / m^2*h
Delta  = 0.1447326371     # ** (T) pendiente de la curva de presión de saturación de vapor  kPa / °C
gamma  = 0.000665         # constante psicrométrica  kPa / °C
e_0    = 2.338281271      # ** (T) presión de saturación de vapor a temperatura del aire T kPa
e_a    = 1.286054699      # promedio horario de la presión real de vapor  kPa
u_2    = v_air            # promedio horario de la velocidad del viento  m/s

## ----- Pasos -----
dt = 1  
dx = 2*v_air*dt
dy = (4*k_air*dt)**0.5

print(f'dx = {dx}')
print(f'dy = {dy}')

# --- Vectores y matrices ---
x = np.arange(0, 100/dx, dx)  # Vector posición en el largo del río [m]
t = np.arange(0, 100   , dt)  # Vector tiempo [s]
y = np.arange(0, 100/dy, dy)  # Vector posición en el ancho del río [m]

# %% Inicialización de la malla
T = np.ones((len(x), len(y),len(t))) * 293  # Temp. inicial homogénea (293 K)

# Condicion de borde más caliente.
for k in range(0, int(len(t)/2)):
    for j in range(0, len(y)):
        T[0,j,k]=300     


# --- Completar la matriz A con diferencias finitas ---
for k in range(1, len(t) - 1):  # Iterar en el tiempo
    for j in range(1, len(y) - 1):
        for i in range(1, len(x)):  # Empezar desde i=1 para evitar problemas con i-1
            #Término radiación del suelo
            rad_suelo = A*e_soil*sigma*(T_soil**4 - T[i, j, k]**4)

            # Albedo , absorbancias, reflectancia
            rad_solar = a_soil*tau*Gr*A - alpha_soil*tau*Gr*A + alpha_air*Gr*A - rho_air*Gr*A

            # Covection
            convec = - h_out*A*(T[i,j,k] - T_air)
        
            # Difusión
            advec = -rho*cp*A*v_air*(1/dx)*(T[i,j,k] - T[i-1 , j, k])

            # Conduction 
            cond = k_air*V*(1/dy**2)*(T[i, j+1, k] - 2*T[i, j, k] + T[i, j-1, k])         

            # TOTAL
            T[i, j, k + 1] = T[i, j, k] + dt/(rho*cp*V)*(rad_suelo + rad_solar + convec + advec + cond)

# %% Graficar y animar
fig, ax = plt.subplots()
cax = ax.imshow(T[:, :, 0].T, extent=[0, len(x), 0, len(y)], origin='lower', aspect='auto', cmap='hot')
cbar = plt.colorbar(cax, ax=ax, label='T [K]')

ax.set_xlabel('Largo del dominio (x) [m]')
ax.set_ylabel('Ancho del dominio (y) [m]')

for k in range(len(t)):
    cax.set_data(T[:, :, k].T)  # Actualizar solo los datos de la gráfica
    cax.set_clim(vmin=T.min(), vmax=T.max())  # Asegurar que los colores estén bien escalados
    plt.title(f'Animación en el tiempo: {t[k]} [s]')
    plt.pause(0.1)  # Pausa para que se vea la animación

plt.show()  # Mantener la animación visible al final


# %% Graficar y animar
#fig, ax = plt.subplots()
#cax = ax.imshow(T[:, :, 0], extent=[0, len(x), 0, len(y)], origin='lower', aspect='auto', cmap='hot')
#cbar = plt.colorbar(cax, ax=ax, label='T [K]')

#ax.set_ylabel('Posición en el largo del río [m]')
#ax.set_xlabel('Posición en el ancho del río [m]')

#for k in range(len(t)):
#    cax.set_data(T[:, :, k])  # Actualizar solo los datos de la gráfica
#    cax.set_clim(vmin=T.min(), vmax=T.max())  # Asegurar que los colores estén bien escalados
#    plt.title(f'Animación en el tiempo: {t[k]} [s]')
#    plt.pause(0.1)  # Pausa para que se vea la animación

#plt.show()  # Mantener la animación visible al final