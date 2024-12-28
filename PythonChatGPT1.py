import numpy as np
import matplotlib.pyplot as plt

# %% Parámetros físicos
sigma = 5.67 * 10**(-8)  # Constante de Stefan-Boltzmann [W/m^2 K^4]
V = 1                    # Volumen [m^3]
A = 1                    # Área de transferencia [m^2]
e_soil = 0.9             # Emisividad del suelo
Gr = 709.84              # Radiación solar incidente [W/m^2]
a_soil = 0.09            # Albedo del suelo
alpha_soil = 0.01        # Absorción del suelo
alpha_sup = 0.1          # Absorción de la superficie
rho = 1.2                # Densidad del aire [kg/m^3]
c_p = 1005               # Calor específico del aire [J/kg K]
k = 0.025                # Conductividad térmica del aire [W/m K]
v_air = 10 * 1000 / 3600 # Velocidad del aire [m/s]
h_out = 10               # Coef. transferencia de calor [W/m^2 K]
T_aire = 293             # Temp. del aire [K]
tau = 0.9                # Transmitancia atmosférica
rho_air = 0.0            # Reflectancia del aire

# %% Discretización y malla
dt = 0.1  
dx = 2*v_air*dt
dy = (4*k*dt)**0.5

nx = int(100 / dx)        # Puntos en x
ny = int(100 / dy)        # Puntos en y
nt = 100        # Pasos de tiempo

max_temp = 350  # Límite superior de temperatura [K]
min_temp = 273  # Límite inferior de temperatura (0°C en Kelvin)

# %% Inicialización de la malla
T = np.ones((nx, ny, nt)) * 293  # Temp. inicial homogénea (293 K)

# %% Iteración temporal con diferencias finitas
for k in range(nt - 1):
    for i in range(1, nx - 1):  # Evitar bordes
        for j in range(1, ny - 1):
            T_suelo = T[i, j, k]  # Temp. actual del suelo
            
            # Término de radiación (previniendo overflow)
            rad = A * e_soil * sigma * (np.clip(T_suelo, min_temp, max_temp)**4 - np.clip(T[i, j, k], min_temp, max_temp)**4)
            
            # Término de radiación solar neta (asegurar que sea positivo)
            rad_solar = max(A * (a_soil * tau * Gr - alpha_soil * tau * Gr + alpha_sup * Gr - rho_air * Gr), 0)
            
            # Término difusivo (limitado para evitar valores extremos)
            diff_y = k * (T[i, j+1, k] - 2 * T[i, j, k] + T[i, j-1, k]) / dy**2
            diff_y = np.clip(diff_y, -10, 10)
            
            # Término advectivo (advección en x)
            adv_x = - v_air * (T[i, j, k] - T[i-1, j, k]) / dx
            
            # Término convectivo con el aire
            conv = - h_out * (T[i, j, k] - T_aire) / (rho * c_p)
            
            # Actualización de temperatura (Euler explícito)
            T[i, j, k+1] = np.clip(T[i, j, k] + dt * (rad + rad_solar + diff_y + adv_x + conv), min_temp, max_temp)

# %% Graficar y animar resultado
fig, ax = plt.subplots()
cax = ax.imshow(T[:, :, 0], extent=[0, nx*dx, 0, ny*dy], origin='lower', cmap='hot')
cbar = plt.colorbar(cax, ax=ax, label='Temperatura [K]')

for k in range(nt):
    cax.set_data(T[:, :, k])  # Actualizar datos de la gráfica
    cax.set_clim(vmin=T.min(), vmax=T.max())  # Ajustar escala de color
    plt.title(f'Tiempo: {k*dt:.1f} s')
    plt.pause(0.05)

plt.show()
