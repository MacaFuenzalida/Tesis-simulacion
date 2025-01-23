import numpy as np
import matplotlib.pyplot as plt
from math import sqrt, radians, sin, cos

### -------- Viento --------
# Diccionario para las direcciones del viento y sus ángulos en grados: Rosa de los Vientos
wind_directions = {
    "N": 0, "NNE": 22.5, "NE": 45, "ENE": 67.5,
    "E": 90, "ESE": 112.5, "SE": 135, "SSE": 157.5,
    "S": 180, "SSW": 202.5, "SW": 225, "WSW": 247.5,
    "W": 270, "WNW": 292.5, "NW": 315, "NNW": 337.5}

# Función para calcular las componentes de la velocidad
def calculate_wind_components(v_air, direction):

    # Convertir dirección de texto a ángulo
    if direction not in wind_directions:
        raise ValueError(f"Dirección de viento '{direction}' no válida.")
    
    theta = np.radians(wind_directions[direction])  # Convertir ángulo a radianes
    
    # Descomponer la velocidad en componentes
    v_x = v_air * np.sin(theta)  # Componente en el eje x
    v_y = v_air * np.cos(theta)  # Componente en el eje y
    
    return v_x, v_y


v_air = 10        # Velocidad del viento en m/s
direction = "N"  # Dirección del viento (por ejemplo: "SW")

v_x, v_y = calculate_wind_components(v_air, direction)
print(f"Velocidad descompuesta: v_x = {v_x:.2f} m/s, v_y = {v_y:.2f} m/s")


### -------- Parámetros --------

## --- Constantes ---
sigma = 5.67*10**(-8)  # Stefan-Boltzmann constant W / m^2*K^4
V = 1                 # volume of the parallelepiped m^3 
A = 1                 # Transfer Area m^2
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
k_air     = 0.02551        # Conductivity W / m*K

h_out     = 0.015292       #  W / m^2 * K

## ----- Temperatura -----
T_soil = 297.15            # K
T_air  = 298.15            # K
T_av   = 293.15            # K

## ----- Green Area -----
# ET_0
Rn     = Gr*3600/10**(6)  # Radiación Neta  MJ / m^2*h
G      = 0.1*Rn           # Densidad flujo calor del suelo  MJ / m^2*h
Delta  = 0.1447326371     # ** (T) pendiente de la curva de presión de saturación de vapor  kPa / °C
gamma  = 0.000665         # constante psicrométrica  kPa / °C
e_0    = 2.338281271      # ** (T) presión de saturación de vapor a temperatura del aire T kPa
e_a    = 1.286054699      # promedio horario de la presión real de vapor  kPa
u_2    = v_air            # promedio horario de la velocidad del viento  m/s

# Evapotranspiración de referencia
ET_0 = (0.408*Delta*(Rn - G) + gamma*(37/T_av)*u_2*(e_0-e_a))/(Delta + gamma*(1 + 0.34*u_2))
#print(f'ET_0 = {ET_0}')

# Coeficiente: basal cultivos no estresados
Kcb  = 0.85                # Basal cultivos no estresados
Ke   = 0.35                # Evaporación del suelo
Ks   = 1.025641026         # Factor reducción tranpiración

# Evaporación
rho_w  = 997.13            # densidad del agua kg/m^3
lamda  = 2441.7            # Calor latente de vaporización kJ/kg 
ET_caj = (Ks*Kcb+Ke)*ET_0  # Evapotranspiración ajustada mm/h

### ---- PASOS -----
dt = 1  
if abs(v_x) >= abs(v_y):  # Viento más dominante en el eje x
    dx = 2 * abs(v_x) * dt
    dy = (4 * k_air * dt)**0.5
else:  # Viento más dominante en el eje y
    dx = (4 * k_air * dt)**0.5
    dy = 2 * abs(v_y) * dt

print(f'{dx}')
print(f'{dy}')

# --- Vectores y matrices ---
x = np.arange(0, 60, dx)  # Vector posición en el largo del río [m]
t = np.arange(0, 50, dt)  # Vector tiempo [s]
y = np.arange(0, 600, dy)  # Vector posición en el ancho del río [m]

# %% Inicialización de la malla
T = np.ones((len(x), len(y),len(t))) * 295  # Temp. inicial homogénea (293 K)

# Condicion de borde más caliente.
#for k in range(0, int(len(t)/2)):
#    for j in range(0, len(y)):
#        T[0,j,k]=300     

# ----- Condición de borde Área Verde ------

# Area basal Area Verde
A_b =  dx*3*dy                # Basal Green Area m^2
# Flujo agua evaporada 
m_w        = ET_0*10**(-3)*A_b*rho_w/3500  # kg/s
#print(f'm_w = {m_w}')

## Matriz de ceros 
AV = np.zeros((len(x),len(y)))

ancho_AV  = 3      #Dimensiones área verde en pixceles
altura_AV = 3      #Dimensiones área verde en pixceles

## Posición área verde
# Centrada
center_x = len(x) // 2
center_y = len(y) // 2

# Establecer los límites del área verde
start_x = center_x - ancho_AV // 2
end_x = center_x + ancho_AV // 2 + 1
start_y = center_y - altura_AV // 2
end_y = center_y + altura_AV // 2 + 1

# Asignar la posición del área verde
AV[start_x:end_x, start_y:end_y] = 1


# Borde Área Verde
# Crear una nueva matriz de ceros con las mismas dimensiones que AV
periferia_AV = np.zeros_like(AV)

# Iterar para encontrar las celdas fuera del área verde en contacto con el área verde
for i in range(1, AV.shape[0] - 1):  # Evitar bordes externos
    for j in range(1, AV.shape[1] - 1):
        if AV[i, j] == 0:  # Si está fuera del área verde
            # Verificar si algún vecino inmediato es parte del área verde
            if (AV[i - 1, j] == 1 or AV[i + 1, j] == 1 or
                AV[i, j - 1] == 1 or AV[i, j + 1] == 1):
                periferia_AV[i, j] = 1  # Marcar como parte de la periferia


# Determinar el número de celdas en la periferia del área verde
num_celdas_periferia = np.sum(periferia_AV)  # Contar las celdas con valor 1 en periferia_AV
if num_celdas_periferia > 0:
    m_wn = m_w / num_celdas_periferia  # Distribuir m_w entre las celdas de la periferia
else:
    raise ValueError("El número de celdas en la periferia es cero.")


## DIFERENCIAS FINITAS

for k in range(1, len(t) - 1):  # Iterar en el tiempo

    # Mallas auxiliares
    T_x = np.copy(T[:, :, k])  # Para advección en x y difusión en y
    T_y = np.copy(T[:, :, k])  # Para advección en y y difusión en x

    # ----- Matriz T_x (advección en x, difusión en y) -----
    for j in range(1, len(y) - 1):
        for i in range(1, len(x) - 1):
            # Omitir el cálculo en las celdas del área verde
            if AV[i, j] == 1:
                continue

            # Radiación del suelo
            rad_suelo = A * e_soil * sigma * (T_soil**4 - T[i, j, k]**4)

            # Radiación solar (albedo, absorbancia, reflectancia)
            rad_solar = (a_soil * tau * Gr * A - alpha_soil * tau * Gr * A + alpha_air * Gr * A - rho_air * Gr * A)

            # Convección
            convec = -h_out * A * (T[i, j, k] - T_air)

            # Advección en x
            if v_x > 0:  # Flujo hacia la derecha
                advec_x = -rho * cp * A * v_x * (T[i, j, k] - T[max(i-1, 0), j, k]) / dx
            elif v_x < 0:  # Flujo hacia la izquierda
                advec_x = -rho * cp * A * v_x * (T[min(i+1, len(x)-1), j, k] - T[i, j, k]) / dx
            else:
                advec_x = 0

            # Difusión en y
            diff_y = k_air * V * (T[i, j+1, k] - 2 * T[i, j, k] + T[i, j-1, k]) / dy**2

            # Actualizar T_x
            T_x[i, j] = (T[i, j, k] + dt / (rho * cp * V) * (advec_x + diff_y + rad_suelo + rad_solar + convec))

    # ----- Matriz T_y (advección en y, difusión en x) -----
    for j in range(1, len(y) - 1):
        for i in range(1, len(x) - 1):
            # Omitir el cálculo en las celdas del área verde
            if AV[i, j] == 1:
                continue

            # Radiación del suelo
            rad_suelo = A * e_soil * sigma * (T_soil**4 - T[i, j, k]**4)

            # Radiación solar (albedo, absorbancia, reflectancia)
            rad_solar = (a_soil * tau * Gr * A - alpha_soil * tau * Gr * A + alpha_air * Gr * A - rho_air * Gr * A)

            # Convección
            convec = -h_out * A * (T[i, j, k] - T_air)

            # Advección en y
            if v_y > 0:  # Flujo hacia arriba
                advec_y = -rho * cp * A * v_y * (T[i, j, k] - T[i, max(j-1, 0), k]) / dy
            elif v_y < 0:  # Flujo hacia abajo
                advec_y = -rho * cp * A * v_y * (T[i, min(j+1, len(y)-1), k] - T[i, j, k]) / dy
            else:
                advec_y = 0

            # Difusión en x
            diff_x = k_air * V * (T[i+1, j, k] - 2 * T[i, j, k] + T[i-1, j, k]) / dx**2

            # Actualizar T_y
            T_y[i, j] = (T[i, j, k] + dt / (rho * cp * V) * (advec_y + diff_x + rad_suelo + rad_solar + convec))

    # ----- Suma ponderada según la dirección del viento -----
    # Calcular los pesos basados en la dirección del viento
    theta = radians(wind_directions[direction])
    weight_x = abs(cos(theta))  # Ponderación para T_x
    weight_y = abs(sin(theta))  # Ponderación para T_y

    # Normalizar los pesos
    #total_weight = weight_x + weight_y
    #weight_x /= total_weight
    #weight_y /= total_weight

    # Combinar las matrices auxiliares para obtener T[i, j, k+1]
    T[:, :, k+1] = weight_x * T_x + weight_y * T_y

    # Condición de borde en la periferia del área verde
    for i in range(len(x)):
        for j in range(len(y)):
            if periferia_AV[i, j] == 1:  # Si estamos en una celda de la periferia del área verde
                T[i, j, k + 1] += -m_wn * lamda  # Agregar el calor de cambio de fase

# Mantener la temperatura fija en el área verde
for i_av in range(AV.shape[0]):
    for j_av in range(AV.shape[1]):
        if AV[i_av, j_av] == 1:  # Si es parte del área verde
            T[i_av, j_av, :] = T_av  # Sobrescribir con temperatura fija


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

