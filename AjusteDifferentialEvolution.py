import numpy as np
import matplotlib.pyplot as plt
from math import sqrt, radians, sin, cos

# Función para leer el archivo de velocidades
def read_wind_data(filename):
    data = np.loadtxt(filename, delimiter=',', dtype=str)  # Leer datos como texto
    v_air = data[:, 0].astype(float)  # Convertir la primera columna a números
    directions = data[:, 1]  # La segunda columna ya es texto
    return v_air, directions

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

# Cargar datos de viento
filename = "wind_data.txt"
v_air_list, direction_list = read_wind_data(filename)

# Calcular vx y vy para cada hora
dx_list, dy_list = [], []
v_x_list, v_y_list = [], []

k_air     = 0.02551        # Conductivity W / m*K
constants = [k_air]
# ------- Parámetros a ajustar : Adivinanzas inicial --------
#Kcb   = 0.85                # Basal cultivos no estresados  (vegetal) ** Sospechoso
#Ke    = 0.35                # Evaporación del suelo                   ** Sospechoso
#Ks    = 1.025641026         # Factor reducción tranpiración (vegetal) ** Sospechoso
Kc    = 0.93                # Césped (Tee and Green)
h_out = 0.015292            #  W / m^2 * K                            ** Sospechoso

params = [Kc, h_out]
# Ojo, lo más probable sólo ajuste Kc=(Ks*Kcb+Ke)

## ----Pasos -----
## Separar cuando estoy en los ejes y cuando no y desde ahi elegir los dx y dy que cumplan todas las inecuaciones
dt = 0.1  

for i in range(len(v_air_list)):
    v_x, v_y = calculate_wind_components(v_air_list[i], direction_list[i])
    v_x_list.append(v_x)
    v_y_list.append(v_y)

# Cálculo de dx y dy considerando estabilidad
def cumple_estabilidad(dx, dy, dt, v_x, v_y, k_air):
    nu_x = abs(v_x) * dt / dx
    r_x = k_air * dt / dx**2
    nu_y = abs(v_y) * dt / dy
    r_y = k_air * dt / dy**2
    return (nu_x <= 0.5 and r_x <= 0.25 and 2 * nu_x**2 <= nu_x + r_y and
            nu_y <= 0.5 and r_y <= 0.25 and 2 * nu_y**2 <= nu_y + r_x)

# Generar todos los posibles dx y dy
dx_options = [2 * abs(v_x) * dt for v_x in v_x_list] + [(4 * k_air * dt) ** 0.5] * len(v_x_list)
dy_options = [2 * abs(v_y) * dt for v_y in v_y_list] + [(4 * k_air * dt) ** 0.5] * len(v_y_list)

# Selección de dx y dy que cumplen estabilidad
dx_final, dy_final = None, None

for dx in sorted(dx_options, reverse=True):
    for dy in sorted(dy_options, reverse=True):
        if all(cumple_estabilidad(dx, dy, dt, v_x_list[i], v_y_list[i], k_air) for i in range(len(v_x_list))):
                dx_final, dy_final = dx, dy
                break
    if dx_final is not None:
        break

# Verificar si se encontró un dx y dy adecuados
    if dx_final is None or dy_final is None:
        raise ValueError("No se encontró un dx y dy que cumplan todas las condiciones de estabilidad para todas las velocidades.")

    dx, dy = dx_final, dy_final
    print(f"dx elegido: {dx_final}")
    print(f"dy elegido: {dy_final}")

# --- Vectores y matrices ---
num_horas = len(v_air_list)
x = np.arange(0, 100, dx)  # Vector posición en x
t = np.arange(0, 20, dt)  # Vector tiempo
y = np.arange(0, 100, dy)  # Vector posición en y
T = np.ones((len(x), len(y), len(t))) * 295  # Inicialización de temperatura 

# ----- Condición de borde Área Verde ------

## Matriz de ceros 
AV = np.zeros((len(x),len(y)))

ancho_AV_m = 3  # Ancho del área verde en metros
altura_AV_m = 3  # Altura del área verde en metros

# Area basal Area Verde
A_b =  ancho_AV_m * altura_AV_m                # Basal Green Area m^2

ancho_AV = int(ancho_AV_m / dx)  # Convertir a número de celdas
altura_AV = int(altura_AV_m / dy)  # Convertir a número de celdas

# ------ Posición área verde  ----------
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

# ------------------------------------------------------------------------------------------
## --------------------------------- FUNCIÓN SIMULACIÓN ------------------------------------
def run_simulation(constants, params, v_air_list, direction_list, dx, dy, dt, x, y, t):

#----- Parámetros -----
    Kc, h_out = params

#--- Constantes ---
    k_air = constants[0]
    sigma = 5.67*10**(-8)  # Stefan-Boltzmann constant W / m^2*K^4
    V = 1                  # volume of the parallelepiped m^3 
    A = 1                  # Transfer Area m^2
    L = 1                  # Characteristic length m
    Gr = 709.84            # effective incident solar radiation W / m^2

# --- Soil Proporties ---
    e_soil = 0.9           # emissivity
    a_soil = 0.09          # albedo
    alpha_soil = 0.01      # absorbance

## --- Air Proporties ---
    rho       = 1.184          # density kg / m^3
    mu        = 0.066564       # viscosity Kg / m*h
    cp        = 1007           # specific heat J / kg*K
    tau       = 0.9            # Transmittance 
    alpha_air = 0.1            # aborbance
    rho_air   = 0              # reflectance
    Pr        = 0.7296         # Prandtl Number

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

# Evaporación
    rho_w  = 997.13             # densidad del agua kg/m^3
    lamda  = 2441.7             # Calor latente de vaporización kJ/kg 

## --------------- Diferencias Finitas -----------

    for k in range(1, len(t) - 1):  # Iterar en el tiempo
        hora_actual = (k // (len(t) // num_horas)) % num_horas  # Determina la hora del día
        v_x, v_y = v_x_list[hora_actual], v_y_list[hora_actual]
        print(f"Hora {hora_actual}: v_x = {v_x}, v_y = {v_y}")  # Verificación

    #calculos de evaporación que dependen del viento
        u_2 = v_air_list[hora_actual]
        ET_0 = (0.408*Delta*(Rn - G) + gamma*(37/T_av)*u_2*(e_0-e_a))/(Delta + gamma*(1 + 0.34*u_2))
        ET_caj = (Kc)*ET_0
        m_w        = ET_caj*10**(-3)*A_b*rho_w/3500  # kg/s
        if num_celdas_periferia > 0:
            m_wn = m_w / num_celdas_periferia        # Distribuir m_w entre las celdas de la periferia
        else:
            raise ValueError("El número de celdas en la periferia es cero.")

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
            #   print(f'T_x = {T_x[i,j]}')

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
            #print(f'T_y = {T_y[i,j]}')

# Suma ponderada según la dirección del viento 
        theta_deg = wind_directions[direction_list[hora_actual]]  # Ángulo en grados

# Inicializar pesos
        weight_x = 0
        weight_y = 0

# Calcular los pesos de manera lineal por cuadrantes
        if 0 <= theta_deg <= 90:  # Cuadrante 1 (N → E)
            weight_y = 1 - (theta_deg / 90)  # De 1 a 0
            weight_x = theta_deg / 90        # De 0 a 1

        elif 90 < theta_deg <= 180:  # Cuadrante 2 (E → S)
            theta_rel = theta_deg - 90
            weight_x = 1 - (theta_rel / 90)  # De 1 a 0
            weight_y = theta_rel / 90        # De 0 a 1

        elif 180 < theta_deg <= 270:  # Cuadrante 3 (S → W)
            theta_rel = theta_deg - 180
            weight_y = 1 - (theta_rel / 90)  # De 1 a 0
            weight_x = theta_rel / 90        # De 0 a 1

        elif 270 < theta_deg <= 360:  # Cuadrante 4 (W → N)
            theta_rel = theta_deg - 270
            weight_x = 1 - (theta_rel / 90)  # De 1 a 0
            weight_y = theta_rel / 90        # De 0 a 1

# Casos especiales para ángulos exactos
        if theta_deg == 0 or theta_deg == 180:  # N o S
            weight_x = 0
            weight_y = 1
        elif theta_deg == 90 or theta_deg == 270:  # E o W
            weight_x = 1
            weight_y = 0

# Aplicar la ponderación a las matrices
        T[:, :, k+1] = weight_x * T_x + weight_y * T_y


    # Condición de borde en la periferia del área verde
        for i in range(len(x)):
            for j in range(len(y)):
                if periferia_AV[i, j] == 1:          # Si estamos en una celda de la periferia del área verde
                    T[i, j, k + 1] += -m_wn * lamda  # Agregar el calor de cambio de fase

# Mantener la temperatura fija en el área verde
    for i_av in range(AV.shape[0]):
        for j_av in range(AV.shape[1]):
            if AV[i_av, j_av] == 1:      # Si es parte del área verde
                T[i_av, j_av, :] = T_av  # Sobrescribir con temperatura fija

    return T

T_simulado = run_simulation(constants, params, v_air_list, direction_list, dx, dy, dt, x, y, t)

# ------------------------------------------------------------------------------------------
## --------------------------------- FUNCIÓN ERROR -----------------------------------------

# Datos inventados
# Definir cantidad de datos de prueba
n_datos = 10

# Elegimos 10 puntos dentro de los rangos posibles
np.random.seed(42)  # Semilla para reproducibilidad
x_data = np.random.choice(x, size=n_datos)
y_data = np.random.choice(y, size=n_datos)
t_data = np.random.choice(t, size=n_datos)

T_obs = [T_simulado[int(xi/dx), int(yi/dy), int(ti/dt)] + np.random.normal(0, 0.5)
         for xi, yi, ti in zip(x_data, y_data, t_data)]

def error_function_scalar(params, x_data, y_data, t_data, T_obs, v_air_list, direction_list, dx, dy, dt, x, y, t):
    
    T_model = run_simulation(constants, params, v_air_list, direction_list, dx, dy, dt, x, y, t)
  
    errores = []
    for xi, yi, ti, T_obs_i in zip(x_data, y_data, t_data, T_obs):
        # Encontrar las posiciones en la malla
        i = int(xi / dx)
        j = int(yi / dy)
        k = int(ti / dt)

        # Verifica que no se salga del dominio
        if i < 0 or i >= len(x) or j < 0 or j >= len(y) or k < 0 or k >= len(t):
            continue  # o puedes lanzar un error

        # Obtener la temperatura simulada
        T_model_i = T_model[i, j, k]

        # Calcular error y guardar
        error_i = T_model_i - T_obs_i
        errores.append(error_i**2)
    mse = np.mean(errores)
    return np.sqrt(mse)  # Arreglo que el optimizador tratará de hacer cero
    
# ------------------------------------------------------------------------------------------
## --------------------------------- OPTIMIZADOR -----------------------------------------

from scipy.optimize import differential_evolution

params0 = [0.9, 0.01]  # Valores iniciales

bounds = [(0.8  ,  1.1),     # Ks
          (0.005, 0.05)]     # h_out


res = differential_evolution(error_function_scalar, bounds,
    args=(x_data, y_data, t_data, T_obs, v_air_list, direction_list, dx, dy, dt, x, y, t),
    strategy='best1bin',                # Estrategia estándar
    mutation=(0.5, 1),                  # Rango de mutación
    recombination=0.7,                 # Recombinación entre soluciones
    seed=42,
    disp=True                           # Mostrar info del progreso
)

# Resultados
print("\n🔍 Resultado con Differential Evolution:")
print("Parámetros ajustados:", res.x)
print("Error final (RMSE):", res.fun)