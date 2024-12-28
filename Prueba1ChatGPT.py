import numpy as np
import matplotlib.pyplot as plt

# Parámetros físicos y geométricos
Nx, Ny = 100, 100  # Tamaño de la malla
Lx, Ly = 10.0, 10.0  # Dimensiones físicas (m)
dx = Lx / (Nx - 1)
dy = Ly / (Ny - 1)
dt = 10  # Paso de tiempo (s)
Tmax = 1800  # Tiempo máximo de simulación (s)
Nt = int(Tmax / dt) + 1  # Número de pasos temporales

# Propiedades del material y condiciones iniciales
rho = 1.2  # Densidad (kg/m^3)
cp = 1000  # Calor específico (J/kg K)
k = 0.5  # Conductividad térmica (W/mK)
alpha = k / (rho * cp)  # Difusividad térmica (m^2/s)
h_out = 15  # Coeficiente de convección (W/m^2K)
T_air = 300  # Temperatura del aire (K)

# Parámetros adicionales
sigma = 5.67e-8  # Constante de Stefan-Boltzmann
Gr = 9.81  # Gravedad
epsilon_soil = 0.9  # Emisividad del suelo
alpha_soil = 0.7  # Absorción del suelo
a_soil = 0.8  # Albedo del suelo
tau = 0.9  # Transmitancia atmosférica
alpha_sup = 0.6  # Absorción de la superficie
rho_air = 1.225  # Densidad del aire (kg/m^3)
v = 0.1  # Velocidad del aire (m/s)

# Vectores y matrices
x = np.arange(0, Lx + dx, dx)  # Vector posición en x
y = np.arange(0, Ly + dy, dy)  # Vector posición en y
t = np.arange(0, Tmax + dt, dt)  # Vector tiempo

T = np.zeros((len(x), len(y), len(t)))  # Matriz de ceros (mallado)

# Condiciones iniciales y de borde
T[:, :, 0] = 300  # Temperatura inicial uniforme
T[:, 0, :] = 310  # Borde caliente
T[:, -1, :] = 290  # Borde frío
T[0, :, :] = 300
T[-1, :, :] = 300

# Función para calcular diferencias finitas
def dTdt(T, Tlap, dx, dy):
    return (
        (epsilon_soil * sigma * (T**4 - T_air**4)) +  # Radiación suelo
        (a_soil * tau * Gr) -  # Absorción de suelo
        (alpha_soil * tau * Gr) +  # Absorción superficial
        (alpha_sup * Gr) -  # Calor en la superficie
        (rho_air * Gr) -  # Pérdida de calor al aire
        (h_out * (T - T_air)) -  # Convección
        (rho * cp * v * ((np.roll(T, -1, axis=0) - np.roll(T, 1, axis=0)) / (2 * dx) +
                         (np.roll(T, -1, axis=1) - np.roll(T, 1, axis=1)) / (2 * dy))) +  # Transporte por advección
        (k * Tlap)  # Conducción
    ) / (rho * cp)

# Función para calcular el laplaciano con diferencias finitas
def laplaciano(T, dx, dy):
    Txx = (np.roll(T, -1, axis=0) - 2 * T + np.roll(T, 1, axis=0)) / dx**2
    Tyy = (np.roll(T, -1, axis=1) - 2 * T + np.roll(T, 1, axis=1)) / dy**2
    return Txx + Tyy

# Simulación en el tiempo
for k in range(len(t) - 1):
    Tlap = laplaciano(T[:, :, k], dx, dy)
    for j in range(len(y)):
        for i in range(1, len(x)):
            T[i, j, k + 1] = T[i, j, k] + dTdt(T[i, j, k], Tlap[i, j], dx, dy) * dt

    # Visualización en tiempo real (cada cierto paso)
    if k % 10 == 0:
        plt.clf()
        plt.imshow(T[:, :, k], cmap='hot', origin='lower', extent=[0, Ly, 0, Lx])
        plt.colorbar(label='Temperature (K)')
        plt.title(f'Tiempo = {t[k]} s')
        plt.pause(0.1)

plt.show()
