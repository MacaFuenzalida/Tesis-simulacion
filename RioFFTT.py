import numpy as np
import matplotlib.pyplot as plt

# %% Parámetros
v = 2  # Velocidad del río [m/s]
dt = 10  # Delta t [s]
dx = dt * v  # Delta x [m]

ancho = 300  # Ancho del río [m]
dy = 1  # Delta y [m]

# Posición de los pueblos
i_pueblo_1 = int(3000 / dx)
i_pueblo_2 = int(np.ceil(3200 / dx))

# %% Vectores y matrices
x = np.arange(0, 4000 + dx, dx)  # Vector posición en el largo del río [m]
t = np.arange(0, 1800 + dt, dt)  # Vector tiempo [s]
y = np.arange(0, ancho + dy, dy)  # Vector posición en el ancho del río [m]

A = np.zeros((len(x), len(y), len(t)))  # Matriz de ceros (mallado)

# %% Condiciones iniciales
k_derrame = int(180 / dt + 1)  # Tiempo de fin del derrame (180 s)

for k in range(k_derrame):
    for j in range(len(y)):
        A[0, j, k] = 10  # Concentración inicial del derrame [mol/m³]

# %% Completar la matriz A con diferencias finitas
for k in range(len(t) - 1):  # Iterar en el tiempo
    for j in range(len(y)):
        for i in range(1, len(x)):  # Empezar desde i=1 para evitar problemas con i-1
            A[i, j, k + 1] = A[i, j, k] - v * (dt / dx) * (A[i, j, k] - A[i - 1, j, k])

# %% Graficar y animar
# %% Graficar y animar
fig, ax = plt.subplots()
cax = ax.imshow(A[:, :, 0], extent=[0, ancho, 0, 4000], origin='lower', aspect='auto', cmap='hot')
cbar = plt.colorbar(cax, ax=ax, label='Concentración [mol/m³]')

# Marcar pueblos
ax.axhline(y=x[i_pueblo_1], color='cyan', linestyle='--')
ax.axhline(y=x[i_pueblo_2], color='cyan', linestyle='--')

ax.set_ylabel('Posición en el largo del río [m]')
ax.set_xlabel('Posición en el ancho del río [m]')

for k in range(len(t)):
    cax.set_data(A[:, :, k])  # Actualizar solo los datos de la gráfica
    cax.set_clim(vmin=A.min(), vmax=A.max())  # Asegurar que los colores estén bien escalados
    plt.title(f'Animación en el tiempo: {t[k]} [s]')
    plt.pause(0.1)  # Pausa para que se vea la animación

plt.show()  # Mantener la animación visible al final
