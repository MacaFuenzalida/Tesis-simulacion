import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

# Simulación de una matriz 5x5 que cambia con el tiempo
tiempo_total = 30
matrices = [np.random.uniform(25, 45, (5, 5)) for _ in range(tiempo_total)]

fig, ax = plt.subplots()
cax = ax.matshow(matrices[0], cmap='hot')
plt.colorbar(cax)

def update(frame):
    cax.set_array(matrices[frame])
    ax.set_title(f'Tiempo: {frame}s')
    return cax,

anim = FuncAnimation(fig, update, frames=tiempo_total, interval=200)
plt.show()

