import numpy as np
import matplotlib.pyplot as plt

grid_size = (50, 50)  
dt = 0.5 
diffusion_coefficient = 0.1 
n_iter = 50  

temperature = np.zeros(grid_size)

source_position = (25, 25) 
temperature[source_position] = 100 

def update_temperature(temperature):
    new_temperature = temperature.copy()
    for x in range(1, grid_size[0] - 1):
        for y in range(1, grid_size[1] - 1):
            new_temperature[x, y] += diffusion_coefficient * dt * (
                temperature[x + 1, y] + temperature[x - 1, y] +
                temperature[x, y + 1] + temperature[x, y - 1] -
                4 * temperature[x, y]
            )
    return new_temperature

plt.ion()
for t in range(n_iter):
    temperature = update_temperature(temperature)

    if t % 10 == 0:
        plt.clf()
        plt.imshow(temperature, cmap="hot", origin="lower")
        plt.colorbar(label="Temperature")
        plt.title(f"Heat Diffusion at Iteration {t}")
        plt.pause(0.1)

plt.ioff()
plt.show()