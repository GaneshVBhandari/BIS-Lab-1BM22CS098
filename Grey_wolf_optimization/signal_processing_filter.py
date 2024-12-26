import numpy as np
import matplotlib.pyplot as plt

def objective_function(filter_coeffs, desired_response, freqs):
    w, h = freqs, np.fft.fft(filter_coeffs, len(freqs))
    actual_response = np.abs(h)
    
    mse = np.mean((desired_response - actual_response) ** 2)
    return mse

def gwo_signal_processing(desired_response, freqs, num_wolves=10, max_iter=100):
    num_coeffs = len(desired_response) 
    alpha_pos = np.zeros(num_coeffs)   
    beta_pos = np.zeros(num_coeffs)    
    delta_pos = np.zeros(num_coeffs)    

    alpha_score = float("inf")
    beta_score = float("inf")
    delta_score = float("inf")

    wolves = np.random.uniform(-1, 1, (num_wolves, num_coeffs))

    for t in range(max_iter):
        for i in range(num_wolves):
            fitness = objective_function(wolves[i], desired_response, freqs)

            if fitness < alpha_score:
                alpha_score, beta_score, delta_score = fitness, alpha_score, beta_score
                alpha_pos, beta_pos, delta_pos = wolves[i].copy(), alpha_pos.copy(), beta_pos.copy()
            elif fitness < beta_score:
                beta_score, delta_score = fitness, beta_score
                beta_pos, delta_pos = wolves[i].copy(), beta_pos.copy()
            elif fitness < delta_score:
                delta_score = fitness
                delta_pos = wolves[i].copy()

        a = 2 - 2 * (t / max_iter)  

        for i in range(num_wolves):
            for j in range(num_coeffs):
                r1, r2 = np.random.rand(), np.random.rand()
                A1, C1 = 2 * a * r1 - a, 2 * r2
                D_alpha = abs(C1 * alpha_pos[j] - wolves[i, j])
                X1 = alpha_pos[j] - A1 * D_alpha

                r1, r2 = np.random.rand(), np.random.rand()
                A2, C2 = 2 * a * r1 - a, 2 * r2
                D_beta = abs(C2 * beta_pos[j] - wolves[i, j])
                X2 = beta_pos[j] - A2 * D_beta

                r1, r2 = np.random.rand(), np.random.rand()
                A3, C3 = 2 * a * r1 - a, 2 * r2
                D_delta = abs(C3 * delta_pos[j] - wolves[i, j])
                X3 = delta_pos[j] - A3 * D_delta

                wolves[i, j] = (X1 + X2 + X3) / 3

    return alpha_pos, alpha_score

np.random.seed(42)
num_points = 50
freqs = np.linspace(0, 1, num_points) 

desired_response = np.where(freqs <= 0.3, 1, 0)

best_filter, best_score = gwo_signal_processing(desired_response, freqs)

plt.figure(figsize=(10, 5))
plt.plot(freqs, desired_response, label="Desired Response", linestyle="--")
plt.plot(freqs, np.abs(np.fft.fft(best_filter, len(freqs))), label="Optimized Filter Response")
plt.title("Filter Design using Grey Wolf Optimizer")
plt.xlabel("Normalized Frequency")
plt.ylabel("Amplitude")
plt.legend()
plt.grid()
plt.show()

print("Best Filter Coefficients:", best_filter)
print("Best Score (MSE):", best_score)