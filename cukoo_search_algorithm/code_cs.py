import numpy as np
from scipy.special import gamma

class CuckooSearch:
    def __init__(self, n_nests, n_iterations, alpha, beta, levy_exponent, bounds, num_nodes):
        self.n_nests = n_nests  
        self.n_iterations = n_iterations  
        self.alpha = alpha  
        self.beta = beta  
        self.levy_exponent = levy_exponent  
        self.bounds = bounds  
        self.num_nodes = num_nodes  
        self.nests = self.initialize_nests()  

    def initialize_nests(self):
        """Randomly initialize sensor node positions within bounds."""
        return np.random.uniform(self.bounds[0], self.bounds[1], (self.n_nests, self.num_nodes, 2))

    def evaluate_fitness(self, nest):
        """
        Objective function:
        Minimize total energy consumption, defined as the sum of distances between all nodes
        (representing communication cost).
        """
        total_energy = 0
        for i in range(len(nest)):
            for j in range(i + 1, len(nest)):
                distance = np.linalg.norm(nest[i] - nest[j])
                total_energy += distance 
        return total_energy

    def levy_flight(self):
        """Generate Lévy flight step sizes."""
        sigma = (gamma(1 + self.levy_exponent) * np.sin(np.pi * self.levy_exponent / 2) /
                 (gamma((1 + self.levy_exponent) / 2) * self.levy_exponent * 2 ** ((self.levy_exponent - 1) / 2))) ** (1 / self.levy_exponent)
        u = np.random.normal(0, sigma, size=(self.num_nodes, 2))
        v = np.random.normal(0, 1, size=(self.num_nodes, 2))
        step = u / (np.abs(v) ** (1 / self.levy_exponent))
        return step

    def optimize(self):
        """Main optimization process."""
        best_nest = self.nests[0]
        best_fitness = self.evaluate_fitness(best_nest)

        for iteration in range(self.n_iterations):
            for i in range(self.n_nests):
                step = self.alpha * self.levy_flight()
                self.nests[i] += step
                self.nests[i] = np.clip(self.nests[i], self.bounds[0], self.bounds[1]) 

                current_fitness = self.evaluate_fitness(self.nests[i])
                if current_fitness < best_fitness:
                    best_nest = np.copy(self.nests[i])
                    best_fitness = current_fitness

            random_nests = self.initialize_nests()[:self.n_nests // 2]
            self.nests[self.n_nests // 2:] = random_nests

            print(f"Iteration {iteration + 1}, Best Fitness: {best_fitness}")

        return best_nest, best_fitness


num_nodes = 10  
bounds = [0, 100]  
n_nests = 20  
n_iterations = 50 
alpha = 0.5
beta = 1.5 
levy_exponent = 1.5 

cs = CuckooSearch(n_nests=n_nests, n_iterations=n_iterations, alpha=alpha, beta=beta,
                  levy_exponent=levy_exponent, bounds=bounds, num_nodes=num_nodes)
best_solution, best_fitness = cs.optimize()

print("\nOptimal Node Placement:")
for i, coord in enumerate(best_solution):
    print(f"Node {i + 1}: {coord}")
print(f"\nBest Fitness (Total Energy Consumption): {best_fitness}")
