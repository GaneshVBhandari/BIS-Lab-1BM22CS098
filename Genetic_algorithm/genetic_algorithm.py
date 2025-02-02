import numpy as np

population_size = 20
num_cities = 10
generations = 100
crossover_rate = 0.8
mutation_rate = 0.2

np.random.seed(42) 
cities = np.random.rand(num_cities, 2) * 100  

distance_matrix = np.sqrt(
    np.sum((cities[:, np.newaxis, :] - cities[np.newaxis, :, :]) ** 2, axis=2)
)

def calculate_route_distance(route):
    distance = 0
    for i in range(len(route) - 1):
        distance += distance_matrix[route[i], route[i + 1]]
    distance += distance_matrix[route[-1], route[0]] 
    return distance

def initialize_population(size, num_cities):
    return np.array([np.random.permutation(num_cities) for _ in range(size)])

def evaluate_fitness(population):
    distances = np.array([calculate_route_distance(route) for route in population])
    fitness = 1 / (1 + distances)
    return fitness

def select_parents(population, fitness):
    probabilities = fitness / fitness.sum()
    indices = np.random.choice(len(population), size=2, p=probabilities)
    return population[indices[0]], population[indices[1]]

def crossover(parent1, parent2, crossover_rate):
    if np.random.rand() < crossover_rate:
        start, end = sorted(np.random.choice(len(parent1), size=2, replace=False))
        child1 = np.full_like(parent1, -1)
        child2 = np.full_like(parent2, -1)

        child1[start:end] = parent1[start:end]
        child2[start:end] = parent2[start:end]

        fill_index1 = end
        for gene in parent2:
            if gene not in child1:
                child1[fill_index1 % len(parent1)] = gene
                fill_index1 += 1

        fill_index2 = end
        for gene in parent1:
            if gene not in child2:
                child2[fill_index2 % len(parent2)] = gene
                fill_index2 += 1

        return child1, child2
    else:
        return parent1, parent2

def mutate(individual, mutation_rate):
    if np.random.rand() < mutation_rate:
        i, j = np.random.choice(len(individual), size=2, replace=False)
        individual[i], individual[j] = individual[j], individual[i]
    return individual

def genetic_algorithm():
    population = initialize_population(population_size, num_cities)

    for generation in range(generations):
        fitness = evaluate_fitness(population)
        best_distance = 1 / fitness.max() - 1  

        print(f"Generation {generation + 1}: Best Distance = {best_distance:.2f}")

        new_population = []

        for _ in range(population_size // 2):

            parent1, parent2 = select_parents(population, fitness)
  
            child1, child2 = crossover(parent1, parent2, crossover_rate)

            child1 = mutate(child1, mutation_rate)
            child2 = mutate(child2, mutation_rate)

            new_population.extend([child1, child2])

        population = np.array(new_population)

    fitness = evaluate_fitness(population)
    best_index = np.argmax(fitness)
    best_route = population[best_index]
    best_distance = 1 / fitness[best_index] - 1 

    print(f"\nBest Route: {best_route}")
    print(f"Best Distance: {best_distance:.2f}")

genetic_algorithm()
