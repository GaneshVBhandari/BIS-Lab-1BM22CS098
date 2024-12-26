import random

def initialize_population(size, gene_length):
    return [[random.randint(0, 1) for _ in range(gene_length)] for _ in range(size)]

def fitness_function(solution):
    return sum(solution)

def selection(population, fitnesses):
    total_fitness = sum(fitnesses)
    probabilities = [f / total_fitness for f in fitnesses]
    return population[random.choices(range(len(population)), probabilities)[0]]

def crossover(parent1, parent2):
    point = random.randint(1, len(parent1) - 1)
    return parent1[:point] + parent2[point:]

def mutation(solution, mutation_rate):
    for i in range(len(solution)):
        if random.random() < mutation_rate:
            solution[i] = 1 - solution[i]
    return solution

def gene_expression_algorithm(pop_size, gene_length, generations, mutation_rate):
    population = initialize_population(pop_size, gene_length)
    for _ in range(generations):
        fitnesses = [fitness_function(ind) for ind in population]
        new_population = []
        for _ in range(pop_size):
            parent1 = selection(population, fitnesses)
            parent2 = selection(population, fitnesses)
            child = crossover(parent1, parent2)
            child = mutation(child, mutation_rate)
            new_population.append(child)
        population = new_population
    fitnesses = [fitness_function(ind) for ind in population]
    best_solution = population[fitnesses.index(max(fitnesses))]
    return population, best_solution, max(fitnesses)

pop_size = 10
gene_length = 8
generations = 20
mutation_rate = 0.1

print("Input Parameters:")
print("Population Size:", pop_size)
print("Gene Length:", gene_length)
print("Generations:", generations)
print("Mutation Rate:", mutation_rate)

final_population, best_solution, best_fitness = gene_expression_algorithm(pop_size, gene_length, generations, mutation_rate)

print("Final Population:", final_population)
print("Best Solution:", best_solution)
print("Best Fitness:", best_fitness)
