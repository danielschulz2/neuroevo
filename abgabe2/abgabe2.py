import random
import test_exercise_2 as test
import time
import numpy as np


# Aufgabe 1a
class Die:
    def __init__(self):
        self.value = random.randint(1, 6)

    def roll(self):
        self.value = random.randint(1, 6)

    def __repr__(self):
        return f"[{self.value}]"


# Aufgabe 1b
def random_roll_method(die_class) -> bool:
    # 100 zufällig initialisierte Würfel generieren
    dice = [die_class() for _ in range(100)]

    # Zufällige Target-Summe wählen
    target_sum = random.randint(100, 600)

    start_time = time.time()
    max_duration = 180

    print(f"Target sum:{target_sum}")

    # Würfeln bis die Summe erreicht ist oder die Zeit abläuft
    while True:
        current_sum = sum(d.value for d in dice)

        if current_sum == target_sum:
            print("Erfolg")
            return True

        # Abbruchbedingung Zeit
        if (time.time() - start_time) > max_duration:
            print("Fail")
            return False

        # Alle Würfel neu würfeln
        for d in dice:
            d.roll()


class Individual:
    def __init__(self):
        # Erstellt eine Liste mit 100 Die-Objekten
        self.dice = [Die() for _ in range(100)]

        # Speichert den Fitnesswert und initialen Fitnesswert direkt berechnen
        self.fitness = 0.0
        self.update_fitness()

    def update_fitness(self):
        # Je mehr Sechser, desto höher die Fitness (max. Fitness ist 100)
        self.fitness = sum(1 for die in self.dice if die.value == 6)
        return self.fitness

    def mutate(self, p: float):
        # Geht jeden Würfel durch und würfelt ihn mit der Wahrscheinlichkeit p neu.
        # p: Wahrscheinlichkeit zwischen 0.0 und 1.0.
        for die in self.dice:
            if random.random() < p:
                die.roll()

        self.update_fitness()

    def __repr__(self):
        return f"Individual(Fitness: {self.fitness})"


# Aufgabe 1d
POPULATION_SIZE = 100
SURVIVAL_RATE = 0.25
MUTATION_RATE = 0.07  # Für mutate()-Funktion
MAX_GENERATIONS = 1000
TARGET_FITNESS = 100


# Aufgabe 1e
def run_evolution(pop_size=POPULATION_SIZE, mutation_rate=MUTATION_RATE,
                  max_gen=MAX_GENERATIONS, survival_rate=SURVIVAL_RATE):
    # Initialisierung
    population = [Individual() for _ in range(pop_size)]
    num_survivors = int(pop_size * survival_rate)

    for iteration in range(max_gen):
        # Fitness absteigend sortieren
        population.sort(key=lambda ind: ind.fitness, reverse=True)

        # Haben wir hundert Sechser?
        if population[0].fitness == 100:
            break

        # Selektion
        survivors = population[:num_survivors]

        # Reproduktion
        next_gen = list(survivors)
        while len(next_gen) < POPULATION_SIZE:
            # Zwei zufällige Eltern aus den Überlebenden auswählen
            parent1 = random.choice(survivors)
            parent2 = random.choice(survivors)

            # Neues Kind erstellen
            child = Individual()

            for i in range(100):
                # Mit 50% Wahrscheinlichkeit Gen von Parent 1, sonst von Parent 2
                if random.random() < 0.5:
                    child.dice[i].value = parent1.dice[i].value
                else:
                    child.dice[i].value = parent2.dice[i].value

            # Mutation auf das Kind anwenden
            child.mutate(MUTATION_RATE)

            # Kind der neuen Generation hinzufügen
            next_gen.append(child)

        # Die alte Population durch die neue Gen ersetzen
        population = next_gen

    best_ind = population[0]
    return [die.value for die in best_ind.dice]


params = {
        "pop_size": POPULATION_SIZE,
        "mutation_rate": MUTATION_RATE,
        "max_gen": MAX_GENERATIONS,
        "survival_rate": SURVIVAL_RATE
    }

test.test_task_2_1_e(run_evolution, params)

"""
Aufgabe 1f:
Anstelle davon, dass die ganzen Die- und Individual-Objekte durchgegangen werden,
erstelle ich eine Matrix mit numpy, die die Struktur repräsentiert, wobei
(Zeilen = Individuen, Spalten = Würfel).
Des Weiteren benutze ich np.sum um die Fitness schneller auszurechnen.
Ich benutze auch np.argsort zum sortieren.
"""


def run_evolution_optimized(pop_size=POPULATION_SIZE, mutation_rate=MUTATION_RATE,
                            max_gen=MAX_GENERATIONS, survival_rate=SURVIVAL_RATE):
    num_dice = 100
    num_survivors = int(pop_size * survival_rate)

    # Initialisierung
    population = np.random.randint(1, 7, size=(pop_size, num_dice))

    for _ in range(max_gen):
        # Fitness berechnen (Anzahl der Sechsen pro Zeile)
        fitness = np.sum(population == 6, axis=1)

        # Sortieren (Indizes der besten Individuen finden)
        sorted_indices = np.argsort(fitness)[::-1]
        best_fitness = fitness[sorted_indices[0]]

        if best_fitness == num_dice:
            return population[sorted_indices[0]]

        # Selektion
        survivors = population[sorted_indices[:num_survivors]]

        # Reproduktion (Auffüllen der Population)
        new_population = np.zeros_like(population)
        new_population[:num_survivors] = survivors

        # Reproduktion
        for i in range(num_survivors, pop_size):
            # Zwei zufällige Eltern aus den Überlebenden wählen
            parent1 = survivors[random.randint(0, num_survivors - 1)]
            parent2 = survivors[random.randint(0, num_survivors - 1)]

            # Ein neues Kind erstellen
            child = np.zeros(num_dice, dtype=int)

            # Crossover
            for g in range(num_dice):
                if random.random() < 0.5:
                    child[g] = parent1[g]
                else:
                    child[g] = parent2[g]

            # Mutation
            for g in range(num_dice):
                if random.random() < mutation_rate:
                    child[g] = random.randint(1, 6)

            new_population[i] = child
        population = new_population

    final_fitness = np.sum(population == 6, axis=1)
    return population[np.argmax(final_fitness)]


params_def = {
    "pop_size": POPULATION_SIZE,
    "mutation_rate": MUTATION_RATE,
    "max_gen": MAX_GENERATIONS,
    "survival_rate": SURVIVAL_RATE
}

params_opt = {
    "pop_size": POPULATION_SIZE,
    "mutation_rate": MUTATION_RATE,
    "max_gen": MAX_GENERATIONS,
    "survival_rate": SURVIVAL_RATE
}

test.test_task_2_1_f(
    run_evolution,
    params_def,
    run_evolution_optimized,
    params_opt
    )
