import random
import time
import test_exercise_2 as test


class Die:
    def __init__(self):
        self.value = None
        self.roll()

    def roll(self):
        self.value = random.randint(1, 6)


def aufgabe_1b(die_class):
    # Initialisierung
    dice = [die_class() for _ in range(100)]
    target_sum = random.randint(100, 600)

    start_time = time.time()
    timeout = 180

    while time.time() - start_time < timeout:
        for d in dice:
            d.roll()

        if sum(d.value for d in dice) == target_sum:
            return True  # Target value wurde erreicht

    return False  # Target value wurde nicht erreicht


class Individual:
    def __init__(self, dice_count=100, init_values=None):
        if init_values is not None:
            self.dice = []
            for val in init_values:
                d = Die()
                d.value = val
                self.dice.append(d)
        else:
            self.dice = [Die() for _ in range(dice_count)]
        self.fitness = 0
        self.update_fitness()

    def update_fitness(self):
        self.fitness = sum(1 for d in self.dice if d.value == 6)

    def mutate(self, probability):
        for d in self.dice:
            if random.random() < probability:
                d.roll()
        self.update_fitness()


def evolution_algorithm():
    pop_size = 100
    survivor_rate = 0.2
    mut_rate = 0.05

    population = [Individual() for _ in range(pop_size)]
    for _ in range(1000):
        population.sort(key=lambda ind: ind.fitness, reverse=True)
        if population[0].fitness == 100:
            break

        survivors = population[:int(pop_size * survivor_rate)]
        next_gen = list(survivors)
        while len(next_gen) < pop_size:
            p1, p2 = random.sample(survivors, 2)
            child_vals = [p1.dice[i].value if random.random() < 0.5
                          else p2.dice[i].value for i in range(100)]
            child = Individual(init_values=child_vals)
            child.mutate(mut_rate)
            next_gen.append(child)
        population = next_gen

    return [d.value for d in population[0].dice]


def fast_evolution_algorithm():
    pop_size = 100
    mut_rate = 0.05
    pop = [([random.randint(1, 6) for _ in range(100)], 0) for _ in range(pop_size)]
    pop = [(g, g.count(6)) for g, _ in pop]

    for _ in range(1000):
        pop.sort(key=lambda x: x[1], reverse=True)
        if pop[0][1] == 100:
            break
        survivors = pop[:20]
        next_gen = list(survivors)
        while len(next_gen) < pop_size:
            p1, p2 = random.choice(survivors)[0], random.choice(survivors)[0]
            pt = random.randint(1, 99)
            child = p1[:pt] + p2[pt:]
            for i in range(100):
                if random.random() < mut_rate:
                    child[i] = random.randint(1, 6)
            next_gen.append((child, child.count(6)))
        pop = next_gen
    return pop[0][0]


if __name__ == "__main__":

    test.test_task_2_1_a(Die)

    test.test_task_2_1_b(aufgabe_1b, kwargs={"die_class": Die}, timeout=180)

    test.test_task_2_1_e(evolution_algorithm, kwargs={})

    test.test_task_2_1_f(
        evolution_algorithm, {},
        fast_evolution_algorithm, {}
    )

