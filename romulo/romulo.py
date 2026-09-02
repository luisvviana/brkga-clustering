# romulo.py

import math
import random

from deap import creator, algorithms, base, tools
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

class RomuloIndividual:

    def __init__(self, number_data, number_clusters, number_attributes):
        self.number_data = number_data
        self.number_clusters = max(2, random.randint(1, number_clusters))
        self.number_attributes = number_attributes
        self.labels = [-1 for _ in range(0, number_data)]
        self.subspace = [random.randint(0, 1) for _ in range(number_attributes)]
        while all(not flag for flag in self.subspace):
            self.subspace = [random.randint(0, 1) for _ in range(number_attributes)]
        self.silhouette = -math.inf
        self.davies_bouldin = math.inf
        self.calinski_harabasz = -math.inf

class Clusterer:

    # Pesos por métrica: (k, métrica)
    # k é sempre minimizado (-1.00)
    METRIC_WEIGHTS = {
        "sil": (-1.00, +1.00),          # minimiza k, maximiza silhouette
        "db":  (-1.00, -1.00),          # minimiza k, minimiza davies-bouldin
        "ch":  (-1.00, +1.00),          # minimiza k, maximiza calinski-harabasz
        "all": (-1.00, +1.00, -1.00, +1.00),  # minimiza k, max sil, min db, max ch
    }

    def __init__(self, configuration, metric="sil"):
        if metric not in self.METRIC_WEIGHTS:
            raise ValueError(f"Métrica inválida: '{metric}'. Use: sil, db, ch ou all.")

        self.metric = metric

        if "RomuloFitnessMax" in creator.__dict__:
            del creator.RomuloFitnessMax
        if "RomuloIndividual" in creator.__dict__:
            del creator.RomuloIndividual

        creator.create("RomuloFitnessMax", base.Fitness, weights=self.METRIC_WEIGHTS[metric])
        creator.create("RomuloIndividual", RomuloIndividual, fitness=creator.RomuloFitnessMax,
                       number_data=None, number_clusters=None, number_attributes=None)

        self.CXPB = configuration["CXPB"]
        self.MUTPB = configuration["MUTPB"]
        self.NGEN = configuration["NGEN"]
        self.PSIZE = configuration["PSIZE"]
        self.MU = math.ceil(configuration["MU"] * configuration["PSIZE"])
        self.LAMBDA = math.ceil(configuration["LAMBDA"] * configuration["PSIZE"])

    def work(self, dataset):
        self.dataset = dataset
        number_data = self.dataset.shape[0]
        number_attributes = len(self.dataset.columns)

        number_clusters = int(math.ceil(len(self.dataset.index) * 0.10))
        number_clusters = max([2, number_clusters])
        number_clusters = min([len(self.dataset.index) - 1, number_clusters])

        self.toolbox = base.Toolbox()
        self.toolbox.register("evaluate", self.evaluate)
        self.toolbox.register("mate", self.mate)
        self.toolbox.register("mutate", self.mutate)
        self.toolbox.register("select", self.select)
        self.toolbox.register("generate_individual", creator.RomuloIndividual,
                              number_data=number_data, number_clusters=number_clusters,
                              number_attributes=number_attributes)
        self.toolbox.register("generate_population", tools.initRepeat, list,
                              self.toolbox.generate_individual, self.PSIZE)

        population = self.toolbox.generate_population()
        offspring, _ = algorithms.eaMuPlusLambda(
            population=population,
            toolbox=self.toolbox,
            ngen=self.NGEN,
            cxpb=self.CXPB,
            mutpb=self.MUTPB,
            mu=self.MU,
            lambda_=self.LAMBDA,
            stats=None,
            verbose=False
        )
        winners = tools.selBest(offspring, k=1)
        return winners[0]

    def _fallback_fitness(self):
        """Retorna fitness de fallback adequado para a métrica atual."""
        if self.metric == "sil":
            return (1, -1.0)
        elif self.metric == "db":
            return (1, math.inf)
        elif self.metric == "ch":
            return (1, -1.0)
        else:  # all
            return (1, -1.0, math.inf, 0.0)

    def evaluate(self, individual):
        attribute_indices = [i for i in range(len(individual.subspace)) if individual.subspace[i] == 1]

        if not attribute_indices:
            return self._fallback_fitness()

        projection = self.dataset.iloc[:, attribute_indices]

        if projection.shape[0] < individual.number_clusters:
            return self._fallback_fitness()

        if len(self.dataset.index) <= 1000:
            clusterer = KMeans(n_clusters=individual.number_clusters)
        else:
            clusterer = MiniBatchKMeans(n_clusters=individual.number_clusters,
                                        init_size=individual.number_clusters)

        clusterer.fit(projection)
        individual.labels = clusterer.predict(projection)
        number_clusters = len(set(individual.labels))

        if number_clusters == 1 or number_clusters == len(individual.labels):
            individual.silhouette = -1.0
            individual.davies_bouldin = math.inf
            individual.calinski_harabasz = 0.0
            return self._fallback_fitness()

        sil = silhouette_score(projection, individual.labels)
        db  = davies_bouldin_score(projection, individual.labels)
        ch  = calinski_harabasz_score(projection, individual.labels)

        individual.silhouette = sil
        individual.davies_bouldin = db
        individual.calinski_harabasz = ch

        if self.metric == "sil":
            return (number_clusters, sil)
        elif self.metric == "db":
            return (number_clusters, db)
        elif self.metric == "ch":
            return (number_clusters, ch)
        else:  # all
            return (number_clusters, sil, db, ch)

    def mate(self, individual1, individual2):
        individual1.number_clusters, individual2.number_clusters = \
            individual2.number_clusters, individual1.number_clusters
        tools.cxOnePoint(individual1.subspace, individual2.subspace)
        return individual1, individual2

    def select(self, population, size):
        return tools.selLexicase(population, size)

    def mutate(self, individual):
        if individual.number_clusters > 2:
            individual.number_clusters -= 1
        elif individual.number_clusters < 10:
            individual.number_clusters += 1
        tools.mutFlipBit(individual.subspace, indpb=self.MUTPB)
        return individual,