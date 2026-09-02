#include "../headers/decoder.hpp"
#include "../headers/silhouette.hpp"
#include "../headers/davies_bouldin.hpp"
#include "../headers/calinski_harabasz.hpp"
#include <iostream>
#include <vector>
#include <set>
#include <cmath>
#include <limits>
#include <random>
#include <algorithm>
#include <chrono>
#include <omp.h>

using namespace std;
using namespace BRKGA;

HybridDecoderV2::HybridDecoderV2(const HybridInstance& instance, const std::string& metric)
    : instance(instance), lambda_k(instance.lambda_k), metric(metric)
{}

// K-means com OpenMP
static vector<int> kmeans(const vector<vector<double>>& data, int k, int max_iter = 100) {
    int n = (int)data.size();
    int dim = (int)(n > 0 ? data[0].size() : 0);
    if (n == 0 || dim == 0 || k <= 0) return {};

    vector<vector<double>> centers(k, vector<double>(dim));
    for (int i = 0; i < k; ++i)
        centers[i] = data[i % n];

    vector<int> labels(n, -1);
    bool changed = true;
    int iter = 0;

    while (changed && iter < max_iter) {
        changed = false;

        #pragma omp parallel for schedule(static)
        for (int i = 0; i < n; ++i) {
            double best_dist = numeric_limits<double>::max();
            int best_c = -1;
            for (int c = 0; c < k; ++c) {
                double dist = 0.0;
                for (int d = 0; d < dim; ++d) {
                    double diff = data[i][d] - centers[c][d];
                    dist += diff * diff;
                }
                if (dist < best_dist) {
                    best_dist = dist;
                    best_c = c;
                }
            }
            if (labels[i] != best_c) {
                labels[i] = best_c;
                changed = true;
            }
        }

        vector<int> counts(k, 0);
        centers.assign(k, vector<double>(dim, 0.0));

        #pragma omp parallel for schedule(static)
        for (int i = 0; i < n; ++i) {
            int c = labels[i];
            #pragma omp atomic
            counts[c]++;
            for (int d = 0; d < dim; ++d) {
                #pragma omp atomic
                centers[c][d] += data[i][d];
            }
        }

        for (int c = 0; c < k; ++c) {
            if (counts[c] > 0)
                for (int d = 0; d < dim; ++d)
                    centers[c][d] /= counts[c];
            else
                centers[c] = data[rand() % n];
        }

        iter++;
    }
    return labels;
}

// MiniBatch KMeans com OpenMP
static vector<int> minibatch_kmeans(const vector<vector<double>>& data, int k, int batch_size = 100, int max_iter = 100) {
    int n = (int)data.size();
    int dim = (int)(n > 0 ? data[0].size() : 0);
    if (n == 0 || dim == 0 || k <= 0) return {};

    vector<vector<double>> centers(k, vector<double>(dim));
    for (int i = 0; i < k; ++i)
        centers[i] = data[i % n];

    vector<int> labels(n, -1);
    vector<int> counts(k, 0);

    std::default_random_engine rng(std::random_device{}());
    std::uniform_int_distribution<int> dist(0, n - 1);

    for (int iter = 0; iter < max_iter; ++iter) {
        vector<int> batch_indices(batch_size);
        for (int i = 0; i < batch_size; ++i)
            batch_indices[i] = dist(rng);

        #pragma omp parallel for schedule(static)
        for (int b = 0; b < batch_size; ++b) {
            int idx = batch_indices[b];
            double best_dist = numeric_limits<double>::max();
            int best_c = -1;
            for (int c = 0; c < k; ++c) {
                double dist_sq = 0.0;
                for (int d = 0; d < dim; ++d) {
                    double diff = data[idx][d] - centers[c][d];
                    dist_sq += diff * diff;
                }
                if (dist_sq < best_dist) {
                    best_dist = dist_sq;
                    best_c = c;
                }
            }

            #pragma omp critical
            {
                counts[best_c]++;
                double eta = 1.0 / counts[best_c];
                for (int d = 0; d < dim; ++d)
                    centers[best_c][d] = (1 - eta) * centers[best_c][d] + eta * data[idx][d];
                labels[idx] = best_c;
            }
        }
    }

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) {
        double best_dist = numeric_limits<double>::max();
        int best_c = -1;
        for (int c = 0; c < k; ++c) {
            double dist_sq = 0.0;
            for (int d = 0; d < dim; ++d) {
                double diff = data[i][d] - centers[c][d];
                dist_sq += diff * diff;
            }
            if (dist_sq < best_dist) {
                best_dist = dist_sq;
                best_c = c;
            }
        }
        labels[i] = best_c;
    }

    return labels;
}

BRKGA::fitness_t HybridDecoderV2::decode(Chromosome& chromosome, bool) {

    // Calculating the amount of selected features
    const int n_samples = (int)instance.X.size();
    const int n_features = (int)instance.X[0].size();

    double percentage = chromosome[1];
    percentage = max(0.0, min(1.0, percentage));

    int selected_vars = (int)round(n_features * percentage);
    selected_vars = min(selected_vars, n_features);

    if (selected_vars < 2) return 1e6;

    // Ranking the features
    vector<pair<double, int>> ranking;
    ranking.reserve(n_features);

    for (int i = 0; i < n_features; ++i)
        ranking.emplace_back(chromosome[i + 2], i);

    sort(ranking.begin(), ranking.end()); // menor = melhor

    vector<int> selectedCols;
    selectedCols.reserve(selected_vars);

    for (int i = 0; i < selected_vars; ++i)
        selectedCols.push_back(ranking[i].second);

    // Filtering the dataset
    vector<vector<double>> X_sel(n_samples, vector<double>(selected_vars));

    for (int r = 0; r < n_samples; ++r)
        for (int c = 0; c < selected_vars; ++c)
            X_sel[r][c] = instance.X[r][selectedCols[c]];

    set<vector<double>> uniqueRows(X_sel.begin(), X_sel.end());
    int unique_samples = (int)uniqueRows.size();

    if (unique_samples < 2) return 1e6;

    // finding k
    double k_real = chromosome[0];
    k_real = std::max(0.0, std::min(1.0, k_real));

    int kmax = unique_samples / 2;
    if (kmax < 2) return 1e6;

    // ---- exponencial aqui ----
    double alpha = 1.5;  // parâmetro de viés

    double numerator   = std::exp(alpha * k_real) - 1.0;
    double denominator = std::exp(alpha) - 1.0;

    double normalized = numerator / denominator;

    int k = 2 + (int)((kmax - 2) * normalized + 0.5);
    k = std::min(k, kmax);

    if (k <= 1) return 1e6;

    // k-means
    vector<int> labels;

    if (n_samples > 1000)
        labels = minibatch_kmeans(X_sel, k);
    else
        labels = kmeans(X_sel, k);

    if ((int)labels.size() != n_samples) return 1e6;

    set<int> uniqueLabels(labels.begin(), labels.end());
    if ((int)uniqueLabels.size() < k) return 1e6;

    // fitness
    // Nota: BRKGA minimiza — todas as métricas são convertidas para "menor = melhor"
    // sil: [-1, 1]   → maior é melhor → retorna -score
    // db:  [0, +inf]  → menor é melhor → retorna score
    // ch:  [0, +inf]  → maior é melhor → retorna -score (normalizado por log para estabilidade)
    double score = 0.0;

    if (metric == "sil") {
        score = silhouetteScore(X_sel, labels, k);
        return -score;
    }
    else if (metric == "db") {
        score = daviesBouldinIndex(X_sel, labels, k);
        return score;
    }
    else if (metric == "ch") {
        score = calinskiHarabaszIndex(X_sel, labels, k);
        // CH pode crescer na casa dos milhares dependendo do dataset;
        // log suaviza a escala e mantém a comparabilidade entre gerações
        if (score <= 0.0) return 1e6;
        return -std::log(score);
    }
    else {
        // métrica desconhecida — fallback para silhueta com aviso
        std::cerr << "[HybridDecoderV2] Métrica desconhecida: \"" << metric
                  << "\". Usando silhouette como fallback.\n";
        score = silhouetteScore(X_sel, labels, k);
        return -score;
    }
}