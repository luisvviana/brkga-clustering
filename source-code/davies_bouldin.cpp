#include "../headers/davies_bouldin.hpp"
#include <cmath>
#include <limits>
#include <vector>

using namespace std;

// Distância Euclidiana entre dois vetores
static double euclideanDist(const vector<double>& a, const vector<double>& b) {
    double dist = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        double diff = a[i] - b[i];
        dist += diff * diff;
    }
    return sqrt(dist);
}

double daviesBouldinIndex(const vector<vector<double>>& data,
                          const vector<int>& labels, int k) {
    int n = (int)data.size();
    if (n == 0 || k <= 1) return numeric_limits<double>::max();
    int dim = (int)data[0].size();

    // Calcula centroides
    vector<vector<double>> centroids(k, vector<double>(dim, 0.0));
    vector<int> counts(k, 0);
    for (int i = 0; i < n; ++i) {
        int c = labels[i];
        counts[c]++;
        for (int d = 0; d < dim; ++d)
            centroids[c][d] += data[i][d];
    }
    for (int c = 0; c < k; ++c) {
        if (counts[c] == 0) return numeric_limits<double>::max(); // cluster vazio penaliza
        for (int d = 0; d < dim; ++d)
            centroids[c][d] /= counts[c];
    }

    // Calcula dispersão (S_i)
    vector<double> S(k, 0.0);
    for (int i = 0; i < n; ++i) {
        int c = labels[i];
        S[c] += euclideanDist(data[i], centroids[c]);
    }
    for (int c = 0; c < k; ++c) {
        if (counts[c] > 0)
            S[c] /= counts[c];
        else
            return numeric_limits<double>::max();
    }

    // Índice DB
    double db_index = 0.0;
    for (int i = 0; i < k; ++i) {
        double max_ratio = 0.0;
        for (int j = 0; j < k; ++j) {
            if (i == j) continue;
            double M = euclideanDist(centroids[i], centroids[j]);
            if (M == 0) return numeric_limits<double>::max(); // clusters coincidentes penalizam
            double ratio = (S[i] + S[j]) / M;
            if (ratio > max_ratio) max_ratio = ratio;
        }
        db_index += max_ratio;
    }

    return db_index / k;
}
