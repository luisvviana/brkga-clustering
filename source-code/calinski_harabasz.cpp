#include "../headers/calinski_harabasz.hpp"
#include <cmath>
#include <limits>
#include <vector>

using namespace std;

static double euclideanDistSq(const vector<double>& a, const vector<double>& b) {
    double dist = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        double diff = a[i] - b[i];
        dist += diff * diff;
    }
    return dist;
}

double calinskiHarabaszIndex(const vector<vector<double>>& data,
                              const vector<int>& labels, int k) {
    int n = (int)data.size();
    if (n == 0 || k <= 1 || k >= n) return 0.0;
    int dim = (int)data[0].size();

    // Validações defensivas
    for (int i = 1; i < n; ++i)
        if ((int)data[i].size() != dim)
            return 0.0;

    for (int i = 0; i < n; ++i)
        if (labels[i] < 0 || labels[i] >= k)
            return 0.0;

    // Centroide global
    vector<double> globalCentroid(dim, 0.0);
    for (int i = 0; i < n; ++i)
        for (int d = 0; d < dim; ++d)
            globalCentroid[d] += data[i][d];
    for (int d = 0; d < dim; ++d)
        globalCentroid[d] /= n;

    // Centroides por cluster
    vector<vector<double>> centroids(k, vector<double>(dim, 0.0));
    vector<int> counts(k, 0);
    for (int i = 0; i < n; ++i) {
        int c = labels[i];
        counts[c]++;
        for (int d = 0; d < dim; ++d)
            centroids[c][d] += data[i][d];
    }
    for (int c = 0; c < k; ++c) {
        if (counts[c] == 0) return 0.0; // cluster vazio
        for (int d = 0; d < dim; ++d)
            centroids[c][d] /= counts[c];
    }

    // BCSS — Between-Cluster Sum of Squares
    // Soma ponderada da distância de cada centroide ao centroide global
    double BCSS = 0.0;
    for (int c = 0; c < k; ++c)
        BCSS += counts[c] * euclideanDistSq(centroids[c], globalCentroid);

    // WCSS — Within-Cluster Sum of Squares
    // Soma das distâncias de cada ponto ao seu centroide
    double WCSS = 0.0;
    for (int i = 0; i < n; ++i)
        WCSS += euclideanDistSq(data[i], centroids[labels[i]]);

    if (WCSS < 1e-12) return numeric_limits<double>::max(); // clusters perfeitamente compactos

    // CH = (BCSS / (k-1)) / (WCSS / (n-k))
    return (BCSS / (k - 1)) / (WCSS / (n - k));
}