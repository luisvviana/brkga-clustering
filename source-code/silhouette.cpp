#include "../headers/silhouette.hpp"
#include <vector>
#include <cmath>
#include <limits>
#include <algorithm>

using namespace std;

double euclideanDistance(const vector<double>& a, const vector<double>& b) {
    double dist = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        double diff = a[i] - b[i];
        dist += diff * diff;
    }
    return sqrt(dist);
}

double silhouetteScore(const vector<vector<double>>& X,
                       const vector<int>& labels,
                       int k) {
    int n_samples = X.size();
    vector<vector<int>> clusters(k);
    for (int i = 0; i < n_samples; ++i)
        clusters[labels[i]].push_back(i);

    double totalScore = 0.0;

    for (int i = 0; i < n_samples; ++i) {
        int label_i = labels[i];
        const vector<double>& xi = X[i];

        // a: mean intra-cluster distance
        double a = 0.0;
        if (clusters[label_i].size() > 1) {
            for (int j : clusters[label_i]) {
                if (i != j)
                    a += euclideanDistance(xi, X[j]);
            }
            a /= (clusters[label_i].size() - 1);
        }

        // b: mean nearest-cluster distance
        double b = numeric_limits<double>::max();
        for (int l = 0; l < k; ++l) {
            if (l == label_i || clusters[l].empty()) continue;
            double dist = 0.0;
            for (int j : clusters[l]) {
                dist += euclideanDistance(xi, X[j]);
            }
            dist /= clusters[l].size();
            b = min(b, dist);
        }

        // silhouette score
        double s = 0.0;
        if (a < b)
            s = 1.0 - (a / b);
        else if (a > b)
            s = (b / a) - 1.0;

        totalScore += s;
    }

    return totalScore / n_samples;
}