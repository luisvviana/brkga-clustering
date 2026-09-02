#ifndef INSTANCE_HPP
#define INSTANCE_HPP

#include <vector>
#include <algorithm>

class HybridInstance
{
public:
    std::vector<std::vector<double>> X;
    int kmax;
    double lambda_k;

    HybridInstance() : lambda_k(0.01), kmax(2), X() {}

    HybridInstance(const std::vector<std::vector<double>>& X, double lambda_k = 0.01)
        : X(X), lambda_k(lambda_k)
    {
        int n_samples = static_cast<int>(X.size());
        kmax = std::max(2, n_samples / 2);
    }
};

#endif // INSTANCE_HPP
