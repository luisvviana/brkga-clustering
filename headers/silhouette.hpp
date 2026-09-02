#ifndef SILHOUETTE_HPP
#define SILHOUETTE_HPP

#include <vector>

double silhouetteScore(const std::vector<std::vector<double>>& X,
                       const std::vector<int>& labels,
                       int k);

#endif