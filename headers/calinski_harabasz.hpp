#ifndef CALINSKI_HARABASZ_HPP
#define CALINSKI_HARABASZ_HPP

#include <vector>

double calinskiHarabaszIndex(const std::vector<std::vector<double>>& data,
                             const std::vector<int>& labels, int k);

#endif