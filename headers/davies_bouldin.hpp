#ifndef DAVIES_BOULDIN_HPP
#define DAVIES_BOULDIN_HPP

#include <vector>

double daviesBouldinIndex(const std::vector<std::vector<double>>& data,
                          const std::vector<int>& labels, int k);

#endif