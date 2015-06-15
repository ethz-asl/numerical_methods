#include <cmath>
#include <limits>
#include <random>
#include <utility>
#include <vector>

#include <Eigen/Dense>
#include <gtest/gtest.h>

#include "numerical-methods/common-definitions.h"
#include "numerical-methods/utility/linear-model.h"

namespace numerical_methods {

// TODO(gabrieag): Unit tests for linear model.

} // namespace numerical_methods

int main(int num_arguments, char** arguments) {
  testing::InitGoogleTest(&num_arguments, arguments);
  return RUN_ALL_TESTS();
}
