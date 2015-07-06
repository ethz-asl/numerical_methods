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

// Sample for testing linear model.
template <typename Type, int Size>
class TestSample {
  // TODO(gabrieag): Implement.
};

template <class Model>
class LinearModelTest : public testing::Test {
public:
  typedef typename Model::type type;
  static constexpr int size = Model::size;
protected:
  virtual void SetUp() {}
  static const std::vector<int> sizes;
  static const std::vector<typename Model::type> discounts;
  static const std::vector<typename Model::type> regularizers;
  static const typename Model::type tolerance;
};

template <class Model>
const std::vector<int> 
    LinearModelTest<Model>::sizes = 
        {1, 2, 5, 10, 20, 50, 100, 200, 500, 1000};

template <class Model>
const std::vector<typename Model::type> 
    LinearModelTest<Model>::discounts = 
        {0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0};

template <class Model>
const std::vector<typename Model::type> 
    LinearModelTest<Model>::regularizers = {0.0, 0.01, 0.1, 1.0};

template <class Model>
const typename Model::type LinearModelTest<Model>::tolerance = 
    std::sqrt(std::numeric_limits<typename Model::type>::epsilon());

template <class Model>
class StaticLinearModelTest : public LinearModelTest<Model> {
  static_assert(Model::size != Eigen::Dynamic, "");
protected:
  static constexpr int dimension = Model::size;
};

typedef testing::Types<LinearModel<double, 1>, 
                       LinearModel<double, 2>, 
                       LinearModel<double, 5>, 
                       LinearModel<double, 10>> StaticTypes;

TYPED_TEST_CASE(StaticLinearModelTest, StaticTypes);

// TODO(gabrieag): Test case for linear model.
TYPED_TEST(StaticLinearModelTest, TestCase) {
  // TODO(gabrieag): Design test case and implement.
}

template <class Model>
class DynamicLinearModelTest : public LinearModelTest<Model> {
  static_assert(Model::size == Eigen::Dynamic, "");
protected:
  static const std::vector<int> dimensions;
};

template <class Model>
const std::vector<int> 
    DynamicLinearModelTest<Model>::dimensions = {1, 2, 5, 10};

typedef testing::Types<LinearModel<double, Eigen::Dynamic>> DynamicTypes;

TYPED_TEST_CASE(DynamicLinearModelTest, DynamicTypes);

// TODO(gabrieag): Test case for linear model.
TYPED_TEST(DynamicLinearModelTest, TestCase) {
  // TODO(gabrieag): Design test case and implement.
}

} // namespace numerical_methods

int main(int num_arguments, char** arguments) {
  testing::InitGoogleTest(&num_arguments, arguments);
  return RUN_ALL_TESTS();
}
