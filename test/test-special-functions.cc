#include <cmath>

#include <gtest/gtest.h>

#include "numerical-methods/common-definitions.h"
#include "numerical-methods/special-functions/special-function.h"
#include "numerical-methods/special-functions/incomplete-beta-function.h"

namespace numerical_methods {

template <class Function>
class SpecialFunctionTest : public testing::Test {
public:
  typedef typename Function::type type;
protected:
  virtual void SetUp() {}
  static const std::vector<typename Function::type> accuracies;
};

template <class Function>
const std::vector<typename Function::type> 
    SpecialFunctionTest<Function>::accuracies = 
        {1.0e-3, 1.0e-6, 1.0e-9, 1.0e-12};

template <class Function>
class IncompleteBetaFunctionTest : public SpecialFunctionTest {
public:
  typedef typename Function::type type;
protected:
  virtual void SetUp() {}
  static const std::vector<typename Function::type> parameters;
  static const std::vector<typename Function::type> arguments;
};

template <class Function>
const std::vector<typename Function::type> 
    IncompleteBetaFunctionTest<Function>::parameters = 
        {0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0};

template <class Function>
const std::vector<typename Function::type> 
    IncompleteBetaFunctionTest<Function>::arguments = 
        {0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0};

typedef testing::Types<IncompleteBetaFunction<double>> Types;

TYPED_TEST_CASE(IncompleteBetaFunctionTest, Types);

// Check that incomplete beta function values are within desired accuracy.
TYPED_TEST(IncompleteBetaFunctionTest, AchievesDesiredAccuracy) {
  using Type = typename TypeParam::type;
  for (const Type accuracy : this->accuracies) {
    for (const Type a : this->parameters) {
      for (const Type b : this->parameters) {
        IncompleteBetaFunction function(a, b, accuracy);
        for (const Type argument : this->arguments) {
          const Type value = function.evaluate(argument);
          Type y; // TODO(gabrieag): Find a way to evaluate.
          EXPECT_LT(std::abs(value - y), tolerance + problem.margin);
        }
      }
    }
    TypeParam function(accuracy);
    for (const TestProblem<Type>& problem : this->problems) {
      Type value = method.integrate(problem.function, problem.a, problem.b);
      EXPECT_LT(std::abs(value - problem.value), tolerance + problem.margin);
    }
  }
}

} // namespace numerical_methods

int main(int num_arguments, char** arguments) {
  testing::InitGoogleTest(&num_arguments, arguments);
  return RUN_ALL_TESTS();
}
