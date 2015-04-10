#include <cmath>
#include <functional>

#include <gtest/gtest.h>

#include "numerical-methods/common-definitions.h"
#include "numerical-methods/integration/integration-method.h"
#include "numerical-methods/integration/log-tanh-sinh-method.h"

namespace numerical_methods {

template <typename Type>
struct Problem {
Problem(const std::function<Type(Type)>& function, Type a, Type b, Type value, 
    Type tolerance) : function(function), a(a), b(b), value(value), 
    tolerance(tolerance) {};
typedef Type type;
const std::function<Type(Type)> function;
const Type a, b;
const Type value;
const Type tolerance;
};

template <typename Type>
std::vector<Problem<Type>> defProblems() {
  std::vector<Problem<Type>> problems;
  {
    
    const std::function<Type(Type)> function = [](Type x) -> Type {
      return std::log(x) + std::log(std::log1p(x));
    };
    const Type a = 0.0, b = 1.0;
    const Type value = std::log(1.0 / 4.0);
    const Type tolerance = 0.0;
    Problem<Type> problem(function, a, b, value, tolerance);
    problems.push_back(problem);
    
  }
  return problems;
}

template <class Method>
class IntegrationMethodTest : public testing::Test {
public:
  typedef typename Method::type type;
protected:
  virtual void SetUp() {
    problems = defProblems<typename Method::type>();
  }
  std::vector<Problem<typename Method::type>> problems;
  const std::vector<typename Method::type> 
      errors = {1.0e-3, 1.0e-6, 1.0e-9, 1.0e-12};
};

// TODO(gabrieag): Find out why test fails with quadruple-precision.
typedef testing::Types<LogTanhSinhMethod<double>> Types;

TYPED_TEST_CASE(IntegrationMethodTest, Types);

// Check that integration method achieves desired error.
TYPED_TEST(IntegrationMethodTest, AchievesDesiredError) {
  for (typename TypeParam::type error : this->errors) {
    TypeParam method(error);
    for (const Problem<typename TypeParam::type>& problem : this->problems) {
      typename TypeParam::type value = method.integrate(problem.function, 
          problem.a, problem.b);
      EXPECT_LT(std::abs(value - problem.value), error + problem.tolerance);
    }
  }
}

} // namespace numerical_methods

int main(int num_arguments, char** arguments) {
  testing::InitGoogleTest(&num_arguments, arguments);
  return RUN_ALL_TESTS();
}
