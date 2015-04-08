#include <cmath>
#include <functional>

#include <gtest/gtest.h>

#include "numerical-methods/common-definitions.h"
#include "numerical-methods/integration/integration-method.h"
#include "numerical-methods/integration/tanh-sinh-method.h"

namespace numerical_methods {

template <typename Type>
struct Problem {
Problem(const std::function<Type(Type)>& fun, Type a, Type b, Type val) : 
    fun(fun), a(a), b(b), val(val) {};
typedef Type type;
const std::function<Type(Type)>& fun;
const Type a, b;
const Type val;
};

template <typename Type>
class IntegrationMethodTest : public testing::Test {
protected:
  virtual void SetUp() {
    {
      const std::function<Type(Type)> function = [](Type x) {
        return x * std::log1p(x);
      };
      const Type a = 0.0, b = 1.0;
      const Type val = 1.0 / 4.0;
      Problem<Type> problem(function, a, b, val);
      problems.push_back(problem);
    }
    {
      const std::function<Type(Type)> function = [](Type x) {
        return (x * x) * std::atan(x);
      };
      const Type a = 0.0, b = 1.0;
      const Type val = (getPi<Type>() - 2.0 + 2.0 * std::log(2.0)) / 12.0;
      Problem<Type> problem(function, a, b, val);
      problems.push_back(problem);
    }
  }
  std::vector<Problem<Type>> problems;
  const std::vector<Type> errors = {1.0e-3, 1.0e-6, 1.0e-9, 1.0e-12};
  const std::size_t levels = 12;
};

typedef testing::Types<float, double, long double> Types;

TYPED_TEST_CASE(IntegrationMethodTest, Types);

TYPED_TEST(IntegrationMethodTest, TanhSinh) {
  for (TypeParam error : this->errors) {
    TanhSinhMethod<TypeParam> method(error, this->levels);
    for (const Problem<TypeParam>& problem : this->problems) {
      TypeParam val = method.integrate(problem.fun, problem.a, problem.b);
      EXPECT_LT(std::abs(val - problem.val), error);
    }
  }
}

} // namespace numerical_methods

int main(int num_arguments, char** arguments) {
  testing::InitGoogleTest(&num_arguments, arguments);
  return RUN_ALL_TESTS();
}
