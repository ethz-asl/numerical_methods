#include <cmath>
#include <functional>

#include <glog/logging.h>
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
std::vector<Problem<Type>> createProblems() {
  std::vector<Problem<Type>> problems;
  {
    const std::function<Type(Type)> function = [](Type x) {
      return x * std::log1p(x);
    };
    const Type a = 0.0, b = 1.0;
    const Type val = 1.0 / 4.0;
    problems.push_back(problem(function, a, b, val));
  }
  {
    const std::function<Type(Type)> function = [](Type x) {
      return (x * x) * std::atan(x);
    };
    const Type a = 0.0, b = 1.0;
    const Type val = (getPi<Type>() - 2.0 + 2.0 * std::log(2.0)) / 12.0;
    problems.push_back(problem(function, a, b, val));
  }
  return problems;
}

} // namespace numerical_methods
