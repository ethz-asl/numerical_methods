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
    
    // Log-normalization constant of uniform distribution.
    const Type alpha = 2.0, beta = 5.0;
    const std::function<Type(Type)> function = [alpha, beta](Type x) -> Type {
      return 0.0;
    };
    const Type a = alpha, b = beta;
    const Type value = std::log(beta - alpha);
    const Type tolerance = 0.0;
    Problem<Type> problem(function, a, b, value, tolerance);
    problems.push_back(problem);
    
  }
  {
    
    // Log-normalization constant of beta distribution.
    const Type alpha = 5.0, beta = 5.0;
    const std::function<Type(Type)> function = [alpha, beta](Type x) -> Type {
      x = std::min<Type>(x, 1.0 - getEps<Type>());
      x = std::max<Type>(x, getEps<Type>());
      return (alpha - 1.0) * std::log(x) + (beta - 1.0) * std::log(1.0 - x);
    };
    const Type a = 0.0, b = 1.0;
    const Type value = std::lgamma(alpha) + std::lgamma(beta) 
        - std::lgamma(alpha + beta);
    const Type tolerance = 0.0;
    Problem<Type> problem(function, a, b, value, tolerance);
    problems.push_back(problem);
    
  }
  {
    
    // Log-normalization constant of logit normal distribution.
    const Type mu = 0.0, sigma = 2.0;
    const std::function<Type(Type)> function = [mu, sigma](Type x) -> Type {
      x = std::min<Type>(x, 1.0 - getEps<Type>());
      x = std::max<Type>(x, getEps<Type>());
      return - std::log(x * (1.0 - x)) 
          - std::pow((std::log(x / (1.0 - x)) - mu) / sigma, 2) / 2.0;
    };
    const Type a = 0.0, b = 1.0;
    const Type value = std::log(2.0 * getPi<Type>()) / 2.0 + std::log(sigma);
    const Type tolerance = 0.0;
    Problem<Type> problem(function, a, b, value, tolerance);
    problems.push_back(problem);
    
  }
  {
    
    // Log-normalization constant of semicircle distribution.
    const Type rho = 2.0;
    const std::function<Type(Type)> function = [rho](Type x) -> Type {
      x = std::min<Type>(x, rho - getEps<Type>());
      x = std::max<Type>(x, - rho + getEps<Type>());
      return std::log(std::pow(rho, 2) - std::pow(x, 2)) / 2.0;
    };
    const Type a = - rho, b = rho;
    const Type value = std::log(getPi<Type>() / 2.0) + 2.0 * std::log(rho);
    const Type tolerance = 0.0;
    Problem<Type> problem(function, a, b, value, tolerance);
    problems.push_back(problem);
    
  }
  {
    
    // Log-normalization constant of raised cosine distribution.
    const Type mu = 0.0, sigma = 3.0;
    const std::function<Type(Type)> function = [mu, sigma](Type x) -> Type {
      x = std::min<Type>(x, mu + sigma / 2.0 - getEps<Type>());
      x = std::max<Type>(x, mu - sigma / 2.0 + getEps<Type>());
      return std::log1p(std::cos(2.0 * getPi<Type>() * (x - mu) / sigma));
    };
    const Type a = mu - sigma / 2.0, b = mu + sigma / 2.0;
    const Type value = std::log(sigma);
    const Type tolerance = 0.0;
    Problem<Type> problem(function, a, b, value, tolerance);
    problems.push_back(problem);
    
  }
  {
    
    // Log-normalization constant of Kumaraswamy distribution.
    const Type alpha = 5.0, beta = 1.0;
    const std::function<Type(Type)> function = [alpha, beta](Type x) -> Type {
      x = std::min<Type>(x, 1.0 - getEps<Type>());
      x = std::max<Type>(x, getEps<Type>());
      return (alpha - 1.0) * std::log(x) 
          + (beta - 1.0) * std::log(1.0 - std::pow(x, alpha));
    };
    const Type a = 0.0, b = 1.0;
    const Type value = - std::log(alpha) - std::log(beta);
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
