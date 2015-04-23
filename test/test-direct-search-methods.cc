#include <cmath>
#include <functional>
#include <initializer_list>

#include <Eigen/Dense>
#include <gtest/gtest.h>

#include "numerical-methods/common-definitions.h"
#include "numerical-methods/direct-search/direct-search-method.h"
#include "numerical-methods/direct-search/nelder-mead-method.h"

namespace numerical_methods {

template <typename Type, int Size>
Eigen::Matrix<Type, Size, 1> 
    vectorize(const std::initializer_list<Type>& list) {
  Eigen::Matrix<Type, Size, 1> vector;
  if (Size == Eigen::Dynamic) {
    vector.resize(list.size());
  }
  std::size_t i = 0;
  for (const Type& element : list) {
    vector(i++) = element;
  }
  return vector;
}

template <typename Type, int Size>
struct Problem {
using vector = Eigen::Matrix<Type, Size, 1>;
template <bool Static = Size != Eigen::Dynamic>
Problem(const std::function<Type(const vector&)>& function, 
    const vector& initial_guess, const vector& minimum, Type tolerance, 
    typename std::enable_if<Static>::type* = nullptr) : dimension(Size), 
    function(function), initial_guess(initial_guess), minimum(minimum), 
    tolerance(tolerance) {}
template <bool Dynamic = Size == Eigen::Dynamic>
Problem(int dimension, const std::function<Type(const vector&)>& function, 
    const vector& initial_guess, const vector& minimum, Type tolerance, 
    typename std::enable_if<Dynamic>::type* = nullptr) : dimension(dimension), 
    function(function), initial_guess(initial_guess), minimum(minimum), 
    tolerance(tolerance) {
  CHECK_GT(dimension, 0) << "Dimension must be positive.";
}
typedef Type type;
const int dimension;
const std::function<Type(const vector&)> function;
const vector initial_guess;
const vector minimum;
const Type tolerance;
};

// Create suite of integration test problems. Based on More et al. (1981).
// 
// J. J. More, B. S. Garbow and K. E. Hillstrom, "Testing Unconstrained 
// Optimization Software," in ACM Transactions on Mathematical Software, 
// vol. 7, no. 1, pp. 136-140 (1981).
template <typename Type, int Size>
std::vector<Problem<Type, Size>> defProblems() {
  using vector = Eigen::Matrix<Type, Size, 1>;
  std::vector<Problem<Type, Size>> problems;
  {
    
    // Rosenbrock function in More et al. (1981).
    const std::function<Type(const vector&)> 
        function = [](const vector& x) -> Type {
      vector y;
      if (Size == Eigen::Dynamic) {
        y.resize(2);
      }
      y(0) = 10.0 * (x(1) - std::pow(x(0), 2));
      y(1) = 1.0 - x(0);
      return y.squaredNorm();
    };
    const int dimension = 2;
    const vector initial_guess = vectorize<Type, Size>({0.0, 1.0});
    const vector minimum = vectorize<Type, Size>({1.0, 1.0});
    const Type tolerance = 1.0e-3;
    Problem<Type, Size> problem(dimension, function, initial_guess, minimum, 
        tolerance);
    problems.push_back(problem);
    
  }/*
  {
    
    // Freudenstein and Roth function in More et al. (1981).
    const std::function<Type(const vector&)>
        function = [](const vector& x) -> Type {
      vector y;
      if (Size == Eigen::Dynamic) {
        y.resize(2);
      }
      y(0) = - 13.0 + x(0) + ((5.0 - x(1)) * x(1) - 2.0) * x(1);
      y(1) = - 29.0 + x(0) + ((1.0 + x(1)) * x(1) - 14.0) * x(1);
      return y.squaredNorm();
    };
    const int dimension = 2;
    const vector initial_guess = vectorize<Type, Size>({0.5, - 2.0});
    const vector minimum = vectorize<Type, Size>({5.0, 4.0});
    const Type tolerance = 1.0e-3;
    Problem<Type, Size> problem(dimension, function, initial_guess, minimum, 
        tolerance);
    problems.push_back(problem);
    
  }*/
  return problems;
}

template <class Method>
class DirectSearchMethodTest : public testing::Test {
public:
  typedef typename Method::type type;
  static constexpr int size = Method::size;
protected:
  virtual void SetUp() {
    problems = defProblems<typename Method::type, Method::size>();
  }
  std::vector<Problem<typename Method::type, Method::size>> problems;
  const typename Method::type min_iterations = 10;
  const typename Method::type max_iterations = 1000;
  const typename Method::type abs_tolerance = 1.0e-16;
  const typename Method::type rel_tolerance = 1.0e-16;
  const typename Method::type init_scale = 1.0;
};

typedef testing::Types<NelderMeadMethod<double, Eigen::Dynamic>> Types;

TYPED_TEST_CASE(DirectSearchMethodTest, Types);

// Check that integration method achieves desired error.
TYPED_TEST(DirectSearchMethodTest, FindsMinimum) {
  for (const Problem<typename TypeParam::type, TypeParam::size>& 
       problem : this->problems) {
    TypeParam method(problem.dimension);
    method.options.setMinIterations(this->min_iterations);
    method.options.setMaxIterations(this->max_iterations);
    method.options.setAbsTolerance(this->abs_tolerance);
    method.options.setRelTolerance(this->rel_tolerance);
    method.options.setInitScale(this->init_scale);
    const Eigen::Matrix<typename TypeParam::type, TypeParam::size, 1> 
        minimum = method.minimize(problem.function, problem.initial_guess);
    EXPECT_LT((minimum - problem.minimum).norm(), problem.tolerance);
  }
}

} // namespace numerical_methods

int main(int num_arguments, char** arguments) {
  testing::InitGoogleTest(&num_arguments, arguments);
  return RUN_ALL_TESTS();
}
