#include <cmath>
#include <functional>
#include <initializer_list>

#include <Eigen/Dense>
#include <gtest/gtest.h>

#include "numerical-methods/common-definitions.h"
#include "numerical-methods/optimization/optimization-method.h"
#include "numerical-methods/optimization/nelder-mead-method.h"

namespace numerical_methods {

// Suite of test problems for function minimization methods, based on 
// Surjanovic and Bingham (2013).
// 
// S. Surjanovic and D. Bingham, "Virtual Library of Simulation Experiments - 
// Test Functions and Datasets: Optimization Test Problems," available online 
// at http://www.sfu.ca/~ssurjano/optimization.html [accessed on April, 2015], 
// Simon Fraser University (2013).
template <typename Type, int Size>
class TestSuite {
public:
  TestSuite(int dimension, Type tolerance) : dimension_(dimension), 
      tolerance_(tolerance) {
    if (Size != Eigen::Dynamic) {
      CHECK_EQ(dimension, Size) << "Dimension must be consistent.";
    } else {
      CHECK_GT(dimension, 0) << "Dimension must be positive.";
    }
    CHECK_GE(tolerance, 0.0) << "Tolerance must be non-negative.";
    initialize();
  }
  class Problem {
  public:
    Problem(int dimension, 
        const std::function<Type(const Eigen::Matrix<Type, Size, 1>&)>& 
            function, 
        const Eigen::Matrix<Type, Size, 1>& init_guess, 
        const Eigen::Matrix<Type, Size, 1>& glob_minimum, 
        Type param_tolerance) : dimension_(dimension), 
                                function_(function), 
                                init_guess_(init_guess), 
                                glob_minimum_(glob_minimum), 
                                param_tolerance_(param_tolerance) {
      if (Size != Eigen::Dynamic) {
        CHECK_EQ(dimension, Size) << "Dimension must be consistent.";
      } else {
        CHECK_GT(dimension, 0) << "Dimension must be positive.";
      }
    }
    int getDimension() const {
      return dimension_;
    }
    const std::function<Type(const Eigen::Matrix<Type, Size, 1>&)>& 
        getFunction() const {
      return function_;
    }
    const Eigen::Matrix<Type, Size, 1>& getInitGuess() const {
      return init_guess_;
    }
    const Eigen::Matrix<Type, Size, 1>& getGlobMinimum() const {
      return glob_minimum_;
    }
    Type getParamTolerance() const {
      return param_tolerance_;
    }
    typedef Type type;
  private:
    const int dimension_;
    const std::function<Type(const Eigen::Matrix<Type, Size, 1>&)> function_;
    const Eigen::Matrix<Type, Size, 1> init_guess_;
    const Eigen::Matrix<Type, Size, 1> glob_minimum_;
    const Type param_tolerance_;
  };
  std::size_t getNumProblems() const {
    return problems_.size();
  }
  const Problem& getProblem(std::size_t i) const {
    CHECK_LT(i, problems_.size());
    return problems_[i];
  }
private:
  int dimension_;
  Type tolerance_;
  std::vector<Problem> problems_;
  void initialize() {
    
    std::function<Type(const Eigen::Matrix<Type, Size, 1>&)> function;
    Eigen::Matrix<Type, Size, 1> init_guess(dimension_);
    Eigen::Matrix<Type, Size, 1> glob_minimum(dimension_);
      
    // Rosenbrock function in S. Surjanovic and D. Bingham (2013).
    function = [this](const Eigen::Matrix<Type, Size, 1>& x) -> Type {
      Type value = 0.0;
      for (int i = 1; i < this->dimension_; ++i) {
        value += 100.0 * std::pow(x(i) - std::pow(x(i - 1), 2), 2) 
            + std::pow(x(i - 1) - 1.0, 2);
      }
      return value;
    };
    init_guess.fill(- 1.0);
    glob_minimum.setOnes();
    problems_.push_back(Problem(dimension_, function, init_guess, 
        glob_minimum, tolerance_));
    
    // Rotated ellipsoid function in S. Surjanovic and D. Bingham (2013).
    function = [this](const Eigen::Matrix<Type, Size, 1>& x) -> Type {
      Type value = 0.0;
      for (int i = 0; i < this->dimension_; ++i) {
        for (int j = 0; j <= i; ++j) {
          value += std::pow(x(j), 2);
        }
      }
      return value;
    };
    for (int i = 0; i < dimension_; ++i) {
      init_guess(i) = std::pow(- 1.0, i);
    }
    glob_minimum.setZero();
    problems_.push_back(Problem(dimension_, function, init_guess, 
        glob_minimum, tolerance_));
    
    // Sum-of-powers function in S. Surjanovic and D. Bingham (2013).
    function = [this](const Eigen::Matrix<Type, Size, 1>& x) -> Type {
      Type value = 0.0;
      for (int i = 0; i < this->dimension_; ++i) {
        value += std::pow(std::abs(x(i)), i + 2);
      }
      return value;
    };
    init_guess.setOnes();
    glob_minimum.setZero();
    problems_.push_back(Problem(dimension_, function, init_guess, 
        glob_minimum, tolerance_));
    
    // Zakharov function in S. Surjanovic and D. Bingham (2013).
    function = [this](const Eigen::Matrix<Type, Size, 1>& x) -> Type {
      Type value = 0.0;
      Type increment = 0.0;
      for (int i = 0; i < this->dimension_; ++i) {
        value += static_cast<Type>(i + 1) * x(i) / 2.0;
        increment += std::pow(x(i), 2);
      }
      value = std::pow(value, 2);
      value *= value + 1.0;
      value += increment;
      return value;
    };
    for (int i = 0; i < dimension_; ++i) {
      init_guess(i) = std::pow(- 1.0, i + 1);
    }
    glob_minimum.setZero();
    problems_.push_back(Problem(dimension_, function, init_guess, 
        glob_minimum, tolerance_));
    
    // Dixon-Price function in S. Surjanovic and D. Bingham (2013).
    function = [this](const Eigen::Matrix<Type, Size, 1>& x) -> Type {
      Type value = std::pow(x(0) - 1.0, 2);
      for (int i = 1; i < this->dimension_; ++i) {
        value += static_cast<Type>(i + 1) * std::pow(2.0 * std::pow(x(i), 2) 
            - x(i - 1), 2);
      }
      return value;
    };
    init_guess.setOnes();
    for (int i = 0; i < dimension_; ++i) {
      glob_minimum(i) = std::pow(2.0, 1.0 / std::pow(2.0, i) - 1.0);
    }
    problems_.push_back(Problem(dimension_, function, init_guess, 
        glob_minimum, tolerance_));
    
    // Styblinski-Tang function in S. Surjanovic and D. Bingham (2013).
    function = [this](const Eigen::Matrix<Type, Size, 1>& x) -> Type {
      Type value = 0.0;
      for (int i = 0; i < this->dimension_; ++i) {
        value += (std::pow(x(i), 4) - 16.0 * std::pow(x(i), 2) 
            + 5.0 * x(i)) / 2.0;
      }
      return value;
    };
    init_guess.fill(- 1.0);
    glob_minimum.fill(- 2.9035);
    problems_.push_back(Problem(dimension_, function, init_guess, 
        glob_minimum, tolerance_));
    
  }
};

template <class Method>
class OptimizationMethodTest : public testing::Test {
public:
  typedef typename Method::type type;
  static constexpr int size = Method::size;
protected:
  virtual void SetUp() {}
  const std::vector<int> dimensions = {2, 3, 4, 5};
  const typename Method::type tolerance = 1.0e-2;
  const typename Method::type min_iterations = 10;
  const typename Method::type max_iterations = 2000;
  const typename Method::type param_tolerance = 1.0e-9;
  const typename Method::type func_tolerance = 1.0e-6;
};

template <class Method>
class StaticOptimizationMethodTest : public OptimizationMethodTest<Method> {
protected:
  const int dimension = Method::size;
};

typedef testing::Types<NelderMeadMethod<double, 2>, 
                       NelderMeadMethod<double, 3>, 
                       NelderMeadMethod<double, 4>, 
                       NelderMeadMethod<double, 5>> StaticTypes;
                       
TYPED_TEST_CASE(StaticOptimizationMethodTest, StaticTypes);

// Check that integration method achieves desired error.
TYPED_TEST(StaticOptimizationMethodTest, FindsMinimum) {
  TestSuite<typename TypeParam::type, TypeParam::size> 
      test_suite(this->dimension, this->tolerance);
  for (std::size_t i = 0; i < test_suite.getNumProblems(); ++i) {
    const typename TestSuite<typename TypeParam::type, 
        TypeParam::size>::Problem& problem = test_suite.getProblem(i);
    TypeParam method;
    method.options.setMinIterations(this->min_iterations);
    method.options.setMaxIterations(this->max_iterations);
    method.options.setParamTolerance(this->param_tolerance);
    method.options.setFuncTolerance(this->func_tolerance);
    const Eigen::Matrix<typename TypeParam::type, TypeParam::size, 1> 
        glob_minimum = method.minimize(problem.getFunction(), 
                                       problem.getInitGuess());
    EXPECT_LT((glob_minimum - problem.getGlobMinimum()).norm(), 
        problem.getParamTolerance());
  }
}

template <class Method>
class DynamicOptimizationMethodTest : public OptimizationMethodTest<Method> {
protected:
  const std::vector<int> dimensions = {2, 3, 4, 5};
};

typedef testing::Types<NelderMeadMethod<double, Eigen::Dynamic>> DynamicTypes;

TYPED_TEST_CASE(DynamicOptimizationMethodTest, DynamicTypes);

// Check that integration method achieves desired error.
TYPED_TEST(DynamicOptimizationMethodTest, FindsMinimum) {
  for (int dimension : this->dimensions) {
    TestSuite<typename TypeParam::type, TypeParam::size> 
        test_suite(dimension, this->tolerance);
    for (std::size_t i = 0; i < test_suite.getNumProblems(); ++i) {
      const typename TestSuite<typename TypeParam::type, 
          TypeParam::size>::Problem& problem = test_suite.getProblem(i);
      TypeParam method(problem.getDimension());
      method.options.setMinIterations(this->min_iterations);
      method.options.setMaxIterations(this->max_iterations);
      method.options.setParamTolerance(this->param_tolerance);
      method.options.setFuncTolerance(this->func_tolerance);
      const Eigen::Matrix<typename TypeParam::type, TypeParam::size, 1> 
          glob_minimum = method.minimize(problem.getFunction(), 
                                         problem.getInitGuess());
      EXPECT_LT((glob_minimum - problem.getGlobMinimum()).norm(), 
          problem.getParamTolerance());
    }
  }
}

} // namespace numerical_methods

int main(int num_arguments, char** arguments) {
  testing::InitGoogleTest(&num_arguments, arguments);
  return RUN_ALL_TESTS();
}
