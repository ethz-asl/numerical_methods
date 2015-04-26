#include <cmath>
#include <functional>
#include <initializer_list>

#include <Eigen/Dense>
#include <gtest/gtest.h>

#include "numerical-methods/common-definitions.h"
#include "numerical-methods/direct-search/direct-search-method.h"
#include "numerical-methods/direct-search/nelder-mead-method.h"

namespace numerical_methods {

// Create suite of test problems.
template <typename Type, int Size>
class TestSuite {
public:
  template <bool Static = Size != Eigen::Dynamic>
  TestSuite(Type tolerance, 
      typename std::enable_if<Static>::type* = nullptr) : 
      dimension_(Size), tolerance_(tolerance) {
    CHECK_GE(tolerance, 0.0) << "Tolerance must be non-negative.";
    initialize();
  }
  template <bool Dynamic = Size == Eigen::Dynamic>
  TestSuite(int dimension, Type tolerance, 
      typename std::enable_if<Dynamic>::type* = nullptr) : 
      dimension_(dimension), tolerance_(tolerance) {
    CHECK_GT(dimension, 0) << "Dimension must be positive.";
    CHECK_GE(tolerance, 0.0) << "Tolerance must be non-negative.";
    initialize();
  }
  class Problem {
  public:
    template <bool Static = Size != Eigen::Dynamic>
    Problem(
        const std::function<Type(const Eigen::Matrix<Type, Size, 1>&)>& 
            function, 
        const Eigen::Matrix<Type, Size, 1>& init_guess, 
        const Eigen::Matrix<Type, Size, 1>& glob_minimum, 
        Type param_tolerance, 
        typename std::enable_if<Static>::type* = nullptr) : 
            dimension_(Size), 
            function_(function), 
            init_guess_(init_guess), 
            glob_minimum_(glob_minimum), 
            param_tolerance_(param_tolerance) {}
    template <bool Dynamic = Size == Eigen::Dynamic>
    Problem(int dimension, 
        const std::function<Type(const Eigen::Matrix<Type, Size, 1>&)>& 
            function, 
        const Eigen::Matrix<Type, Size, 1>& init_guess, 
        const Eigen::Matrix<Type, Size, 1>& glob_minimum, 
        Type param_tolerance, 
        typename std::enable_if<Dynamic>::type* = nullptr) : 
            dimension_(dimension), 
            function_(function), 
            init_guess_(init_guess), 
            glob_minimum_(glob_minimum), 
            param_tolerance_(param_tolerance) {
      CHECK_GT(dimension, 0) << "Dimension must be positive.";
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
      
    // Rosenbrock function.
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
    
    // Rotated ellipsoid function.
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
    
    // Sum-of-powers function.
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
    
    /*
    
    TODO(gabrieag): Remove this function.
    
    // Perm function.
    function = [this](const Eigen::Matrix<Type, Size, 1>& x) -> Type {
      Type value = 0.0;
      for (int i = 0; i < this->dimension_; ++i) {
        Type increment = 0.0;
        for (int j = 0; j < this->dimension_; ++j) {
          increment += static_cast<Type>(std::pow(j + 1, i + 1)) 
              * (std::pow(x(j) / static_cast<Type>(j + 1), i + 1) - 1.0);
        }
        value += std::pow(increment, 2);
      }
      return value;
    };
    init_guess.fill(- 1.0);
    for (int i = 0; i < dimension_; ++i) {
      glob_minimum(i) = static_cast<Type>(i + 1);
    }
    problems_.push_back(Problem(dimension_, function, init_guess, 
        glob_minimum, tolerance_));
    
    */
    
    // Zakharov function.
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
    
    // Dixon-Price function.
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
    
    // Styblinski-Tang function.
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
class DirectSearchMethodTest : public testing::Test {
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

typedef testing::Types<NelderMeadMethod<double, Eigen::Dynamic>> Types;

TYPED_TEST_CASE(DirectSearchMethodTest, Types);

// Check that integration method achieves desired error.
TYPED_TEST(DirectSearchMethodTest, FindsMinimum) {
  if (TypeParam::size != Eigen::Dynamic) {
    /*
    
    TODO(gabrieag): Add static test cases.
    
    TestSuite<typename TypeParam::type, TypeParam::size> 
        test_suite(this->tolerance);
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
    
    */
  } else {
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
}

} // namespace numerical_methods

int main(int num_arguments, char** arguments) {
  testing::InitGoogleTest(&num_arguments, arguments);
  return RUN_ALL_TESTS();
}
