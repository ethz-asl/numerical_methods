#ifndef NUMERICAL_METHODS_DIRECT_SEARCH_DIRECT_SEARCH_METHOD_H_
#define NUMERICAL_METHODS_DIRECT_SEARCH_DIRECT_SEARCH_METHOD_H_

#include <type_traits>

#include <Eigen/Dense>
#include <glog/logging.h>

#include "numerical-methods/common-definitions.h"

namespace numerical_methods {

template <typename Type, typename Size>
class DirectSearchMethod {
public:
  
  typedef Type type;
  typedef Size size;
  
  template <bool Static = !std::is_same<Size, Eigen::Dynamic>::value>
  DirectSearchMethod(typename std::enable_if<Static>::type* = nullptr) : 
      dimension_(Size) {
    initialize();
  }
  template <bool Dynamic = std::is_same<Size, Eigen::Dynamic>::value>
  explicit DirectSearchMethod(int dimension, 
      typename std::enable_if<Dynamic>::type* = nullptr) : 
      dimension_(dimension) {
    CHECK_GT(dimension, 0) << "Dimension must be positive.";
    initialize();
  }
  virtual ~DirectSearchMethod() {}
  
  // Return dimension.
  inline Size getDimension() const {
    return dimension_;
  }
  
  class Options {
  public:
    Options() : min_iterations_(1), 
                max_iterations_(100), 
                abs_tolerance_(1.0e-6), 
                rel_tolerance_(1.0e-3) {}
    Options(int min_iterations, 
            int max_iterations, 
            Type abs_tolerance, 
            Type rel_tolerance) : min_iterations_(min_iterations), 
                                  max_iterations_(max_iterations), 
                                  abs_tolerance_(abs_tolerance),  
                                  rel_tolerance_(rel_tolerance) {
      CHECK_GT(min_iterations, 0) 
          << "Minimum number of iterations must be positive.";
      CHECK_GT(max_iterations, 0) 
          << "Maximum number of iterations must be positive.";
      CHECK_GT(abs_tolerance, Type(0.0)) 
          << "Absolute tolerance must be positive.";
      CHECK_GT(rel_tolerance, Type(0.0)) 
          << "Relative tolerance must be positive.";
    }
    inline int getMinIterations() const {
      return min_iterations_;
    }
    inline int getMaxIterations() const {
      return max_iterations_;
    }
    inline Type getAbsTolerance() const {
      return abs_tolerance_;
    }
    inline Type getRelTolerance() const {
      return rel_tolerance_;
    }
    inline void setMinIterations(int min_iterations) {
      CHECK_GT(min_iterations, 0) 
          << "Minimum number of iterations must be positive.";
      min_iterations_ = min_iterations;
    }
    inline void setMaxIterations(int max_iterations) {
      CHECK_GT(max_iterations, 0) 
          << "Maximum number of iterations must be positive.";
      max_iterations_ = max_iterations;
    }
    inline void setAbsTolerance(Type abs_tolerance) {
      CHECK_GT(abs_tolerance, Type(0.0)) 
          << "Absolute tolerance must be positive.";
      abs_tolerance_ = abs_tolerance;
    }
    inline void setRelTolerance(Type rel_tolerance) {
      CHECK_GT(rel_tolerance, Type(0.0)) 
          << "Relative tolerance must be positive.";
      rel_tolerance_ = rel_tolerance;
    }
  private:
    int min_iterations_;
    int max_iterations_;
    Type abs_tolerance_;
    Type rel_tolerance_;
  } options;
  
  // Find minimum of function.
  template <class Function>
  void findMinimum(const Function& function, 
      const Eigen::Matrix<Type, Size, 1>& point) const {}
  
protected:
  virtual initialize() {};
  
private:
  
  // Disallow dangerous copy and assignment constructors.
  DirectSearchMethod(const DirectSearchMethod&) = delete;
  DirectSearchMethod& operator=(const DirectSearchMethod&) = delete;
  
  const int dimension_;
  
}; // DirectSearchMethod

} // namespace numerical_methods

#endif // NUMERICAL_METHODS_DIRECT_SEARCH_DIRECT_SEARCH_METHOD_H_
