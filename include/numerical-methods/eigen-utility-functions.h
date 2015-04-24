#ifndef NUMERICAL_METHODS_EIGEN_UTILITY_FUNCTIONS_H_
#define NUMERICAL_METHODS_EIGEN_UTILITY_FUNCTIONS_H_

#include <Eigen/Dense>
#include <glog/logging.h>

namespace numerical_methods {

// Vector plus scalar.
template <typename Type, int Size>
Eigen::Matrix<Type, Size, 1> operator+(const Eigen::Matrix<Type, Size, 1>& x, 
    Type a) {
  Eigen::Matrix<Type, Size, 1> y(x);
  for (std::size_t i = 0; i < y.size(); ++i) {
    y(i) += a;
  }
  return y;
}

// Scalar plus vector.
template <typename Type, int Size>
Eigen::Matrix<Type, Size, 1> operator+(Type a, 
    const Eigen::Matrix<Type, Size, 1>& x) {
  Eigen::Matrix<Type, Size, 1> y(x);
  for (std::size_t i = 0; i < y.size(); ++i) {
    y(i) += a;
  }
  return y;
}

// Vector minus scalar.
template <typename Type, int Size>
Eigen::Matrix<Type, Size, 1> operator-(const Eigen::Matrix<Type, Size, 1>& x, 
    Type a) {
  Eigen::Matrix<Type, Size, 1> y(x);
  for (std::size_t i = 0; i < y.size(); ++i) {
    y(i) -= a;
  }
  return y;
}

// Scalar minus vector.
template <typename Type, int Size>
Eigen::Matrix<Type, Size, 1> operator-(Type a,
    const Eigen::Matrix<Type, Size, 1>& x) {
  Eigen::Matrix<Type, Size, 1> y(- x);
  for (std::size_t i = 0; i < y.size(); ++i) {
    y(i) += a;
  }
  return y;
}

} // namespace numerical_methods

#endif // NUMERICAL_METHODS_EIGEN_UTILITY_FUNCTIONS_H_
