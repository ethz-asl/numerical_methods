#ifndef NUMERICAL_METHODS_DIRECT_SEARCH_NELDER_MEAD_METHOD_H_
#define NUMERICAL_METHODS_DIRECT_SEARCH_NELDER_MEAD_METHOD_H_

#include <cmath>
#include <type_traits>

#include <Eigen/Dense>
#include <glog/logging.h>

#include "numerical-methods/common-definitions.h"
#include "numerical-methods/direct-search/direct-search-method.h"

namespace numerical_methods {

// This class implements multi-dimensional minimization by the downhill simplex 
// search method of Nelder and Mead (1965), as implemented by O'Neill (1971), 
// with subsequent comments by Chambers and Ertel (1974), Benyon (1976) and 
// Hill (1978).
// 
// J. A. Nelder and R. Mead, "A Simplex Method for Function Minimization," The 
// Computer Journal, vol. 7, pp. 308–313 (1965).
// 
// R. O'Neill, "Statistical Algorithms - Algorithm AS 47: Function Minimization 
// using a Simplex Procedure," Applied Statistics, vol. 20, no. 3, pp. 338-345 
// (1971).
// 
// J. M. Chambers and J. E. Ertel, "Statistical Algorithms - Remark AS R11: A 
// Remark on Algorithm AS 47: Function Minimization using a Simplex Procedure," 
// Applied Statistics, vol. 23, no. 2, pp. 250-251 (1974).
// 
// P. R. Benyon, "Statistical Algorithms - Remark AS R15: Function Minimization 
// using a Simplex Procedure," Applied Statistics, vol. 25, no. 1, p. 97 (1976).
// 
// I. D. Hill, "Remark AS R28: A Remark on Algorithm AS 47: Function 
// Minimization using a Simplex Procedure," Journal of the Royal Statistical 
// Society, Series C (Applied Statistics), vol. 27, no. 3, pp. 380-382 (1978).
template <typename Type, typename Size>
class NelderMeadMethod : public DirectSearchMethod {
public:
  
  template <class Function>
  void findMinimum(const Function& function, 
      const Eigen::Matrix<Type, Size, 1>& point) const {
    
    std::vector<std::size_t> indices(dimension + 1);
    for (std::size_t n = 0; n <= dimension; ++n) {
      indices[n] = n;
    }
    
    Eigen::Matrix<Type, Size, 1> centroid;
    centroid.resize(dimension);
    
    // Sort points in simplex according to value.
    std::sort(indices.begin(), indices.end(), 
        [&values_](std::size_t i, std::size_t j) {
          return values_[i] < values_[j];
        });
    
    // Compute centroid.
    for (std::size_t n = 0; n <= dimension; ++n) {
      centroid += points_.col(n);
    }
    centroid /= dimension;
    
    // Store best, second-worst and worst points.
    const std::size_t i = indices[0];
    const std::size_t j = indices[dimension - 1];
    const std::size_t k = indices[dimension];
    
    // Reflect point.
    Eigen::Matrix<Type, Size, 1> 
        ref_point = centroid + alpha * (centroid - points_.col(k));
    Type ref_value = function(ref_point);
    if ((ref_value < values_[j]) && (ref_value > values_[i])) {
      points_.col(k) = ref_point;
      values_[k] = ref_value;
    } else if (ref_value < values_[i]) {
      
      // Expand point.
      Eigen::Matrix<Type, Size, 1> 
          exp_point = centroid + gamma * (centroid - points_.col(k));
      Type exp_value = function(exp_point);
      if (exp_value < ref_value) {
        points_.col(k) = exp_point;
        values_[k] = exp_value;
      } else {
        points_.col(k) = ref_point;
        values_[k] = ref_value;
      }
      
    } else {
      
      // Contract point.
      Eigen::Matrix<Type, Size, 1> 
          con_point = centroid + rho * (centroid - points_.col(k));
      Type con_value = function(con_point);
      if (con_value < values_[k]) {
        points_.col(k) = con_point;
        values_[k] = con_value;
      } else {
        
        Eigen::Matrix<Type, Size, 1> x = points_.col(i);
        for (std::size_t n = 0; n < dimension; ++n) {
          points_.col(n) = x + sigma * (points_.col(n) - x);
          values_[n] = function(points_.col(n));
        }
        
      }
      
    }
    
  }
  
protected:
  virtual initialize() {
    points_.resize(dimension_, dimension_ + 1);
    values_.resize(dimension_ + 1);
  }
  
private:
  
  static constexpr bool dynamic = std::is_same<Size, Eigen::Dynamic>::value;
  
  std::conditional<dynamic, 
      Eigen::Matrix<Type, Size, Size>, 
      Eigen::Matrix<Type, Size, Size + 1>>::type points_;
  
  std::conditional<dynamic, 
      Eigen::Matrix<Type, Size, 1>, 
      Eigen::Matrix<Type, Size + 1, 1>>::type values_;
  
}; // NelderMeadMethod

} // namespace numerical_methods

#endif // NUMERICAL_METHODS_DIRECT_SEARCH_NELDER_MEAD_METHOD_H_
