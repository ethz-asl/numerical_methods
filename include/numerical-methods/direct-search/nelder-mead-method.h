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
// J. A. Nelder and R. Mead, "A Simplex Method for Function Minimization," in 
// The Computer Journal, vol. 7, pp. 308–313 (1965).
// 
// R. O'Neill, "Statistical Algorithms - Algorithm AS 47: Function Minimization 
// using a Simplex Procedure," in Applied Statistics, vol. 20, no. 3, 
// pp. 338-345 (1971).
// 
// J. M. Chambers and J. E. Ertel, "Statistical Algorithms - Remark AS R11: A 
// Remark on Algorithm AS 47: Function Minimization using a Simplex Procedure," 
// in Applied Statistics, vol. 23, no. 2, pp. 250-251 (1974).
// 
// P. R. Benyon, "Statistical Algorithms - Remark AS R15: Function Minimization 
// using a Simplex Procedure," in Applied Statistics, vol. 25, no. 1, p. 97 
// (1976).
// 
// I. D. Hill, "Remark AS R28: A Remark on Algorithm AS 47: Function 
// Minimization using a Simplex Procedure," in Journal of the Royal Statistical 
// Society, Series C (Applied Statistics), vol. 27, no. 3, pp. 380-382 (1978).
template <typename Type, typename Size>
class NelderMeadMethod : public DirectSearchMethod {
public:
  
  template <class Function>
  Eigen::Matrix<Type, Size, 1> minimize(const Function& function, 
      const Eigen::Matrix<Type, Size, 1>& point) const {
    
    const int dimension = this->getDimension();
    
    // Initialize simplex.
    const Type extent = this->options.getInitExtent();
    points_.fill(point - extent / static_cast<Type>(dimension + 1));
    values_(0) = function(points_.col(0));
    for (std::size_t n = 1; n <= dimension; ++n) {
        points_(n, n) += extent;
        values_(n) = function(points_.col(n));
    }
    
    std::vector<std::size_t> ind(dimension + 1);
    for (std::size_t n = 0; n <= dimension; ++n) {
      ind[n] = n;
    }
    
    Eigen::Matrix<Type, Size, 1> centroid;
    centroid.resize(dimension);
    
    bool is_converged = false;
    for (int iter = 0; iter < this->options.getMaxIterations(); ++iter) {
      
      // Sort points in simplex according to value.
      std::sort(ind.begin(), ind.end(), 
          [&values_](std::size_t i, std::size_t j) {
            return values_(i) < values_(j);
          });
      
      // Compute centroid.
      centroid.setZero();
      for (std::size_t n = 0; n <= dimension; ++n) {
        centroid += points_.col(n);
      }
      centroid /= static_cast<Type>(dimension);
      
      if (iter > this->options.getMinIterations()) {
        
        // Compute maximum distance to centroid.
        Type dist = 0.0;
        for (std::size_t n = 0; n <= dimension; ++n) {
          dist = std::max(dist, (centroid - points_.col(n)).norm());
        }
        
        // Check convergence.
        is_converged = dist <= std::max(this->options.getAbsTolerance(), 
            this-options.getRelTolerance() * centroid.norm());
        if (is_converged) {
          break;
        }
        
      }
      
      // Store indices to best, second-worst and worst points.
      const std::size_t i = ind[0];
      const std::size_t j = ind[dimension - 1];
      const std::size_t k = ind[dimension];
      
      // Reflect worst point.
      const Eigen::Matrix<Type, Size, 1> 
          reflection_point = centroid + alpha_ * (centroid - points_.col(k));
      const Type reflection_value = function(reflection_point);
      if ((reflection_value < values_(j)) && (reflection_value > values_(i))) {
        points_.col(k) = reflection_point;
        values_(k) = reflection_value;
      } else if (reflection_value < values_(i)) {
        
        // Expand worst point.
        const Eigen::Matrix<Type, Size, 1> 
            expansion_point = centroid + gamma_ * (centroid - points_.col(k));
        const Type expansion_value = function(expansion_point);
        if (expansion_value < reflection_value) {
          points_.col(k) = expansion_point;
          values_(k) = expansion_value;
        } else {
          points_.col(k) = reflection_point;
          values_(k) = reflection_value;
        }
        
      } else {
        
        // Contract worst point.
        const Eigen::Matrix<Type, Size, 1> 
            contraction_point = centroid - rho_ * (centroid - points_.col(k));
        const Type contraction_value = function(contraction_point);
        if (contraction_value < values_(k)) {
          points_.col(k) = contraction_point;
          values_(k) = contraction_value;
        } else {
          
          // Redistribute around best point.
          for (std::size_t n = 0; n <= dimension; ++n) {
            if (n != i) {
              points_.col(n) = points_.col(i) 
                  + sigma_ * (points_.col(n) - points_.col(i));
              values_(n) = function(points_.col(n));
            }
          }
          
        }
        
      }
      
    }
    
    LOG_IF(WARNING, !is_converged, 
        "Maximum number of iterations reached. Results may be inaccurate.")
    return centroid;
    
  }
  
  class Options : public DirectSearchMethod<Type, Size>::Options {
  public:
    Options() : min_iterations_(1), 
                max_iterations_(100), 
                abs_tolerance_(1.0e-6), 
                rel_tolerance_(1.0e-3), 
                init_extent_(1.0) {}
    inline Type getInitExtent() const {
      return init_extent_;
    }
    inline void setInitExtent(Type init_extent) {
      CHECK_GT(init_extent, 0.0) << "Initial extent must be positive.";
      init_extent_ = init_extent;
    }
  private:
    Type init_extent_;
  } options;
  
protected:
  virtual initialize() {
    points_.resize(dimension_, dimension_ + 1);
    values_.resize(dimension_ + 1);
  }
  
private:
  
  static const Type alpha_ = 1.0;
  static const Type gamma_ = 2.0;
  static const Type rho_ = 0.5;
  static const Type sigma_ = 0.5;
  
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
