#ifndef NUMERICAL_METHODS_DIRECT_SEARCH_NELDER_MEAD_METHOD_H_
#define NUMERICAL_METHODS_DIRECT_SEARCH_NELDER_MEAD_METHOD_H_

#include <cmath>
#include <type_traits>

#include <Eigen/Dense>
#include <glog/logging.h>

#include "numerical-methods/common-definitions.h"
#include "numerical-methods/eigen-utility-functions.h"
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
template <typename Type, int Size>
class NelderMeadMethod : public DirectSearchMethod<Type, Size> {
public:
  
  template <bool Static = Size != Eigen::Dynamic>
  NelderMeadMethod(typename std::enable_if<Static>::type* = nullptr) : 
      DirectSearchMethod<Type, Size>() {}
  template <bool Dynamic = Size == Eigen::Dynamic>
  explicit NelderMeadMethod(int dimension, 
      typename std::enable_if<Dynamic>::type* = nullptr) : 
      DirectSearchMethod<Type, Size>(dimension) {}
  
  template <class Function>
  Eigen::Matrix<Type, Size, 1> minimize(const Function& function, 
      const Eigen::Matrix<Type, Size, 1>& point) const {
    
    const int dimension = this->getDimension();
    
    typename std::conditional<Size != Eigen::Dynamic, 
        Eigen::Matrix<Type, Size, Size + 1>, 
        Eigen::Matrix<Type, Size, Size>>::type 
            points(dimension, dimension + 1);
    
    typename std::conditional<Size != Eigen::Dynamic, 
        Eigen::Matrix<Type, Size + 1, 1>, 
        Eigen::Matrix<Type, Size, 1>>::type values(dimension + 1);
    
    // Initialize simplex.
    const Type scale = this->options.getInitScale();
    points.col(0) = point;
    values(0) = function(point);
    for (std::size_t n = 1; n <= dimension; ++n) {
      points.col(n) = point - scale / static_cast<Type>(dimension);
      points(n - 1, n) += scale;
      values(n) = function(points.col(n));
    }
    
    std::vector<std::size_t> ind(dimension + 1);
    for (std::size_t n = 0; n <= dimension; ++n) {
      ind[n] = n;
    }
    
    Eigen::Matrix<Type, Size, 1> centroid(dimension);
    
    bool is_converged = false;
    for (int iter = 0; iter < this->options.getMaxIterations(); ++iter) {
      
      // Sort points in simplex according to value.
      std::sort(ind.begin(), ind.end(), 
          [&values](std::size_t i, std::size_t j) {
            return values(i) < values(j);
          });
      
      // Store indices to best, second-worst and worst points.
      const std::size_t i = ind[0];
      const std::size_t j = ind[dimension - 1];
      const std::size_t k = ind[dimension];
      
      if (iter > this->options.getMinIterations()) {
        
        Type diff = 0.0;
        for (std::size_t n = 1; n <= dimension; ++n) {
          diff = std::max(diff, (points.col(ind[n]) - points.col(i)).norm());
        }
        
        // Check convergence.
        is_converged = diff <= std::max(this->options.getAbsTolerance(), 
          this->options.getRelTolerance() * points.col(i).norm());
        if (is_converged) {
          break;
        }
        
      }
      
      // Compute centroid.
      centroid.setZero();
      for (std::size_t n = 0; n <= dimension; ++n) {
        if (n != k) {
          centroid += points.col(n);
        }
      }
      centroid /= static_cast<Type>(dimension);
      
      // Reflect point.
      const Eigen::Matrix<Type, Size, 1> reflection_point = centroid 
          + reflection_coefficient_ * (centroid - points.col(k));
      const Type reflection_value = function(reflection_point);
      if ((reflection_value < values(j)) && (reflection_value >= values(i))) {
        points.col(k) = reflection_point;
        values(k) = reflection_value;
      } else if (reflection_value < values(i)) {
        
        // Expand point.
        const Eigen::Matrix<Type, Size, 1> expansion_point = centroid 
            + expansion_coefficient_ * (centroid - points.col(k));
        const Type expansion_value = function(expansion_point);
        if (expansion_value < reflection_value) {
          points.col(k) = expansion_point;
          values(k) = expansion_value;
        } else {
          points.col(k) = reflection_point;
          values(k) = reflection_value;
        }
        
      } else {
        
        // Contract point.
        const Eigen::Matrix<Type, Size, 1> contraction_point = centroid 
            + contraction_coefficient_ * (centroid - points.col(k));
        const Type contraction_value = function(contraction_point);
        if (contraction_value < values(k)) {
          points.col(k) = contraction_point;
          values(k) = contraction_value;
        } else {
          
          // Reduce points.
          for (std::size_t n = 0; n <= dimension; ++n) {
            if (n != i) {
              points.col(n) = points.col(i) 
                  + reduction_coefficient_ * (points.col(n) - points.col(i));
              values(n) = function(points.col(n));
            }
          }
          
        }
        
      }
      
    }
    
    LOG_IF(WARNING, !is_converged) 
        << "Maximum number of iterations reached. Results may be inaccurate.";
    return centroid;
    
  }
  
  class Options : public DirectSearchMethod<Type, Size>::Options {
  public:
    Options() : DirectSearchMethod<Type, Size>::Options(), init_scale_(1.0) {}
    inline Type getInitScale() const {
      return init_scale_;
    }
    inline void setInitScale(Type init_scale) {
      CHECK_GT(init_scale, 0.0) << "Initial scale must be positive.";
      init_scale_ = init_scale;
    }
  private:
    Type init_scale_;
  } options;
  
private:
  
  // Simplex coefficients.
  static constexpr Type reflection_coefficient_ = 1.0;
  static constexpr Type expansion_coefficient_= 2.0;
  static constexpr Type contraction_coefficient_ = - 0.5;
  static constexpr Type reduction_coefficient_ = 0.5;
  
}; // NelderMeadMethod

template <typename Type, int Size> 
constexpr Type NelderMeadMethod<Type, Size>::reflection_coefficient_;
template <typename Type, int Size> 
constexpr Type NelderMeadMethod<Type, Size>::expansion_coefficient_;
template <typename Type, int Size> 
constexpr Type NelderMeadMethod<Type, Size>::contraction_coefficient_;
template <typename Type, int Size> 
constexpr Type NelderMeadMethod<Type, Size>::reduction_coefficient_;

} // namespace numerical_methods

#endif // NUMERICAL_METHODS_DIRECT_SEARCH_NELDER_MEAD_METHOD_H_
