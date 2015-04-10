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
  
  explicit NelderMeadMethod(int dimension) : 
        DirectSearchMethod<Type, Size>(dimension) {
    points_.resize(dimension, dimension + 1);
  }
  explicit NelderMeadMethod() : DirectSearchMethod<Type, Size>() {}
  
  template <class Function>
  void findMinimum(const Function& function, 
      const Eigen::Matrix<Type, Size, 1>& point) const {
    
  }
  
private:
  
  typedef typename std::conditional<dynamic, 
      std::unordered_map<Eigen::Matrix<Type, Size, Size>, Type>, 
      std::unordered_map<Eigen::Matrix<Type, Size, Size + 1>, Type>>::type 
          values_;
  
}; // NelderMeadMethod

} // namespace numerical_methods

#endif // NUMERICAL_METHODS_DIRECT_SEARCH_NELDER_MEAD_METHOD_H_
