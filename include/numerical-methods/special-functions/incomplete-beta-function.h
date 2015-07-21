#ifndef NUMERICAL_METHODS_SPECIAL_FUNCTIONS_INCOMPLETE_BETA_FUNCTION_H_
#define NUMERICAL_METHODS_SPECIAL_FUNCTIONS_INCOMPLETE_BETA_FUNCTION_H_

#include <cmath>

#include <glog/logging.h>

#include "numerical-methods/common-definitions.h"
#include "numerical-methods/special-functions/incomplete-beta-function.h"

namespace numerical_methods {

// This class implements a method for numerically evaluating the incomplete 
// beta function, defined as
// 
//                  1     /x  a - 1        b - 1    
// I(x; a, b) = -------- |   t      (1 - t)      dx
//              B(a, b) /0
// 
// where B(a, b) is the beta function. Definitions of I(x; a, b) differ, with 
// some authors referring to it as the regularized incomplete beta function, 
// since it is scaled by B(a, b).
// 
// The implementation is based on the description provided by Majumder and 
// Bhattacharjee (1973), which is based on the successive reduction method 
// proposed by Soper (1921). Huynh Ngoc (1978) provides a comparison between 
// this method and continued fraction expansion.
// 
// H. E. Soper, "The Numerical Evaluation of the Incomplete Beta Function," in 
// Tracts for Computers, no. 7, Cambridge University Press (1921).
// 
// K. L. Majumder and G. P. Bhattacharjee, "Algorithm AS 63: The Incomplete 
// Beta Integral," in Journal of the Royal Statistical Society, Series C: 
// Applied Statistics, vol. 22, no. 3, pp. 409-411 (1973).
// 
// P. Huynh Ngoc, "A Note on the Computation of the Incomplete Beta Function," 
// in Advances in Engineering Software, vol. 12, no. 1, pp. 93-44 (1978).
template <typename Type>
class IncompleteBetaFunction : public SpecialFunction<Type> {
public:
  
  IncompleteBetaFunction(Type a, Type b, Type accuracy) : 
      parameters_(std::make_pair(a, b)), accuracy_(accuracy), 
      SpecialFunction<Type>(accuracy) {
    CHECK_GT(std::min(a, b), Type(0.0)) << "Parameters must be positive";
    initialize();
  }
  IncompleteBetaFunction(Type a, Type b) : 
      parameters_(std::make_pair(a, b)), accuracy_(getEps()), 
      SpecialFunction<Type>(getEps()) {
    CHECK_GT(std::min(a, b), Type(0.0)) << "Parameters must be positive";
    initialize();
  }
  
  
  inline const std::pair<Type, Type>& getParameters() const {
    return parameters_;
  }
  
  inline void setParameters(Type a, Type b) {
    CHECK_GT(std::min(a, b), Type(0.0)) << "Parameters must be positive";
    initialize();
  }
  
  // Evaluate the incomplete beta function.
  Type evaluate(Type argument) const {
    
    if ((argument == 0.0) || (argument == 1.0)) {
      return argument;
    }
     
    double x;
    double a;
    double b;
    
    // Switch to tail if needed.
    double k = parameters_.first + parameters_.second;
    double y = 1.0 - argument;
    bool tail = parameters_.first < k * argument;
    if (tail) {
      x = y;
      y = argument;
      a = parameters_.second;
      b = parameters_.first;
    } else {
      x = argument;
      a = parameters_.first;
      b = parameters_.second;
    }
    
    double term = 1.0;
    int m = 1;
    double value = 1.0;
    int n = static_cast<int>(b + k * y);
    
    // Apply reductions.
    double r = x / y;
    double t = b - static_cast<double>(m);
    if (n == 0) {
      r = x;
    }
    while (true) {
      term *= t * r / (a + static_cast<double>(m));
      value += term;
      t = std::abs(term);
      if ((t <= accuracy_) && (t <= accuracy_ * value)) {
        value *= std::exp(a * std::log(x) + (b - 1.0) * std::log(y) 
            - constant_) / a;
        if (tail) {
          value = 1.0 - value;
        }
        break;
      }
      ++m;
      --n;
      if (n >= 0) {
        t = b - static_cast<double>(m);
        if (n == 0) {
          r = x;
        }
      } else {
        t = k;
        k += 1.0;
      }
    }
    
    return value;
    
  }
  
private:
  
  std::pair<Type, Type> parameters_;
  Type accuracy_;
  Type constant_;
  
  void initialize() {
    constant_ = std::lgamma(parameters_.first) 
        + std::lgamma(parameters_.second) 
        - std::lgamma(parameters_.first + parameters_.second);
  }
  
}; // TanhSinhMethod

} // namespace numerical_methods

#endif // NUMERICAL_METHODS_SPECIAL_FUNCTIONS_INCOMPLETE_BETA_FUNCTION_H_
