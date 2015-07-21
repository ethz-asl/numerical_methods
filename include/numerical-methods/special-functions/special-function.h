#ifndef NUMERICAL_METHODS_SPECIAL_FUNCTIONS_SPECIAL_FUNCTION_H_
#define NUMERICAL_METHODS_SPECIAL_FUNCTIONS_SPECIAL_FUNCTION_H_

#include "numerical-methods/common-definitions.h"

namespace numerical_methods {

// This class is a base class for all special functions.
template <typename Type>
class SpecialFunction {
public:
  
  typedef Type type;
  
  explicit SpecialFunction(Type accuracy) : accuracy_(accuracy) {
    CHECK_GT(accuracy, Type(0.0)) << "Desired accuracy must be positive.";
  }
  virtual ~SpecialFunction() {}
  
  inline Type getAccuracy() const {
    return accuracy_;
  }
  
  // Evaluate function at given parameter.
  Type evaluate(Type argument) const {
    return getUndef<Type>();
  }
  
private:
  const Type argument_;
  
}; // SpecialFunction

} // namespace numerical_methods

#endif // NUMERICAL_METHODS_SPECIAL_FUNCTIONS_SPECIAL_FUNCTION_H_
