#ifndef NUMERICAL_METHODS_INTEGRATION_INTEGRATION_METHOD_H_
#define NUMERICAL_METHODS_INTEGRATION_INTEGRATION_METHOD_H_

#include "numerical-methods/common-definitions.h"

namespace numerical_methods {

template <typename Type>
class IntegrationMethod {
public:
  
  // Integrate a function over a given interval.
  template <class Function>
  Type integrate(const Function& fun, Type a, Type b) const {
    return getUndef<Type>();
  }
  
private:
  
  // Disallow dangerous copy and assignment constructors.
  IntegrationMethod(const IntegrationMethod&) = delete;
  IntegrationMethod& operator=(const IntegrationMethod&) = delete;
    
}; // IntegrationMethod

} // namespace numerical_methods

#endif // NUMERICAL_METHODS_INTEGRATION_INTEGRATION_METHOD_H_
