#ifndef NUMERICAL_METHODS_INTEGRATION_INTEGRATION_METHOD_H_
#define NUMERICAL_METHODS_INTEGRATION_INTEGRATION_METHOD_H_

#include "numerical-methods/common-definitions.h"

namespace numerical_methods {

template <typename Type>
class IntegrationMethod {
public:
  
  typedef Type type;
  
  explicit IntegrationMethod(Type error) : error_(error) {
    CHECK_GT(error, Type(0.0)) << "Desired error must be positive.";
  }
  virtual ~IntegrationMethod() {}
  
  // Return desired error.
  inline Type getError() const {
    return error_;
  }
  
  // Integrate function over given interval.
  template <class Function>
  Type integrate(const Function& function, Type a, Type b) const {
    return getUndef<Type>();
  }
  
private:
  const Type error_;
  
}; // IntegrationMethod

} // namespace numerical_methods

#endif // NUMERICAL_METHODS_INTEGRATION_INTEGRATION_METHOD_H_
