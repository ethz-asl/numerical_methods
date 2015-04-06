#ifndef NUMERICAL_METHODS_INTEGRATION_INTEGRATION_METHOD_H_
#define NUMERICAL_METHODS_INTEGRATION_INTEGRATION_METHOD_H_

#include <glog/logging.h>

#include "numerical-methods/common-definitions.h"

namespace numerical_methods {

class IntegrationMethod {
public:
  
  // Integrate a function over a given interval.
  template <class Function>
  double integrate(const Function& fun, double a, double b) const {
  	return getNaN();
  }
  
private:
  
  // Disallow dangerous copy and assignment constructors.
  IntegrationMethod(const IntegrationMethod&) = delete;
  IntegrationMethod& operator=(const IntegrationMethod&) = delete;
    
}; // IntegrationMethod

} // namespace numerical_methods

#endif // NUMERICAL_METHODS_INTEGRATION_INTEGRATION_METHOD_H_