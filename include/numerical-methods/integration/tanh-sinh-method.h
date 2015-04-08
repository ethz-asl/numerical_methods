#ifndef NUMERICAL_METHODS_INTEGRATION_TANH_SINH_METHOD_H_
#define NUMERICAL_METHODS_INTEGRATION_TANH_SINH_METHOD_H_

#include <cmath>

#include <glog/logging.h>

#include "numerical-methods/common-definitions.h"
#include "numerical-methods/integration/integration-method.h"

namespace numerical_methods {

template <typename Type>
class TanhSinhMethod : public IntegrationMethod<Type> {
public:
  
  TanhSinhMethod(Type error, std::size_t levels) : 
      error_(error), levels_(levels) {
  	CHECK_GT(error, 0.0) << "Desired error must be positive.";
  	initNodes();
  }
  
  // Integrate a function over a given interval.
  template <class Function>
  Type integrate(const Function& fun, Type a, Type b) const {
  	
  	const Type c = (b + a) / 2.0;
  	const Type d = (b - a) / 2.0;
  	
  	std::vector<Type> values;
  	
  	Type sum = 0.0;
  	Type step = 1.0;
  	for (std::size_t k = 0; k < levels_; ++k) {
  	  
  	  const std::size_t stride = std::pow(2, levels_ - k - 1);
  	  for (std::size_t i = 0; i < nodes_.size(); i += stride) {
  	    if ((k == 0) || (i % (2 * stride) != 0)) {
  	      const Node& node = nodes_[i];
  	      if (node.point == 0.0) {
  	        sum += node.weight * fun(c) * d;
  	      } else {
  	      	sum += node.weight * fun(c - d * node.point) * d 
  	      	    + node.weight * fun(c + d * node.point) * d;
  	      }
  	    }
  	  }
  	  step /= 2.0;
  	  values.push_back(step * sum);
  	  
  	  if (evalError(values) <= error_) {
  	  	return *values.end();
  	  }
  	  
  	}
  	
  	return *values.end();
  	
  }
  
private:
  
  Type error_;
  std::size_t levels_;
  
  struct Node {
    Type point;
    Type weight;
    Node(std::size_t k, Type h) {
      const Type t = k * h;
      const Type u = (getPi<Type>() * std::sinh(t)) / 2.0;
      const Type v = std::cosh(u);
      point = std::sinh(u) / v;
      weight = (getPi<Type>() / 2.0) * std::cosh(t) / (v * v);
    }
  };
  
  std::vector<Node> nodes_;
  
  void initNodes() {
    std::size_t n = 20 * std::pow(2, levels_) + 1;
    const Type h = 1.0 / std::pow(2, levels_);
    for (std::size_t k = 0; k < n; ++k) {
      Node node(k, h);
      nodes_.push_back(node);
      if (std::abs(node.point - 1.0) <= error_ * error_) {
        break;
      }
    }
  }
  
  Type evalError(const std::vector<Type>& values) const {
  	const std::size_t i = values.size() - 1;
  	if (i <= 0) {
  	  return 1.0;
  	}
  	const Type a = std::log10(std::abs(values[i + 1] - values[i]));
  	const Type b = std::log10(std::abs(values[i + 1] - values[i - 1]));
  	if (isUndef<Type>(a) || isUndef<Type>(b)) {
  	  return error_;
  	}
  	Type digits = std::max<Type>(a * a / b, 2.0 * a);
  	digits = std::min<Type>(std::max<Type>(digits, 
            std::log10(getEps<Type>())), 0.0);
  	return std::pow(0.1, - static_cast<int>(std::round(digits)));
  }
  
}; // TanhSinhMethod

} // namespace numerical_methods

#endif // NUMERICAL_METHODS_INTEGRATION_TANH_SINH_METHOD_H_
