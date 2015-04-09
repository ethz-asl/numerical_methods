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
  
  explicit TanhSinhMethod(Type error) : 
      IntegrationMethod<Type>(error), order_(12) {
    initNodes();
  }
  
  TanhSinhMethod(Type error, int order) : 
      IntegrationMethod<Type>(error), order_(order) {
    CHECK_GT(order, 0) << "Order must be positive.";
  	initNodes();
  }
  
  // Return order.
  inline int getOrder() const {
    return order_;
  }
  
  // Integrate function over given interval.
  template <class Function>
  Type integrate(const Function& function, Type a, Type b) const {
    
    const Type c = (b + a) / 2.0;
    const Type d = (b - a) / 2.0;
    
    std::vector<Type> values;
    
    Type sum = 0.0;
    Type step = 1.0;
    
    for (int k = 0; k < order_; ++k) {
      
      // Update sum.
      const int stride = std::pow(2, order_ - k - 1);
      for (int i = 0; i < nodes_.size(); i += stride) {
        if ((k == 0) || (i % (2 * stride) != 0)) {
          const Node& node = nodes_[i];
          if (node.point == 0.0) {
            sum += node.weight * function(c) * d;
          } else {
            sum += node.weight * function(c - d * node.point) * d 
                + node.weight * function(c + d * node.point) * d;
          }
        }
      }
      
      step /= 2.0;
      values.push_back(step * sum);
      
      // Estimate error and break if desired error achieved.
      if (estimError(values) <= this->getError()) {
        return *values.end();
      }
      
    }
    
    LOG(WARNING) << "Maximum order reached. Result may be inaccurate.";
    return *values.end();
    
  }
  
private:
  
  const int order_;
  
  struct Node {
    Type point;
    Type weight;
    Node(int k, Type h) {
      const Type t = k * h;
      const Type u = (getPi<Type>() * std::sinh(t)) / 2.0;
      const Type v = std::cosh(u);
      point = std::sinh(u) / v;
      weight = (getPi<Type>() / 2.0) * std::cosh(t) / (v * v);
    }
  };
  
  std::vector<Node> nodes_;
  
  // Create integration rule nodes.
  void initNodes() {
    int n = 20 * std::pow(2, order_) + 1;
    const Type h = 1.0 / std::pow(2, order_);
    for (int k = 0; k < n; ++k) {
      Node node(k, h);
      nodes_.push_back(node);
      if (std::abs(node.point - 1.0) <= std::pow(this->getError(), 2)) {
        break;
      }
    }
  }
  
  // Estimate error from previous values.
  Type estimError(const std::vector<Type>& values) const {
  	const int i = values.size() - 1;
  	if (i <= 0) {
  	  return 1.0;
  	}
  	const Type a = std::log10(std::abs(values[i + 1] - values[i]));
  	const Type b = std::log10(std::abs(values[i + 1] - values[i - 1]));
  	if (isUndef<Type>(a) || isUndef<Type>(b)) {
  	  return this->getError();
  	}
  	Type digits = std::max<Type>(a * a / b, 2.0 * a);
  	digits = std::min<Type>(std::max<Type>(digits, 
            std::log10(getEps<Type>())), 0.0);
  	return std::pow(0.1, - static_cast<int>(std::round(digits)));
  }
  
}; // TanhSinhMethod

} // namespace numerical_methods

#endif // NUMERICAL_METHODS_INTEGRATION_TANH_SINH_METHOD_H_
