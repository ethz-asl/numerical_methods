#ifndef NUMERICAL_METHODS_INTEGRATION_TANH_SINH_METHOD_H_
#define NUMERICAL_METHODS_INTEGRATION_TANH_SINH_METHOD_H_

#include <cmath>

#include <glog/logging.h>

#include "numerical-methods/common-definitions.h"
#include "numerical-methods/integration/integration-method.h"

namespace numerical_methods {

// This class implements the tanh-sinh integration method, as described in 
// Bailey et al. (2005), simplified for fixed-precision arithmetic.
// 
// D. H. Bailey, K. Jeyabalan, and X. S. Li, "A Comparison of Three High-
// precision Quadrature Schemes," in Experimental Mathematics, vol. 14, no. 3, 
// pp. 317-329 (2005).
template <typename Type>
class TanhSinhMethod : public IntegrationMethod<Type> {
public:
  
  explicit TanhSinhMethod(Type error) : 
      IntegrationMethod<Type>(error), order_(10) {
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
    
    CHECK_LT(a, b) << "Interval must be non-empty.";
    
    const Type c = (b + a) / 2.0;
    const Type d = (b - a) / 2.0;
    
    std::vector<Type> values;
    values.reserve(order_);
    
    Type step = 1.0;
    Type sum = 0.0;
    for (int k = 0; k < order_; ++k) {
      
      // Update sum.
      const int stride = std::pow(2, order_ - k - 1);
      for (int i = 0; i < nodes_.size(); i += stride) {
        if ((k == 0) || (i % (2 * stride) != 0)) {
          const Node& node = nodes_[i];
          const Type weight = node.weight * d;
          if (node.point == 0.0) {
            sum += weight * function(c);
          } else {
            sum += weight * function(c - d * node.point);
            sum += weight * function(c + d * node.point);
          }
        }
      }
      
      step /= 2.0;
      values.push_back(step * sum);
      
      // Estimate error and break if desired error achieved.
      if (estimError(values) <= this->getError()) {
        return values.back();
      }
      
    }
    
    return values.back();
    
  }
  
private:
  
  const int order_;
  
  struct Node {
    Type point;
    Type weight;
    Node(int k, Type h) {
      const Type t = k * h;
      const Type u = (getPi<Type>() / 2.0) * std::sinh(t);
      const Type v = std::cosh(u);
      point = std::sinh(u) / v;
      weight = (getPi<Type>() / 2.0) * std::cosh(t) / std::pow(v, 2);
    }
  };
  
  std::vector<Node> nodes_;
  
  // Create integration rule nodes.
  void initNodes() {
    int n = 20 * std::pow(2, order_) + 1;
    const Type h = 1.0 / std::pow(2, order_);
    const Type t = std::pow(this->getError(), 2);
    for (int k = 0; k < n; ++k) {
      const Node node(k, h);
      CHECK(!(isUndef(node.point) || isUndef(node.weight)));
      nodes_.push_back(node);
      if (std::abs(node.point - 1.0) <= t) {
        break;
      }
    }
  }
  
  // Estimate error from previous values.
  Type estimError(const std::vector<Type>& values) const {
    int i = values.size() - 1;
    if (i <= 0) {
      return 1.0;
    }
    --i;
    const Type a = std::log10(std::abs(values[i + 1] - values[i]));
    const Type b = std::log10(std::abs(values[i + 1] - values[i - 1]));
    if (isUndef<Type>(a) || isUndef<Type>(b)) {
      return this->getError();
    }
    Type precision = std::max<Type>(std::pow(a, 2) / b, 2.0 * a);
    precision = std::min<Type>(std::max<Type>(precision, 
        std::log10(getEps<Type>())), 0.0);
    return std::pow(Type(0.1), - static_cast<int>(std::round(precision)));
  }
  
}; // TanhSinhMethod

} // namespace numerical_methods

#endif // NUMERICAL_METHODS_INTEGRATION_TANH_SINH_METHOD_H_
