#ifndef NUMERICAL_METHODS_INTEGRATION_TANH_SINH_METHOD_H_
#define NUMERICAL_METHODS_INTEGRATION_TANH_SINH_METHOD_H_

#include <cmath>

#include <glog/logging.h>

#include "numerical-methods/common-definitions.h"
#include "numerical-methods/integration/integration-method.h"

namespace numerical_methods {

class TanhSinhMethod : public IntegrationMethod {
public:
  
  TanhSinhMethod(double precision, std::size_t levels) : 
      precision_(precision), levels_(levels) {
  	CHECK_GT(precision, 0.0);
  	CHECK_LE(levels, 15);
  	initialize();
  }
  
  // Integrate a function over a given interval.
  template <class Function>
  double integrate(const Function& fun, double a, double b) const {
  	
  	const double c = (b + a) / 2.0;
  	const double d = (b - a) / 2.0;
  	
  	std::vector<double> values;
  	
  	double sum = 0.0;
  	double step = 1.0;
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
  	  
  	  const double error = evalError(values);
  	  if error <= precision_ {
  	  	return *values.end();
  	  }
  	  
  	}
  	
  	return *values.end();
  	
  }
  
private:
  
  double precision_;
  int levels_;
  
  struct Node {
  	double point;
  	double weight;
  	Node(std::size_t k, double h) {
  	  const double t = k * h;
  	  const double u = (std::pi() * std::sinh(t)) / 2.0;
  	  const double v = std::cosh(u);
  	  point = std::sinh(u) / v;
  	  weight = (std::pi() / 2.0) * std::cosh(t) / (v * v);
    }
  };
  
  std::vector<Node> nodes_;
  
  void initialize() {
    std::size_t n = 20 * std::pow(2, levels_) + 1;
    const double h = 1.0 / std::pow(2, levels_);
    for (std::size_t k = 0; k < n; ++k) {
  	  nodes_.push_back(Node(k, h));
  	  if std::abs(node.point - 1.0) <= precision_ * precision_ {
  	    break;
  	  }
    }
  }
  
  double evalError(const std::vector<double>& values) const {
  	const std::size_t i = values.size() - 1;
  	if (i <= 0) {
  	  return 1.0;
  	}
  	const double a = std::log10(std::abs(values[i + 1] - values[i]));
  	const double b = std::log10(std::abs(values[i + 1] - values[i - 1]));
  	if (isNaN(a) || isNaN(b)) {
  	  return precision_;
  	}
  	double digits = std::max(a * a / b, 2.0 * a);
  	digits = std::min(std::max(digits, std::log10(getEps())), 0.0);
  	return std::pow(0.1, - static_cast<int>(std::round(digits)));
  }
  
}; // TanhSinhMethod

} // namespace numerical_methods

#endif // NUMERICAL_METHODS_INTEGRATION_TANH_SINH_METHOD_H_