#ifndef NUMERICAL_METHODS_UTILITY_INCREMENTAL_STATISTICS_H_
#define NUMERICAL_METHODS_UTILITY_INCREMENTAL_STATISTICS_H_

#include <type_traits>

#include <Eigen/Dense>
#include <glog/logging.h>

#include "numerical-methods/common-definitions.h"

namespace numerical_methods {

template <typename Type, int Size>
class IncrementalStatistics {
public:
  
  typedef Type type;
  static constexpr int size = Size;
  
  template <bool Static = Size != Eigen::Dynamic>
  IncrementalStatistics(
      typename std::enable_if<Static>::type* = nullptr) : 
      dimension_(Size), 
      discount_(0.0), 
      weight_(0.0), 
      mean_(Size), 
      covariance_(Size, Size) {}
  template <bool Static = Size != Eigen::Dynamic>
  IncrementalStatistics(Type discount, 
      typename std::enable_if<Static>::type* = nullptr) : 
      dimension_(Size), 
      discount_(discount), 
      weight_(0.0), 
      mean_(Size), 
      covariance_(Size, Size) {
    CHECK_GE(discount, 0.0) << "Discount must be non-negative.";
    CHECK_LE(discount, 1.0) << "Discount must be less than or equal to one.";
  }
  template <bool Dynamic = Size == Eigen::Dynamic>
  explicit IncrementalStatistics(int dimension, 
      typename std::enable_if<Dynamic>::type* = nullptr) : 
      dimension_(dimension), 
      discount_(0.0), 
      weight_(0.0), 
      mean_(dimension), 
      covariance_(dimension, dimension) {
    CHECK_GT(dimension, 0) << "Dimension must be positive.";
  }
  template <bool Dynamic = Size == Eigen::Dynamic>
  explicit IncrementalStatistics(int dimension, Type discount, 
      typename std::enable_if<Dynamic>::type* = nullptr) : 
      dimension_(dimension), 
      discount_(discount), 
      weight_(0.0), 
      mean_(dimension), 
      covariance_(dimension, dimension) {
    CHECK_GT(dimension, 0) << "Dimension must be positive.";
    CHECK_GE(discount, 0.0) << "Discount must be non-negative.";
    CHECK_LE(discount, 1.0) << "Discount must be less than or equal to one.";
  }
  virtual ~IncrementalStatistics() {}
  
  // Return dimension or discount.
  inline int getDimension() const {
    return dimension_;
  }
  inline Type getDiscount() const {
    return discount_;
  }
  
  inline const Eigen::Matrix<Type, Size, 1>& getMean() const {
    return mean_;
  }
  inline const Eigen::Matrix<Type, Size, Size>& getCovariance() const {
    return covariance_;
  }
  
  inline void setMean(const Eigen::Matrix<Type, Size, 1>& mean) {
    mean_ = mean;
  }
  inline void setCovariance(const Eigen::Matrix<Type, Size, Size>& covariance) {
    covariance_ = covariance;
  }
  
  // Reset statistics.
  void clear() {
    weight_ = 0.0;
    mean_.setZero();
    covariance_.setZero();
  }
  
  void update(const Eigen::Matrix<Type, Size, 1>& point, Type weight) {
    
    if (discount_ == 1.0) {
      return;
    }
    
    weight_ = (1.0 - discount_) * weight_ + weight;
    weight /= weight_;
    
    const Eigen::Matrix<Type, Size, 1> residual = point - mean_;
    
    // Update sufficient statistics.
    mean_ += weight * residual;
    covariance_ = (1.0 - weight) * (covariance_ + 
        weight * (residual * residual.transpose()));
    
  }
  
  void update(const Eigen::Matrix<Type, Size, 1>& point) {
    update(point, 1.0);
  }
  
private:
  
  const int dimension_;
  const Type discount_;
  
  Type weight_;
  
  // Statistics.
  Eigen::Matrix<Type, Size, 1> mean_;
  Eigen::Matrix<Type, Size, Size> covariance_;
  
}; // IncrementalStatistics

} // namespace numerical_methods

#endif // NUMERICAL_METHODS_UTILITY_INCREMENTAL_STATISTICS_H_