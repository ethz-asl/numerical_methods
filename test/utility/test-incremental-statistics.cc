#include <cmath>
#include <random>
#include <utility>
#include <vector>

#include <Eigen/Dense>
#include <gtest/gtest.h>

#include "numerical-methods/common-definitions.h"
#include "numerical-methods/utility/incremental-statistics.h"

namespace numerical_methods {

// Sequence for testing incremental statistics.
template <typename Type, int Size>
class TestSequence {
public:
  explicit TestSequence(int dimension, int length) : dimension_(dimension), 
      length_(length), mean_(dimension), covariance_(dimension, dimension), 
      valid_(false) {
    if (Size != Eigen::Dynamic) {
      CHECK_EQ(dimension, Size) << "Dimension must be consistent.";
    } else {
      CHECK_GT(dimension, 0) << "Dimension must be positive.";
    }
    CHECK_GT(length, 0) << "Length must be positive.";
    generate();
  }
  inline int getDimension() const {
    return dimension_;
  }
  inline Type getWeight(std::size_t i) const {
    CHECK_LT(i, length_);
    return sequence_[i].first;
  }
  inline const Eigen::Matrix<Type, Size, 1>& getPoint(std::size_t i) const {
    CHECK_LT(i, length_);
    return sequence_[i].second;
  }
  inline const Eigen::Matrix<Type, Size, 1>& getMean() const {
    CHECK(valid_);
    return mean_;
  }
  inline const Eigen::Matrix<Type, Size, Size>& getCovariance() const {
    CHECK(valid_);
    return covariance_;
  }
  inline bool isValid() const {
    return valid_;
  }
  void evalStatistics(Type discount) {
    
    CHECK_GE(discount, 0.0) << "Discount must be non-negative.";
    CHECK_LE(discount, 1.0) << "Discount must be less than or equal to one.";
    
    // Compute mean.
    mean_.setZero();
    Type sum = 0.0;
    for (int i = 0; i < sequence_.size(); ++i) {
      const Type weight = std::pow(1.0 - discount, sequence_.size() - 1 - i);
      mean_ += weight * sequence_[i].first * sequence_[i].second;
      sum += weight;
    }
    mean_ /= sum;
    
    // Compute covariance.
    covariance_.setZero();
    for (int i = 0; i < sequence_.size(); ++i) {
      const Type weight = std::pow(1.0 - discount, sequence_.size() - 1 - i);
      const Eigen::Matrix<Type, Size, 1> 
          residual = sequence_[i].second - mean_;
      covariance_ += weight * sequence_[i].first 
          * residual * residual.transpose();
    }
    covariance_ /= sum;
    
    discount_ = discount;
    valid_ = true;
    
  }
private:
  int dimension_;
  int length_;
  Eigen::Matrix<Type, Size, 1> mean_;
  Eigen::Matrix<Type, Size, Size> covariance_;
  bool valid_;
  Type discount_;
  std::vector<std::pair<Type, Eigen::Matrix<Type, Size, 1>>> sequence_;
  void generate() {
    
    std::uniform_real_distribution<Type> scale_distribution(1.0, 5.0);
    std::uniform_real_distribution<Type> skewness_distribution(- 2.0, 2.0);
    std::uniform_real_distribution<Type> weight_distribution(0.0, 1.0);
    std::normal_distribution<Type> noise_distribution(0.0, 1.0);
    
    std::mt19937 engine;
    std::random_device device;
    engine.seed(device());
    
    // Generate random scale and skewness parameters.
    const Type scale = scale_distribution(engine);
    const Type skewness = skewness_distribution(engine);
    
    // Generate sequence.
    sequence_.reserve(length_);
    Eigen::Matrix<Type, Size, 1> point(dimension_);
    for (int i = 0; i < length_; ++i) {
      const Type weight = weight_distribution(engine);
      for (int j = 0; j < dimension_; ++j) {
        Type noise = scale * noise_distribution(engine);
        if (scale * noise_distribution(engine) > skewness * noise) {
          noise *= - 1.0;
        }
        point(j) += noise;
      }
      sequence_.push_back(std::make_pair(weight, point));
    }
    
  }
};

template <class Statistics>
class IncrementalStatisticsTest : public testing::Test {
public:
  typedef typename Statistics::type type;
  static constexpr int size = Statistics::size;
protected:
  virtual void SetUp() {}
  static const std::vector<int> lengths;
  static const std::vector<typename Statistics::type> discounts;
  static constexpr typename Statistics::type tolerance = 1.0e-12;
};

template <class Statistics>
const std::vector<int> 
    IncrementalStatisticsTest<Statistics>::lengths = 
        {1, 2, 5, 10, 20, 50, 100, 200, 500, 1000};

template <class Statistics>
const std::vector<typename Statistics::type> 
    IncrementalStatisticsTest<Statistics>::discounts = 
        {0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0};

template <class Statistics>
constexpr typename Statistics::type 
    IncrementalStatisticsTest<Statistics>::tolerance;

template <class Statistics>
class StaticIncrementalStatisticsTest : 
    public IncrementalStatisticsTest<Statistics> {
  static_assert(Statistics::size != Eigen::Dynamic, "");
protected:
  static constexpr int dimension = Statistics::size;
};

typedef testing::Types<IncrementalStatistics<double, 1>, 
                       IncrementalStatistics<double, 2>, 
                       IncrementalStatistics<double, 5>, 
                       IncrementalStatistics<double, 10>> StaticTypes;

TYPED_TEST_CASE(StaticIncrementalStatisticsTest, StaticTypes);

// Check that batch and incremental statistics are the same.
TYPED_TEST(StaticIncrementalStatisticsTest, IsEqualToBatch) {
  const int dimension = this->dimension;
  for (int length : this->lengths) {
    TestSequence<typename TypeParam::type, TypeParam::size> 
        sequence(dimension, length);
    for (typename TypeParam::type discount : this->discounts) {
      sequence.evalStatistics(discount);
      IncrementalStatistics<typename TypeParam::type, TypeParam::size> 
          statistics(discount);
      ASSERT_EQ(statistics.getDimension(), dimension);
      ASSERT_EQ(statistics.getDiscount(), discount);
      for (int i = 0; i < length; ++i) {
        statistics.update(sequence.getPoint(i), sequence.getWeight(i));
      }
      EXPECT_LT((statistics.getMean() - sequence.getMean())
          .array().abs().maxCoeff(), this->tolerance);
      EXPECT_LT((statistics.getCovariance() - sequence.getCovariance())
          .array().abs().maxCoeff(), this->tolerance);
    }   
  }
}

template <class Statistics>
class DynamicIncrementalStatisticsTest : 
    public IncrementalStatisticsTest<Statistics> {
  static_assert(Statistics::size == Eigen::Dynamic, "");
protected:
  const std::vector<int> dimensions = {1, 2, 5, 10};
};

typedef testing::Types<IncrementalStatistics<double, Eigen::Dynamic>> 
    DynamicTypes;

TYPED_TEST_CASE(DynamicIncrementalStatisticsTest, DynamicTypes);

// Check that batch and incremental statistics are the same.
TYPED_TEST(DynamicIncrementalStatisticsTest, IsEqualToBatch) {
  for (int dimension : this->dimensions) {
    for (int length : this->lengths) {
      TestSequence<typename TypeParam::type, TypeParam::size> 
          sequence(dimension, length);
      for (typename TypeParam::type discount : this->discounts) {
        sequence.evalStatistics(discount);
        IncrementalStatistics<typename TypeParam::type, TypeParam::size> 
            statistics(dimension, discount);
        ASSERT_EQ(statistics.getDimension(), dimension);
        ASSERT_EQ(statistics.getDiscount(), discount);
        for (int i = 0; i < length; ++i) {
          statistics.update(sequence.getPoint(i), sequence.getWeight(i));
        }
        EXPECT_LT((statistics.getMean() - sequence.getMean())
            .array().abs().maxCoeff(), this->tolerance);
        EXPECT_LT((statistics.getCovariance() - sequence.getCovariance())
            .array().abs().maxCoeff(), this->tolerance);
      }
    }
  }
}

} // namespace numerical_methods

int main(int num_arguments, char** arguments) {
  testing::InitGoogleTest(&num_arguments, arguments);
  return RUN_ALL_TESTS();
}
