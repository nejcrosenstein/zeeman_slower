#ifndef HISTOGRAM_HPP
#define HISTOGRAM_HPP

#include <vector>
#include <optional>

struct Histogram
{
  Histogram(){}

  Histogram(double vel_lowest, double vel_binwidth, int vel_nbins,
    double pos_lowest, double pos_binwidth, int pos_nbins) :
    bins_vel_({ vel_lowest, vel_binwidth, vel_nbins }),
    bins_pos_({ pos_lowest, pos_binwidth, pos_nbins }),
    histogram_(std::vector<int>(vel_nbins*pos_nbins, 0))
  {

  }

  void addSample(double sample_pos, double sample_vel)
  {
    auto pos_and_vel_idx = std::make_pair(
      bins_pos_.binIndex(sample_pos),
      bins_vel_.binIndex(sample_vel));

    if (!current_pos_and_vel_bin_ || 
        *current_pos_and_vel_bin_ != pos_and_vel_idx)
    {
      current_pos_and_vel_bin_ = pos_and_vel_idx;

      int pos_idx = pos_and_vel_idx.first;
      int vel_idx = pos_and_vel_idx.second;

      if (pos_idx >= 0 && pos_idx < bins_pos_.number_of_bins_ &&
          vel_idx >= 0 && vel_idx < bins_vel_.number_of_bins_)
      {
        histogram_[vel_idx*bins_pos_.number_of_bins_ + pos_idx] += 1;
      }
    }
  }

  struct Bins
  {
    double lowest_start_;
    double bin_width_;
    int number_of_bins_;

    // can be out of bounds
    int binIndex(double value)
    {
      return int((value - lowest_start_) / bin_width_);
    }
  };

  Bins bins_vel_;
  Bins bins_pos_;
  std::optional<std::pair<int, int>> current_pos_and_vel_bin_;
  std::optional<int> current_vel_bin_;
  std::vector<int> histogram_;
};

#endif // HISTOGRAM_HPP