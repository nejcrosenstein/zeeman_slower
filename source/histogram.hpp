#ifndef HISTOGRAM_HPP
#define HISTOGRAM_HPP

#include <vector>
#include <optional>

struct Histogram
{

  Histogram(double vel_lowest, double vel_binwidth, int vel_nbins,
    double pos_lowest, double pos_binwidth, int pos_nbins) :
    bins_vel_({ vel_lowest, vel_binwidth, vel_nbins }),
    bins_pos_({ pos_lowest, pos_binwidth, pos_nbins }),
    histogram_(std::vector<int>(vel_nbins*pos_nbins, 0))
  {

  }

  void addSample(double sample_pos, double sample_vel)
  {
    auto pos_idx = bins_pos_.binIndex(sample_pos);

    if (pos_idx != current_pos_bin_)
    {
      current_pos_bin_ = pos_idx;

      auto vel_idx = bins_vel_.binIndex(sample_vel);

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
  int current_pos_bin_ = -1;
  std::vector<int> histogram_;
};

#endif // HISTOGRAM_HPP