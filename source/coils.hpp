#ifndef COILS_HPP
#define COILS_HPP

#include <cmath>
#include <vector>
#include <algorithm>
#include <numeric>
#include <fstream>

#include "physics.hpp"
#include "mathtools.hpp"

//
// Slower properties
//

struct MagneticField1D
{
  double operator()(double posOnAxis) const
  {
    double itp_pos = (posOnAxis - start_z_) * step_z_inv_;
    int itp_pos_lo = (int)std::floor(itp_pos);
    int itp_pos_hi = itp_pos_lo + 1;

    double whi = itp_pos - double(itp_pos_lo);
    double wlo = 1.0 - whi;

#if 0
    if (whi > 1.0 || whi < 0.0 || wlo > 1.0 || wlo < 0.0)
    { 
      std::cerr << "Weights have unexpected sizes" << std::endl;
    }
#endif

    itp_pos_lo = index_clamp(itp_pos_lo, 0, (int)magnetic_fields_tesla_.size1_);
    itp_pos_hi = index_clamp(itp_pos_hi, 0, (int)magnetic_fields_tesla_.size1_);

    return wlo * magnetic_fields_tesla_(0, itp_pos_lo) + 
           whi * magnetic_fields_tesla_(0, itp_pos_hi);
  }

  double start_z_;
  double step_z_;
  double step_z_inv_;
  Arr2D magnetic_fields_tesla_;
};

struct ZeemanSlower : public MagneticField1D
{
  static constexpr double tube_radius_ = 0.02;
  static constexpr double wire_width_ = 0.0025;
  static constexpr double wire_thickness_ = 0.001;

  static constexpr double lengths_m_[9] =
  {
    0.63, 0.57, 0.5, 0.42, 0.36, 0.29, 0.22, 0.14, 0.06
  };

  static constexpr int winding_layers_[9] =
  {
    3, 6, 4, 2, 2, 2, 2, 2, 5
  };

  double solenoidField(double zpos, double inner_rad, double length, int num_layers, double current)
  {
    double total_field = 0.0;

    double start = zpos - 0.5*wire_width_;
    double end = zpos - (length + 0.5*wire_width_);

    for (int i = 0; i < num_layers; ++i)
    {
      double layer_radius = inner_rad + wire_thickness_ * (double(i) + 0.5);
      double part0 = start / sqrt(sq(start) + sq(layer_radius));
      double part1 = end / sqrt(sq(end) + sq(layer_radius));
      total_field += 0.5*(part0 - part1) * current * magnetic_constant / wire_width_;
    }

    return total_field;
  }

  ZeemanSlower(double current_bias_ampere, double current_profile_ampere, double start_z, double end_z, int num_points)
  {
    std::vector<double> currents(9);
    currents[0] = current_bias_ampere;
    for (int i = 1; i < 9; ++i)
      currents[i] = current_profile_ampere;

    magnetic_fields_tesla_.resize(1, num_points);

    this->start_z_ = start_z;
    this->step_z_ = (end_z - start_z) / double(num_points);
    this->step_z_inv_ = 1.0 / step_z_;

    for (int i = 0; i < num_points; ++i)
    {
      double z = start_z_ + step_z_ * double(i);

      double total_field = 0.0;
      double inner_rad = tube_radius_;

      for (int i = 0; i < 9; ++i)
      {
        double current = currents[i];
        double length = lengths_m_[i];
        double winds = winding_layers_[i];

        total_field += solenoidField(z, inner_rad, length, winds, current);

        inner_rad += winds * wire_thickness_;
      }

      magnetic_fields_tesla_(0, i) = total_field;
    }
  }
};


struct ImportedField : public MagneticField1D
{
  ImportedField(double distance_from_slower_entry, const char* file_name)
  {
    std::ifstream in(file_name);

    std::vector<double> ax_poss;
    std::vector<double> fields;
    while (!in.eof())
    {
      double z;
      double b;
      in >> z;
      in >> b;

      ax_poss.push_back(z + distance_from_slower_entry);
      fields.push_back(b);
    } 

    magnetic_fields_tesla_.resize(1, fields.size());
    for (int i = 0; i < fields.size(); ++i)
      magnetic_fields_tesla_(0, i) = fields[i];

    std::vector<double> diffs(ax_poss.size());
    std::adjacent_difference(ax_poss.begin(), ax_poss.end(), diffs.begin());

    for (int i = 1; i < diffs.size(); ++i)
    {
      if (diffs[i] < 1.0e-6)
      {
        std::cerr << "Difference between axis points is greater than 1um" << std::endl;
      }
    }

    this->start_z_ = ax_poss[0];
    this->step_z_ = diffs[1];
    this->step_z_inv_ = 1.0 / step_z_;
  }
};




#endif // COILS_HPP