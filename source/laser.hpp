#ifndef LASER_HPP
#define LASER_HPP

#include "mathtools.hpp"
#include "physics.hpp"

constexpr double focus_point_z = -1.4;

constexpr double beam_spread_angle_rad_ = 0.013;

constexpr double waist_at_oven_exit = 0.004;
constexpr double oven_exit_pos_z = 0.63 - 0.83 - 0.15;

__forceinline double beamWaist(double posOnAxis)
{
  return waist_at_oven_exit + (posOnAxis - oven_exit_pos_z)* beam_spread_angle_rad_;
}





#endif // COILS_HPP