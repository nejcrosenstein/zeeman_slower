﻿#include <iostream>
#include <random>
#include <fstream>
#include <array>

#include "mathtools.hpp"
#include "physics.hpp"
#include "coils.hpp"
#include "beam.hpp"
#include "xorshift.hpp"
#include "histogram.hpp"
#include "laser.hpp"

using Uniform = std::uniform_real_distribution<double>;
using Exponential = std::exponential_distribution<double>;
using RandomGenerator = XorshiftStar;

using Vecd = Vec3D<double>;

struct ParticleQuadruple
{
  alignas(32) double pos_x[4];
  alignas(32) double pos_y[4];
  alignas(32) double pos_z[4];

  alignas(32) double vel_x[4];
  alignas(32) double vel_y[4];
  alignas(32) double vel_z[4];
};

__forceinline void __vectorcall computeLightDirection(
  ParticleQuadruple const& atoms,
  __m256d& dir_x,
  __m256d& dir_y,
  __m256d& dir_z)
{
  // position (packed representation)
  __m256d pos[3];
  pos[0] = _mm256_load_pd(&atoms.pos_x[0]);
  pos[1] = _mm256_load_pd(&atoms.pos_y[0]);
  pos[2] = _mm256_load_pd(&atoms.pos_z[0]);

  // focus point
  __m256d foc[3]; 
  foc[0] = _mm256_setzero_pd();
  foc[1] = _mm256_setzero_pd();
  foc[2] = _mm256_set1_pd(focus_point_z);

  __m256d dirs[3];
  for (int i = 0; i < 3; ++i)
    dirs[i] = _mm256_sub_pd(foc[i], pos[i]);

  __m256d squared[3];
  for (int i = 0; i < 3; ++i)
    squared[i] = _mm256_mul_pd(dirs[i], dirs[i]);

  __m256d squared_components_sum =
    _mm256_add_pd(squared[0], _mm256_add_pd(squared[1], squared[2]));

  __m256d norm = _mm256_sqrt_pd(squared_components_sum);
  __m256d norm_inv = _mm256_div_pd(_mm256_set1_pd(1.0), norm);

  dir_x = _mm256_mul_pd(dirs[0], norm_inv);
  dir_y = _mm256_mul_pd(dirs[1], norm_inv);
  dir_z = _mm256_mul_pd(dirs[2], norm_inv);
}

__forceinline __m256d __vectorcall projectVelocityToLightDir(
  ParticleQuadruple const& atoms,
  __m256d const& dir_x,
  __m256d const& dir_y,
  __m256d const& dir_z)
{
  // velocity components (packed representation)
  __m256d vel[3];
  vel[0] = _mm256_load_pd(&atoms.vel_x[0]);
  vel[1] = _mm256_load_pd(&atoms.vel_y[0]);
  vel[2] = _mm256_load_pd(&atoms.vel_z[0]);

  // scalar product
  __m256d prod0 = _mm256_mul_pd(vel[0], dir_x);
  __m256d prod1 = _mm256_mul_pd(vel[1], dir_y);
  __m256d prod2 = _mm256_mul_pd(vel[2], dir_z);

  return _mm256_add_pd(prod0, _mm256_add_pd(prod1, prod2));
}

template<class RandomGen_t>
static void singlePhotonScatter(
  ParticleQuadruple& atoms,
  RandomGen_t& rand_gen,
  ZeemanSlower const& slower,
  ImportedField const& quadrupole)
{ 
  // Current position and velocity
  __m256d curr_pos_x = _mm256_load_pd(atoms.pos_x);
  __m256d curr_pos_y = _mm256_load_pd(atoms.pos_y);
  __m256d curr_pos_z = _mm256_load_pd(atoms.pos_z);
  
  __m256d curr_vel_x = _mm256_load_pd(atoms.vel_x);
  __m256d curr_vel_y = _mm256_load_pd(atoms.vel_y);
  __m256d curr_vel_z = _mm256_load_pd(atoms.vel_z);

  __m256d intensity = lightIntensity(curr_pos_x, curr_pos_y, curr_pos_z);
  // TODO: make a parameter
  __m256d detuning = _mm256_set1_pd(10.0e6);

  __m256d r_positions = _mm256_setzero_pd();

  // TODO: combine both fields before simulation starts
  __m256d slower_fields = interpolate(slower, r_positions, curr_pos_z);
  __m256d quad_fields = interpolate(quadrupole, r_positions, curr_pos_z);
  __m256d field_tesla = _mm256_add_pd(slower_fields, quad_fields);

  __m256d light_dir[3];
  computeLightDirection(atoms, light_dir[0], light_dir[1], light_dir[2]);

  __m256d vel_along_light_dir = projectVelocityToLightDir(
                                 atoms, light_dir[0], light_dir[1], light_dir[2]);
  
  __m256d scatter_rates = scatteringRate(vel_along_light_dir, intensity, detuning, field_tesla);

  __m256d time_step = _mm256_set1_pd(0.2 * excited_state_lifetime);



  __m256d step_x = _mm256_mul_pd(curr_vel_x, time_step);
  __m256d step_y = _mm256_mul_pd(curr_vel_y, time_step);
  __m256d step_z = _mm256_mul_pd(curr_vel_z, time_step);

  _mm256_store_pd(&atoms.pos_x[0], _mm256_add_pd(curr_pos_x, step_x));
  _mm256_store_pd(&atoms.pos_y[0], _mm256_add_pd(curr_pos_y, step_y));
  _mm256_store_pd(&atoms.pos_z[0], _mm256_add_pd(curr_pos_z, step_z));

  __m256d probabilites = _mm256_mul_pd(scatter_rates, time_step);

  alignas(32) double probs[4];
  _mm256_store_pd(&probs[0], probabilites);

  double vel_change_mags[4] = { 0, 0, 0, 0 };
  double emm_dir_x[4] = { 0, 0, 0, 0 };
  double emm_dir_y[4] = { 0, 0, 0, 0 };
  double emm_dir_z[4] = { 0, 0, 0, 0 };
  for (int i = 0; i < 4; ++i)
  {
    bool scatter_occurs = rand_gen.random(0.0, 1.0) < probs[i];
    if (scatter_occurs)
    {
      auto phi = rand_gen.random(0.0, two_pi);
      auto cos_theta = rand_gen.random(-1.0, 1.0);

      emm_dir_x[i] = cos(phi)*sqrt(1.0 - sq(cos_theta));
      emm_dir_y[i] = sin(phi)*sqrt(1.0 - sq(cos_theta));
      emm_dir_z[i] = cos_theta;

      vel_change_mags[i] = recoil_velocity; // else remains 0
    }
  }

  __m256d change_x = _mm256_add_pd(light_dir[0], _mm256_load_pd(emm_dir_x));
  __m256d change_y = _mm256_add_pd(light_dir[1], _mm256_load_pd(emm_dir_y));
  __m256d change_z = _mm256_add_pd(light_dir[2], _mm256_load_pd(emm_dir_z));

  __m256d vel_recoil = _mm256_load_pd(&vel_change_mags[0]);
    
  __m256d new_vel_x = _mm256_add_pd(curr_vel_x, _mm256_mul_pd(change_x, vel_recoil));
  __m256d new_vel_y = _mm256_add_pd(curr_vel_y, _mm256_mul_pd(change_y, vel_recoil));
  __m256d new_vel_z = _mm256_add_pd(curr_vel_z, _mm256_mul_pd(change_z, vel_recoil));

  _mm256_store_pd(&atoms.vel_x[0], new_vel_x);
  _mm256_store_pd(&atoms.vel_y[0], new_vel_y);
  _mm256_store_pd(&atoms.vel_z[0], new_vel_z);
}

template <typename T, typename Compare>
std::vector<std::size_t> sort_permutation(
  const std::vector<T>& vec,
  Compare& compare)
{
  std::vector<std::size_t> p(vec.size());
  std::iota(p.begin(), p.end(), 0);
  std::sort(p.begin(), p.end(),
    [&](std::size_t i, std::size_t j) { return compare(vec[i], vec[j]); });
  return p;
}

struct InitialStates
{
  template<class RandGen_t>
  InitialStates(
    BeamVelocityDistribution const& init_velocity, int number, RandGen_t& rand_gen)
  {
    for (int i = 0; i < number; ++i)
    {
      double vel_magnitude = init_velocity(rand_gen);

      double init_beam_radius = 0.003; // TODO: model nozzle
      Uniform radius(0.0, init_beam_radius);

      Uniform phi(0.0, two_pi);
      Uniform cos_theta_initvel(cos(beam_spread_angle_rad_), 1.0);

      Vecd vel_init = to_cartesian(phi(rand_gen), cos_theta_initvel(rand_gen))*vel_magnitude;

      double r = radius(rand_gen);
      double angle = phi(rand_gen);
      Vecd pos_init = Vecd(r*cos(angle), r*sin(angle), -0.2);

      positions_.push_back(pos_init);
      velocities_.push_back(vel_init);
    }

    std::vector<std::size_t> per(number);
    std::iota(per.begin(), per.end(), 0);
    std::sort(per.begin(), per.end(),
      [this](std::size_t i, std::size_t j) { return this->velocities_[i].z_ < this->velocities_[j].z_; });


    std::vector<Vecd> sorted(number);
    for (int i = 0; i < number; ++i) sorted[i] = velocities_[per[i]];
    velocities_ = sorted;


    for (int i = 0; i < number; ++i) sorted[i] = positions_[per[i]];
    positions_ = sorted;



    taken_ = 0;
  }

  int taken_;
  std::vector<Vecd> positions_;
  std::vector<Vecd> velocities_;
};

template<class RanGen_t>
void simulateQuadruplePath(RanGen_t& rand_gen,
  ZeemanSlower const& slower,
  ImportedField const& quadrupole,
  InitialStates& init_states,
  std::array<Histogram, 4>& hists)
{
  ParticleQuadruple atoms;

  for (int i = 0; i < 4; ++i)
  {
    int& ix = init_states.taken_;

    auto pos = init_states.positions_[ix];
    auto vel = init_states.velocities_[ix];

    atoms.pos_x[i] = pos.x_;
    atoms.pos_y[i] = pos.y_;
    atoms.pos_z[i] = pos.z_;

    atoms.vel_x[i] = vel.x_;
    atoms.vel_y[i] = vel.y_;
    atoms.vel_z[i] = vel.z_;

    init_states.taken_++;
  }

  for (int i = 0; i < 6000000; ++i)
  {
    int at_end = 0;
    for (int i = 0; i < 4; ++i)
    {
      hists[i].addSample(atoms.pos_z[i], atoms.vel_z[i]);
      at_end += atoms.pos_z[i] > 0.8 ? 1 : 0;
    }
    if (at_end == 4) break;

    singlePhotonScatter(atoms, rand_gen, slower, quadrupole);
  }
}




void simulation()
{
  RandomGenerator random_gen(2); // TODO seed

  ZeemanSlower slower(2.6, 1.3, -0.3, 1.0, 1000);
  ImportedField quadrupole(0.74, "quadrupole.txt");
  BeamVelocityDistribution init_velocity_dist(353.0);

  InitialStates init_states(init_velocity_dist, 800, random_gen);

  Histogram hist(-50.0, 2.0, 200, -0.2, 0.001, 1000);

  std::array<Histogram, 4> hists = { hist, hist, hist, hist };

  for (int j = 0; j < 200; ++j)
  {
    simulateQuadruplePath(random_gen, slower, quadrupole, init_states, hists);

    for (auto const& h : hists)
      for (int i = 0; i < h.histogram_.size(); ++i)
        hist.histogram_[i] = h.histogram_[i];

    std::cout << " iteration  " << j << std::endl;
  }

  std::ofstream out;
  out.open("out.txt");

  std::cout << "here";

  int ix = 0;
  for (int i = 0; i < hist.bins_vel_.number_of_bins_; ++i)
  {
    for (int j = 0; j < hist.bins_pos_.number_of_bins_; ++j)
    {
      out << hist.histogram_[ix++] << ",";
    }
    out << std::endl;
  }
}


int main(int argc, char* argv[])
{

  std::string out_dir = "D:\\Magisterij\\magisterij\\Graphs\\phasespaces\\try";

  uint32_t number_of_particles_ = 10000;

  simulation();

  std::cout << "ended" << std::endl;
  
  
  return 0;
}
