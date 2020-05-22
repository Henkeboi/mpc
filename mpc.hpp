#ifndef __MPC__HPP__
#define __MPC__HPP__

#define EIGEN_STACK_ALLOCATION_LIMIT 0

#include <ros/ros.h>
#include <std_msgs/Float64.h>
#include <std_msgs/Float64MultiArray.h>
#include <Eigen/Dense>
#include "rdv_utils/spline.h"
#include "rdv_control/feedback_linearization.h"

#include <thread>
#include <atomic>
#include <fstream>
#include <chrono>

struct VehicleModel {
    double l_f = 0.813;
    double l_r = 0.717;
    double L = 0.813 + 0.717;
    double vx = 10;
};

struct VehicleState {
    VehicleState(double v_start= 10) : v(v_start) {}
    double x;
    double y;
    double theta;
    double v;
    double chi;
    double delta;
};

struct TrajectoryState {
    Spline2d centerline;
    Spline2d speed_profile;
};

struct MpcParams {
    double lookahead_first;
    double lookahead;
    double t_delta;
    double xte_relative_weight;
    double heading_error_relative_weight;
    double c_ed = 1.0;
    double c_psi = 2.0 * M_PI;
    double c_delta = 2.0 * M_PI;
    double c_psiref = 2.0 * M_PI;
    double c_kapparef_l = 0.0;
    double c_kapparef_u = 1.0;
    double c_Deltadelta = M_PI;
};

template<typename TimeT = std::chrono::microseconds>
struct measure {
    template<typename F, typename ...Args>
    static auto duration(F&& func, Args&&... args) {
        auto start = std::chrono::high_resolution_clock::now();
        std::forward<decltype(func)>(func)(std::forward<Args>(args)...);
        return std::chrono::duration_cast<TimeT>(std::chrono::high_resolution_clock::now() - start);
    }
};

class Mpc {
public:
    Mpc();
    void set_states(VehicleState&& vehicle_state, TrajectoryState&& trajectory_state);
    void set_params(const MpcParams& params);
    double step(double t, bool first_lap);
private:
    typedef Eigen::MatrixXd Points;
    typedef Eigen::VectorXd Distances;
    typedef std::pair<Spline, Spline> Splines;
    typedef std::pair<Points, Distances> PointsDistances;
    int _step_counter = 0;
    static const int _horizon = 20; // Should not be less than 3
    static const int _extra_data_points = 2;
    double _last_delta_u;
    VehicleModel _vehicle_model;
    VehicleState _vehicle_state;
    TrajectoryState _trajectory_state;
    Spline2d _centerline;
    MpcParams _mpc_params;

    void _init_thread_pool();
    std::atomic_flag _signal_xte = ATOMIC_FLAG_INIT;
    std::atomic_flag _signal_setup = ATOMIC_FLAG_INIT;
    std::vector<std::thread> _thread_pool;
    PointsDistances _plant_points_with_distances;
    PointsDistances _traj_points_with_distances;
    void _pool_calculate_xte();
    void _pool_setup(bool init);

    void _calculate_plant_points(double time_step);
    void _calculate_traj_points(double v, double d, double time_step);
    void _calculate_heading_error(const Splines& splines);
    Splines _get_splines();

    // MPC matrixes 
    Eigen::Matrix<double, 2 * _horizon, 1> _ref;
    Eigen::Matrix<double, 6, 6> A;
    Eigen::Matrix<double, 6, 1> B;
    Eigen::Matrix<double, 2, 6> C;
    Eigen::Matrix<double, 6 * _horizon, 6 * _horizon> Q;
    Eigen::Matrix<double, _horizon, _horizon> R;
    Eigen::Matrix<double, 6 * _horizon, _horizon> C_roof;
    Eigen::Matrix<double, 6 * _horizon, 6> A_roof;
    Eigen::Matrix<double, 2 * _horizon, 6 * _horizon> T_roof;
    Eigen::Matrix<double, 2 * _horizon, _horizon> F;
    Eigen::Matrix<double, _horizon, _horizon> H;
    Eigen::Matrix<double, _horizon, 1> G;
    Eigen::Matrix<double, _horizon, 1> U_delta;

    // MPC constraints
    double lb[1] = {-M_PI / 3};
    double ub[1] = {M_PI / 3};
    double A_constraints[6*6] = {0,   _vehicle_model.vx,  _vehicle_model.vx * _vehicle_model.l_r / _vehicle_model.L,    -_vehicle_model.vx,   0,                    1,
                                 0,   0,                  _vehicle_model.vx / _vehicle_model.L,                         0,                    0,                    0,
                                 0,   0,                  0,                                                            0,                    0,                    0,
                                 0,   0,                  0,                                                            0,                    _vehicle_model.vx,    0,
                                 0,   0,                  0,                                                            0,                    0,                    0,
                                 0,   0,                  0,                                                            0,                    0,                    0};

    // Lower and upper bounds on constrains A
    double c_ed = _mpc_params.c_ed;
    double c_psi = _mpc_params.c_psi;
    double c_delta = _mpc_params.c_delta;
    double c_psiref = _mpc_params.c_psiref;
    double c_kapparef_l = _mpc_params.c_kapparef_l;
    double c_kapparef_u = _mpc_params.c_kapparef_u;
    double c_Deltadelta = _mpc_params.c_Deltadelta;
    double lbA[6] = {-c_ed, -c_psi, -c_delta, -c_psiref, -c_kapparef_l, -c_Deltadelta};
    double ubA[6] = {c_ed,  c_psi, c_delta, c_psiref, c_kapparef_u, c_Deltadelta};
};

#endif // __MPC__HPP__
