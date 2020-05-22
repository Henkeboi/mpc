#include "rdv_control/mpc.h"
#include <qpOASES.hpp>
#include <iostream>
#include <fstream>
#include <math.h>
#include <vector>
#include <type_traits>
#include <unistd.h>


Mpc::Mpc()
: _last_delta_u(0)
,_plant_points_with_distances(PointsDistances{})
, _traj_points_with_distances(PointsDistances{})
, _mpc_params()
{
    _ref.setZero();
    _pool_setup(true);
    while (_signal_xte.test_and_set(std::memory_order_acquire)) {;}
    while (_signal_setup.test_and_set(std::memory_order_acquire)) {;}
    _init_thread_pool();
}

void Mpc::_init_thread_pool() {
    _thread_pool.push_back(std::thread{&Mpc::_pool_calculate_xte, this});
    _thread_pool.push_back(std::thread{&Mpc::_pool_setup, this, false});
}

void Mpc::set_states(VehicleState&& vehicle_state, TrajectoryState&& trajectory_state) {
    // Vehicle state
    _vehicle_state.x = vehicle_state.x;
    _vehicle_state.y = vehicle_state.y;
    _vehicle_state.theta =  std::atan2(std::sin(vehicle_state.theta), std::cos(vehicle_state.theta));
    _vehicle_state.v = vehicle_state.v;
    _vehicle_state.chi = vehicle_state.chi;
    _vehicle_state.delta = vehicle_state.delta;
    // Trajectory state
    _trajectory_state.centerline = trajectory_state.centerline;
    _trajectory_state.speed_profile = trajectory_state.speed_profile;
    // Vehicle mode
    _vehicle_model.vx = _vehicle_state.v;
}

void Mpc::set_params(const MpcParams& params) {
    _mpc_params.lookahead_first = params.lookahead_first;
    _mpc_params.lookahead = params.lookahead;
    _mpc_params.t_delta = params.t_delta;
    _mpc_params.xte_relative_weight= params.xte_relative_weight;
    _mpc_params.heading_error_relative_weight = params.heading_error_relative_weight;
    _mpc_params.c_ed = params.c_ed;
    _mpc_params.c_psi = params.c_psi;
    _mpc_params.c_delta = params.c_delta;
    _mpc_params.c_psiref = params.c_psiref;
    _mpc_params.c_kapparef_l = params.c_kapparef_l;
    _mpc_params.c_kapparef_u = params.c_kapparef_u;
    _mpc_params.c_Deltadelta = params.c_Deltadelta;
}

void Mpc::_calculate_plant_points(double time_step) {
    double x = _vehicle_state.x;
    double y =  _vehicle_state.y;
    double v = _vehicle_state.v;
    double delta = _vehicle_state.delta;
    double yaw = _vehicle_state.theta;
    // https://www.researchgate.net/publication/269860931_Towards_Time-Optimal_Race_Car_Driving_Using_Nonlinear_MPC_in_Real-Time
    const double c1 = _vehicle_model.l_r / ( _vehicle_model.l_r + _vehicle_model.l_f);
    const double c2 = 1 / (_vehicle_model.l_r + _vehicle_model.l_f);
    Points plant_points;
    plant_points.resize(2, _horizon + _extra_data_points);
    Distances prev_d;
    prev_d.resize(plant_points.cols());
    for (int i = 0; i < _horizon + _extra_data_points; ++i) {
        plant_points(0, i) = x;
        plant_points(1, i) = y;
        x += time_step * v * cos(yaw + (c1 * delta));
        y += time_step * v * sin(yaw + (c1 * delta));
        yaw += time_step * v * delta * c2;
        if (i == 0) prev_d(0) = std::hypot(plant_points(0, i), plant_points(1, i));
        else prev_d(i) = prev_d(i - 1) + 1;
    }
    _plant_points_with_distances = std::move(std::make_pair(std::move(plant_points), std::move(prev_d)));
}

void Mpc::_calculate_traj_points(double v, double d, double time_step) {
    const Points& plant_points = std::get<0>(_plant_points_with_distances);
    Points traj_points;
    traj_points.resize(2, plant_points.cols());
    Distances prev_d;
    prev_d.resize(plant_points.cols());
    for (int i = 0; i < plant_points.cols(); ++i) {
        d = _trajectory_state.centerline.project(Eigen::Vector2d{plant_points(0, i), plant_points(1, i)}, d, 1.0e-3, 0.999); // NAN error in xte if a new d value is not returned!
        if (d - prev_d(i) < 10 * std::numeric_limits<double>::epsilon()) { // Prevent NAN error when calculating xte
            d += time_step * v;
        }
        Eigen::Vector2d traj_coordinates = std::move(_trajectory_state.centerline.evaluate(d, 0));
        traj_points(0, i) = std::move(traj_coordinates(0));
        traj_points(1, i) = std::move(traj_coordinates(1));
        prev_d(i) = d;
    }
    _traj_points_with_distances = std::move(std::make_pair(std::move(traj_points), std::move(prev_d)));
}

Mpc::Splines Mpc::_get_splines() {
    const auto& [plant_points, plant_points_distances] = _plant_points_with_distances;
    const auto& [traj_points, traj_points_distances] = _traj_points_with_distances;
    Spline plant_spline(plant_points, plant_points_distances, 3, 0);
    Spline traj_spline(traj_points, traj_points_distances, 3, 0);
    return std::make_pair(std::move(plant_spline), std::move(traj_spline));
}

void Mpc::_pool_calculate_xte() {
    // Crosstrack error: https://uknowledge.uky.edu/cgi/viewcontent.cgi?article=1040&context=bae_facpub
    while (true) {
        double ref[_horizon];
        while (_signal_xte.test_and_set(std::memory_order_acquire)) { std::this_thread::sleep_for(std::chrono::microseconds(100)); }
        for (int lookahead = 0; lookahead < _horizon; ++lookahead) {
            double xc = std::get<0>(_plant_points_with_distances)(0, lookahead + 1);
            double yc = std::get<0>(_plant_points_with_distances)(1, lookahead + 1);
            double xp = std::get<0>(_plant_points_with_distances)(0, lookahead);
            double yp = std::get<0>(_plant_points_with_distances)(1, lookahead);
            double x1 = std::get<0>(_traj_points_with_distances)(0, lookahead + 1);
            double y1 = std::get<0>(_traj_points_with_distances)(1, lookahead + 1);
            double x2 = std::get<0>(_traj_points_with_distances)(0, lookahead + 2);
            double y2 = std::get<0>(_traj_points_with_distances)(1, lookahead + 2);

            // Minimum distance between heading vector and current plant position
            double a = (y2 - y1) / (x2 - x1);
            double b = -1;
            double c = y1 - a * x1;
            double xm = ((b * (b * xc - a * yc)) - (a * c)) / (pow(a, 2) + pow(b, 2));
            double ym = ((a * (-b * xc + a * yc)) - (b * c)) / (pow(a, 2) + pow(b, 2));
            // TODO: Skipping check (xm, ym) 
            double crossproduct_xte = ((xc - xp) * (ym - yc)) - ((yc - yp) * (xm - xc));
            double xte_scalar = crossproduct_xte * std::hypot(xm - xc, ym - yc);
            if (crossproduct_xte != crossproduct_xte) { 
                ref[_horizon - lookahead - 1] = 0;
            } else {
                ref[_horizon - lookahead - 1] = 120 * xte_scalar;
            }
        }
        for (int i = 0; i < _horizon; ++i) {
            _ref(1 + 2 * i) = ref[i];
        }
        _signal_xte.clear(std::memory_order_release);  
    }
}

void Mpc::_calculate_heading_error(const Splines& splines) {
    auto& [plant_spline, traj_spline] = splines;
    auto& [plant_points, plant_points_distances] = _plant_points_with_distances;
    auto& [traj_points, traj_points_distances] = _traj_points_with_distances;

    for (int lookahead = 0; lookahead < _horizon; ++lookahead) {
        double xp1 = plant_points(0, lookahead);
        double yp1 = plant_points(1, lookahead);
        double xp2 = plant_points(0, lookahead + 1);
        double yp2 = plant_points(1, lookahead + 1);
        double xt1 = traj_points(0, lookahead);
        double yt1 = traj_points(1, lookahead);
        double xt2 = traj_points(0, lookahead + 1);
        double yt2 = traj_points(1, lookahead + 1);
        double crossproduct_he = ((xp2 - xp1) * (yt2 - yt1)) - ((xt2 - xt1) * (yp2 - yp1));

        double x_traj_component = traj_spline.evaluate(traj_points_distances(lookahead + 1), 0)(0);
        double y_traj_component = traj_spline.evaluate(traj_points_distances(lookahead + 1), 0)(1);
        double x_plant_component = plant_spline.evaluate(plant_points_distances(lookahead + 1), 0)(0);
        double y_plant_component = plant_spline.evaluate(plant_points_distances(lookahead + 1), 0)(1);
        Eigen::Vector2d tv;
        Eigen::Vector2d pv;
        tv << x_traj_component, y_traj_component;
        pv << x_plant_component, y_plant_component;
        double yaw_error = acos(tv.dot(pv) / (tv.norm() * pv.norm())) * crossproduct_he;
        if (yaw_error != yaw_error) {
            yaw_error = 0;
        }
        _ref(2 * (_horizon - lookahead - 1), 0) = 1000 * yaw_error;
    }
}

double Mpc::step(double d, bool first_lap) {    
    USING_NAMESPACE_QPOASES
    _signal_setup.clear(std::memory_order_release);  
    double v = _vehicle_state.v;
    double delta = _vehicle_state.delta;
    double time_step = 0;
    if (first_lap) time_step = _mpc_params.lookahead_first / (static_cast<double>(_horizon) * v); // Look x meters ahead
    else time_step = _mpc_params.lookahead / (static_cast<double>(_horizon) * v); // Look x meters ahead

    _calculate_plant_points(time_step);
    _calculate_traj_points(v, d, time_step);
    _signal_xte.clear(std::memory_order_release);  
    Splines splines = std::move(_get_splines());
    _calculate_heading_error(splines);
    while (_signal_setup.test_and_set(std::memory_order_acquire)) {;}
    while (_signal_xte.test_and_set(std::memory_order_acquire)) {;}
    G = H * U_delta + F.transpose() * _ref;

    SQProblem tracker(1, 6);
    Options myOptions;
    myOptions.setToMPC();
    myOptions.printLevel = PL_LOW;
    tracker.setOptions(myOptions);

    int iterations = 10000000;
    tracker.init(H.data(), G.data(), A_constraints, lb, ub, lbA, ubA, iterations);
    real_t first_result[1];
    tracker.getPrimalSolution(first_result);

    double u = first_result[0];
    double t_delta = 0.1;
    double result = (t_delta  * u) + delta;
    _last_delta_u = t_delta * u;
    ++_step_counter;
    return result;
}

void Mpc::_pool_setup(bool init) {
    // https://moodle.fel.cvut.cz/pluginfile.php/234269/mod_folder/content/0/L_3_4_mpc_tracking.pdf
    bool stop = false;
    while (!stop) {
        while (_signal_setup.test_and_set(std::memory_order_acquire)) { std::this_thread::sleep_for(std::chrono::milliseconds(5)); }
        // B
        B.topLeftCorner(5, 1) << 0, 0, 1, 0, 0;
        B.bottomLeftCorner(1, 1) << 1;
        // A
        A.topLeftCorner(5, 5) << 0,   _vehicle_model.vx,    _vehicle_model.vx*_vehicle_model.l_r/_vehicle_model.L,  -_vehicle_model.vx,   0,
                                 0,   0,                    _vehicle_model.vx/_vehicle_model.L,                     0,                    0,
                                 0,   0,                    0,                                                      0,                    0,
                                 0,   0,                    0,                                                      0,                     _vehicle_model.vx,
                                 0,   0,                    0,                                                      0,                    0;
        A.bottomLeftCorner(1, 5).setZero();
        A.topRightCorner(5, 1) = B.topLeftCorner(5, 1);
        A(5, 5) = _last_delta_u;
        // C
        C << 1, 0, 0,  0, 0, 0,
             0, 1, 0, -1, 0, 0;
        // R
        R.setZero();
        for (int i = 0; i < R.cols(); ++i) {
            R(i, i) = 0.0001;
        }
        // Q
        Eigen::Matrix<double, 2, 2> Qk;
        Qk << _mpc_params.xte_relative_weight,  0,
              0,                                _mpc_params.heading_error_relative_weight;
        Q.setZero();
        int Q_inc = 1;
        for (int i = 0; i < Q.rows(); i += 6) {
            Q.block<6, 6>(i, i) = C.transpose() * Qk * C * Q_inc;
            Q_inc += 0.00;
        }
        // C_roof
        C_roof.setZero();
        Eigen::Matrix<double, 6, 6> A_exp = A;
        Eigen::Matrix<double, 6 * _horizon, 1> AB_col;
        AB_col.block<6, 1>(0, 0) = B;
        for (int i = 6; i < 6 * _horizon; i += 6) {
            AB_col.block<6, 1>(i, 0) = A_exp * B;
            A_exp = A_exp * A;
        }
        for (int i = 0; i < C_roof.cols(); ++i) {
            C_roof.block(6 * i, i, (6 * _horizon) - (6 * i), 1) = AB_col.topLeftCorner((6 * _horizon) - (6 * i), 1);
        }
        // A_roof
        A_exp = A;
        for (int i = 0; i < A_roof.rows(); i += 6) {
            A_roof.block<6, 6>(i, 0) = A_exp;
            A_exp = A_exp * A;
        }
        // T_roof
        T_roof.setZero();
        for (int i = 0; i < T_roof.rows(); i += 2) {
            T_roof.block(i, 3 * i, 2, 6) = Qk * C;
        }
        // U_delta
        for (int i = 0; i < U_delta.rows(); ++i) {
            U_delta(i, 0) = 0.1;
        }
        // F
        F.topLeftCorner(6, _horizon) = A_roof.transpose() * Q * C_roof;
        F.bottomLeftCorner(2 * _horizon, _horizon) = - T_roof * C_roof;
        // Hessian
        H = C_roof.transpose() * Q * C_roof + R;
        _signal_setup.clear(std::memory_order_release);  
        if (init == true) stop = true;
    }
}
