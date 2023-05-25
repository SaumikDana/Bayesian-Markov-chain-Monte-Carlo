#include "RateStateModel.h"
#include <iostream>
#include <cmath>
#include <vector>
#include <random>
#include "/opt/homebrew/Cellar/gsl/2.7.1/include/gsl/gsl_odeiv2.h"
#include "/opt/homebrew/Cellar/gsl/2.7.1/include/gsl/gsl_errno.h"


RateStateModel::RateStateModel(int number_time_steps, double start_time, double end_time)
    : a(0.011), b(0.014), mu_ref(0.6), V_ref(1), k1(1e-7),
      t_start(start_time), t_final(end_time), num_tsteps(number_time_steps),
      delta_t((end_time - start_time) / number_time_steps),
      mu_t_zero(0.6), Dc(0.0), RadiationDamping(true) {}

void RateStateModel::setA(double value) {
    a = value;
}

void RateStateModel::setB(double value) {
    b = value;
}

void RateStateModel::setMuRef(double value) {
    mu_ref = value;
}

void RateStateModel::setVRef(double value) {
    V_ref = value;
}

void RateStateModel::setK1(double value) {
    k1 = value;
}

void RateStateModel::setTStart(double value) {
    t_start = value;
}

void RateStateModel::setTFinal(double value) {
    t_final = value;
}

void RateStateModel::setMuTZero(double value) {
    mu_t_zero = value;
}

void RateStateModel::setDc(double value) {
    Dc = value;
}

// Getter function for Dc
double RateStateModel::getDc() {
    return Dc;
}

void RateStateModel::setRadiationDamping(bool value) {
    RadiationDamping = value;
}

int RateStateModel::getNumTimesteps() const {
    return num_tsteps;
}

int RateStateModel::friction(double t, const double y[], double dydt[], void* params) {
    // Just to help readability
    // y[0] is mu (friction)
    // y[1] is theta
    // y[2] is velocity

    // Cast the parameters to the appropriate types
    double* p = static_cast<double*>(params);
    double V_ref = p[0];
    double a = p[1];
    double b = p[2];
    double dc = p[3];
    double mu_ref = p[4];
    bool RadiationDamping = *reinterpret_cast<bool*>(p + 5);
    double k1 = p[6];

    // Effective spring stiffness
    double kprime = 1e-2 * 10 / dc;

    // Loading
    double a1 = 20;
    double a2 = 10;
    double V_l = V_ref * (1 + exp(-t / a1) * sin(a2 * t));

    // Compute v
    double temp = 1 / a * (y[0] - mu_ref - b * log(V_ref * y[1] / dc));
    double v = V_ref * exp(temp);

    // Time derivative of theta
    dydt[1] = 1. - v * y[1] / dc;

    // Time derivative of mu
    dydt[0] = kprime * V_l - kprime * v;

    // Time derivative of velocity
    dydt[2] = v / a * (dydt[0] - b / y[1] * dydt[1]);

    // Add radiation damping term if specified
    if (RadiationDamping) {
        // Time derivative of mu with radiation damping
        dydt[0] = dydt[0] - k1 * dydt[2];
        // Time derivative of velocity with radiation damping
        dydt[2] = v / a * (dydt[0] - b / y[1] * dydt[1]);
    }

    return GSL_SUCCESS;
}

void RateStateModel::evaluate(std::vector<double>& t, std::vector<double>& acc, std::vector<double>& acc_noise) {
    std::vector<double> mu(num_tsteps), theta(num_tsteps), velocity(num_tsteps);
    t.resize(num_tsteps);
    acc.resize(num_tsteps);
    acc_noise.resize(num_tsteps);

    t[0] = t_start;
    mu[0] = mu_ref;
    theta[0] = Dc / V_ref;
    velocity[0] = V_ref;
    acc[0] = 0.0;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> dist(0.0, 1.0);

    // Create GSL workspace and ODE system
    const gsl_odeiv2_step_type* step_type = gsl_odeiv2_step_rkf45;
    double params[7] = { V_ref, a, b, Dc, mu_ref, static_cast<double>(RadiationDamping), k1 };
    gsl_odeiv2_system sys = { friction, nullptr, 3, params };

    gsl_odeiv2_driver* driver = gsl_odeiv2_driver_alloc_y_new(&sys, step_type, 1e-6, 1e-10, 0.0);

    double t_current = t_start;
    double y[3] = { mu_ref, Dc / V_ref, V_ref };
    double h = delta_t;

    for (int k = 1; k < num_tsteps; ++k) {
        double t_next = t_current + h;

        int status = gsl_odeiv2_driver_apply(driver, &t_current, t_next, y);

        if (status != GSL_SUCCESS) {
            // Handle integration error
            std::cout << "Integration error occurred at time: " << t_current << std::endl;
            // You can choose to exit the loop or take any other appropriate action
            break;
        }

        mu[k] = y[0];
        theta[k] = y[1];
        velocity[k] = y[2];

        acc[k] = (velocity[k] - velocity[k - 1]) / delta_t;
        acc_noise[k] = acc[k] + 1.0 * std::abs(acc[k]) * dist(gen);
        t[k] = t_next;
    }

    gsl_odeiv2_driver_free(driver);
}
