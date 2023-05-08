#include <cmath>
#include <vector>
#include <iostream>
#include <gsl/gsl_odeiv2.h>
#include <torch/torch.h>

class RateStateModel {
public:
    // Constructor
    RateStateModel(
        int number_time_steps=500,
        double start_time=0.0,
        double end_time=50.0
    ) {
        // Define model constants
        a = 0.011;
        b = 0.014;
        mu_ref = 0.6;
        V_ref = 1;
        k1 = 1e-7;

        // Define time range
        t_start = start_time;
        t_final = end_time;
        num_tsteps = number_time_steps;
        delta_t = (end_time - start_time) / number_time_steps;

        // Define initial conditions
        mu_t_zero = 0.6;
        V_zero = 0.0;

        // Add additional model constants
        RadiationDamping = true;
        Dc = nullptr;
    }

    // Compute the time derivatives of the state variables
    std::vector<double> friction(double t, std::vector<double> y) {
        // Just to help readability
        // y[0] is mu (friction)
        // y[1] is theta
        // y[2] is velocity

        double V_ref = this->V_ref;
        double a = this->a;
        double b = this->b;
        double dc = *(this->Dc);

        // effective spring stiffness
        double kprime = 1e-2 * 10 / dc;

        // loading
        double a1 = 20;
        double a2 = 10;
        double V_l = V_ref * (1 + exp(-t/a1) * sin(a2*t));

        // initialize the vector of derivatives
        std::vector<double> dydt(3, 0.0);

        // compute v
        double temp = 1 / a * (y[0] - this->mu_ref - b * log(V_ref * y[1] / dc));
        double v = V_ref * exp(temp);

        // time derivative of theta
        dydt[1] = 1. - v * y[1] / dc;

        // time derivative of mu
        dydt[0] = kprime * V_l - kprime * v;

        // time derivative of velocity
        dydt[2] = v / a * (dydt[0] - b / y[1] * dydt[1]);

        // add radiation damping term if specified
        if (this->RadiationDamping) {
            // time derivative of mu with radiation damping
            dydt[0] = dydt[0] - this->k1 * dydt[2];
            // time derivative of velocity with radiation damping
            dydt[2] = v / a * (dydt[0] - b / y[1] * dydt[1]);
        }

        return dydt;
    }

    std::vector<std::vector<double>> evaluate(std::vector<double> lstm_model={}) {
        // call either full model evaluation or reduced order model evaluation

        if (!lstm_model.empty()) {
            rom_evaluate(lstm_model);
            std::cerr << "rom_evaluate is not implemented yet!" << std::endl;
            std::vector<std::vector<double>> empty_vector;
            return empty_vector;
        }
        else {
            return full_evaluate();
        }
    }

    std::vector<std::vector<double>> full_evaluate() {
        // Calculate the number of steps to take
        double t_start = 0.0;
        double t_final = 10.0;
        double delta_t = 0.01;
        int num_steps = static_cast<int>(std::floor((t_final - t_start) / delta_t));

        // Create vectors to store trajectory
        std::vector<double> t(num_steps);
        std::vector<double> mu(num_steps);
        std::vector<double> theta(num_steps);
        std::vector<double> velocity(num_steps);
        std::vector<double> acc(num_steps);

        t[0] = t_start;
        mu[0] = 0.05;
        theta[0] = 0.01;
        velocity[0] = 1.0;
        acc[0] = 0.0;

        // Set up the ODE solver
        gsl_odeiv2_system sys = {nullptr, nullptr, 3, nullptr};
        gsl_odeiv2_driver *driver = gsl_odeiv2_driver_alloc_y_new(&sys, gsl_odeiv2_step_rkf45, delta_t, 1e-6, 1e-10);

        // Set the initial conditions for the ODE solver
        double y[3] = {mu[0], theta[0], velocity[0]};
        gsl_odeiv2_driver_apply(driver, &t[0], t[1], y);

        // Integrate the ODE(s) across each delta_t timestep
        int k = 1;
        while (t[k-1] < t_final && k < num_steps) {
            gsl_odeiv2_driver_apply(driver, &t[k-1], t[k], y);
            // Store the results to plot later
            mu[k] = y[0];
            theta[k] = y[1];
            velocity[k] = y[2];
            acc[k] = (velocity[k] - velocity[k-1]) / delta_t;
            k++;
        }

        // Add some noise to the acceleration data to simulate real-world noise
        std::default_random_engine generator(std::chrono::system_clock::now().time_since_epoch().count());
        std::normal_distribution<double> distribution(0.0, 1.0);
        for (int i = 0; i < num_steps; i++) {
            acc[i] += 1.0 * std::abs(acc[i]) * distribution(generator);
        }

        // Create vectors to store data for plotting
        std::vector<std::vector<double>> data(num_steps, std::vector<double>(3));
        for (int i = 0; i < num_steps; i++) {
            data[i][0] = t[i];
            data[i][1] = acc[i];
            data[i][2] = acc[i];
        }

        // Clean up
        gsl_odeiv2_driver_free(driver);
        return data;
    }

    std::vector<std::vector<double>> rom_evaluate(torch::nn::LSTM lstm_model) {
        // Calculate the number of steps to take
        double t_start = 0.0;
        double t_final = 10.0;
        double delta_t = 0.01;
        int num_steps = static_cast<int>(std::floor((t_final - t_start) / delta_t));
        int window = 10;

        // Create vectors to store trajectory
        std::vector<std::vector<double>> t(num_steps, std::vector<double>(2));
        std::vector<std::vector<double>> acc(num_steps, std::vector<double>(2));

        t[0][0] = t_start;

        // Calculate the time array
        for (int k = 1; k < num_steps; k++) {
            t[k][0] = t[k-1][0] + delta_t;
        }

        t[0][1] = 1.0;
        for (int i = 1; i < num_steps; i++) {
            t[i][1] = t[0][1];
        }

        int num_steps_ = static_cast<int>(num_steps / window);
        std::vector<std::vector<double>> train_plt(window, std::vector<double>(2));
        std::vector<double> x_pred(window);

        // Predict acceleration using the LSTM model
        for (int ii = 0; ii < num_steps_; ii++) {
            int start = ii * window;
            int end = start + window;

            for (int i = start; i < end; i++) {
                train_plt[i-start][0] = t[i][0];
                train_plt[i-start][1] = t[i][1];
            }

            torch::Tensor input = torch::from_blob(train_plt.data(), {window, 2});
            std::tie(x_pred, std::ignore) = lstm_model->forward(input.unsqueeze(0));
            std::vector<double> y_pred(x_pred.data_ptr<double>(), x_pred.data_ptr<double>() + x_pred.numel());

            for (int i = start; i < end; i++) {
                acc[i][0] = y_pred[i-start];
                acc[i][1] = y_pred[i-start];
            }
        }

        // Return the time and acceleration arrays
        return acc;
    }

private:
    // Model constants
    double a;
    double b;
    double mu_ref;
    double V_ref;
    double k1;

    // Time range
    double t_start;
    double t_final;
    int num_tsteps;
    double delta_t;

    // Initial conditions
    double mu_t_zero;
    double V_zero;

    // Additional model constants
    bool RadiationDamping;
    double* Dc;
    // Define a struct to store the initial values for the ODE solver
    struct initial_values {
        double mu_t_zero;
        double mu_ref;
        double Dc;
        double V_ref;
        double t_start;
        double t_final;
        double delta_t;
    }

};
