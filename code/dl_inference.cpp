#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <gnuplot-iostream.h>

using namespace std;

class RateStateModel {
public:
    int num_tsteps;
    double Dc;
    double t_start;
    double t_final;

    double random_normal() {
        static std::mt19937 generator;
        static std::normal_distribution<double> distribution(0.0, 1.0);
        return distribution(generator);
    }

    RateStateModel() {
        num_tsteps = 100;
        Dc = 0.0;
        t_start = 0.0;
        t_final = 10.0;
    }

    vector<vector<double> > evaluate() {
        vector<vector<double> > result(num_tsteps, vector<double>(2));

        double a = 0.011;
        double b = 0.014;
        double mu_ref = 0.6;
        double V_ref = 1.0;
        double k1 = 1e-7;
        double t_start = 0.0;
        double t_final = 10.0;
        int num_tsteps = 100;
        double delta_t = (t_final - t_start) / num_tsteps;
        double mu_t_zero = 0.6;
        bool RadiationDamping = true;
        double Dc = this->Dc;

        vector<double> t(num_tsteps);
        vector<double> mu(num_tsteps);
        vector<double> theta(num_tsteps);
        vector<double> velocity(num_tsteps);
        vector<double> acc(num_tsteps);

        t[0] = t_start;
        mu[0] = mu_ref;
        theta[0] = Dc / V_ref;
        velocity[0] = V_ref;
        acc[0] = 0.0;

        int num_steps = static_cast<int>(floor((t_final - t_start) / delta_t));

        auto friction = [&](double t, vector<double> y) {
            double V_ref = V_ref;
            double a = a;
            double b = b;
            double dc = Dc;

            double kprime = 1e-2 * 10 / dc;
            double a1 = 20;
            double a2 = 10;
            double V_l = V_ref * (1 + exp(-t/a1) * sin(a2*t));

            vector<double> dydt(3, 0.0);
            double temp = 1 / a * (y[0] - mu_ref - b * log(V_ref * y[1] / dc));
            double v = V_ref * exp(temp);
            dydt[1] = 1. - v * y[1] / dc;
            dydt[0] = kprime * V_l - kprime * v;
            dydt[2] = v / a * (dydt[0] - b / y[1] * dydt[1]);

            if (RadiationDamping) {
                dydt[0] = dydt[0] - k1 * dydt[2];
                dydt[2] = v / a * (dydt[0] - b / y[1] * dydt[1]);
            }

            return dydt;
        };

        vector<double> y(3, 0.0);
        y[0] = mu_t_zero;
        y[1] = theta[0];
        y[2] = velocity[0];

        int k = 1;
        while (k < num_steps) {
            vector<double> yprime = friction(t[k - 1], y);
            for (int i = 0; i < 3; i++) {
                y[i] += delta_t * yprime[i];
            }

            t[k] = t[k - 1] + delta_t;
            mu[k] = y[0];
            theta[k] = y[1];
            velocity[k] = y[2];
            acc[k] = (velocity[k] - velocity[k - 1]) / delta_t;

            k++;
        }

        vector<double> acc_noise(acc.size(), 0.0);
        for (size_t i = 0; i < acc.size(); i++) {
            acc_noise[i] = acc[i] + 1.0 * abs(acc[i]) * random_normal();
        }

        vector<vector<double> > t_acc_noise(num_steps, vector<double>(2));
        vector<vector<double> > acc_vec(num_steps, vector<double>(2));

        for (int i = 0; i < num_steps; i++) {
            t_acc_noise[i][0] = t[i];
            t_acc_noise[i][1] = Dc;
            acc_vec[i][0] = acc[i];
            acc_vec[i][1] = acc[i];
        }

        return t_acc_noise;
    }


};

class rsf {
public:
    int num_dc;
    double lowest_slip_value;
    double largest_slip_value;
    vector<double> dc_list;
    string lstm_file;
    string data_file;
    int num_features;
    bool plotfigs;
    RateStateModel model;
    vector<vector<double> > t_appended;
    vector<vector<double> > acc_appended;

    rsf(int number_slip_values = 1, double lowest_slip_value = 1.0, double largest_slip_value = 1000.0, bool plotfigs = false)
        : num_dc(number_slip_values), lowest_slip_value(lowest_slip_value), largest_slip_value(largest_slip_value), plotfigs(plotfigs) {
        
        // Define the range of values for the critical slip distance
        double start_dc = lowest_slip_value;
        double end_dc = largest_slip_value;
        for (int i = 0; i < num_dc; i++) {
            double dc = start_dc + (end_dc - start_dc) * (i / (double)(num_dc - 1));
            dc_list.push_back(dc);
        }

        // Define file names for saving and loading data and LSTM model
        lstm_file = "model_lstm.pickle";
        data_file = "data.pickle";
        num_features = 2;
    }

    void generate_time_series() {
        // Create arrays to store the time and acceleration data for all values of dc
        int entries = num_dc * model.num_tsteps;
        t_appended.resize(entries, vector<double>(num_features));
        acc_appended.resize(entries, vector<double>(num_features));
        vector<vector<double> > acc_appended_noise(entries, vector<double>(num_features));
        int count_dc = 0;

        for (double dc : dc_list) {
            // Evaluate the model for the current value of dc
            model.Dc = dc;
            vector<vector<double> > result = model.evaluate();
            int start = count_dc * model.num_tsteps;
            int end = start + model.num_tsteps;
            for (int i = start, j = 0; i < end; i++, j++) {
                t_appended[i][0] = result[j][0];
                t_appended[i][1] = dc;
                for (int index = 0; index < num_features; index++) {
                    acc_appended[i][index] = result[j][index + 1];
                    acc_appended_noise[i][index] = result[j][index + 1];
                }
            }
            count_dc++;
            // Generate plots
            plot_time_series(t_appended, acc_appended);
        }

        // // Store the time and acceleration data
        // // ...

        // // Save the data using pickle
        // save_object(acc_appended_noise, data_file);
    }

    void plot_time_series(const std::vector<std::vector<double>>& t, const std::vector<std::vector<double>>& acc) {
        if (plotfigs) {
            Gnuplot gp;

            // Set plot title
            gp << "set title \"$d_c$=" << model.Dc << " $\\mu m$ RSF solution\"\n";

            // Plot the data
            gp << "plot '-' with lines linewidth 1 title 'True'\n";
            gp.send1d(std::make_tuple(t[0], acc[0]));

            // Set x-axis limits
            gp << "set xrange [" << (model.t_start - 2.0) << ":" << model.t_final << "]\n";

            // Set x-axis label
            gp << "set xlabel \"Time (sec)\"\n";

            // Set y-axis label
            gp << "set ylabel \"Acceleration $(\\mu m/s^2)$\"\n";

            // Enable grid
            gp << "set grid on\n";

            // Enable legend
            gp << "set key outside\n";

            // Display the plot
            gp << "replot\n";
        }
    }

    // // Helper functions for saving and loading data using pickle
    // void save_object(vector<vector<double> > data, string filename) {
    //     // Save data using pickle
    //     // ...
    // }

    // vector<vector<double> > load_object(string filename) {
    //     // Load data using pickle
    //     vector<vector<double> > data;
    //     // ...
    //     return data;
    // }

};

int main() {
    // Rate State model
    rsf problem(5, 10.0, 1000.0, true);

    // Generate the time series for the RSF model
    problem.generate_time_series();

    return 0;
}
