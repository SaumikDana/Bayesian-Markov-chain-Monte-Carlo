#include "/Users/saumikdana/gnuplot-iostream/gnuplot-iostream.h"
#include "/opt/homebrew/Cellar/gsl/2.7.1/include/gsl/gsl_odeiv2.h"
#include "/opt/homebrew/Cellar/gsl/2.7.1/include/gsl/gsl_errno.h"

#include <iostream>
#include <cmath>
#include <vector>
#include <random>

using namespace std;

class RateStateModel {
public:
    RateStateModel(int number_time_steps = 500, double start_time = 0.0, double end_time = 50.0)
        : a(0.011), b(0.014), mu_ref(0.6), V_ref(1), k1(1e-7), 
        t_start(start_time), t_final(end_time), num_tsteps(number_time_steps), 
        delta_t((end_time - start_time) / number_time_steps),
        mu_t_zero(0.6), Dc(0.0), RadiationDamping(true) {}

    void setA(double value) {
        a = value;
    }

    void setB(double value) {
        b = value;
    }

    void setMuRef(double value) {
        mu_ref = value;
    }

    void setVRef(double value) {
        V_ref = value;
    }

    void setK1(double value) {
        k1 = value;
    }

    void setTStart(double value) {
        t_start = value;
    }

    void setTFinal(double value) {
        t_final = value;
    }

    void setMuTZero(double value) {
        mu_t_zero = value;
    }

    void setDc(double value) {
        Dc = value;
    }

    void setRadiationDamping(bool value) {
        RadiationDamping = value;
    }

    static int friction(double t, const double y[], double dydt[], void* params) {
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
    
        // effective spring stiffness
        double kprime = 1e-2 * 10 / dc;

        // loading
        double a1 = 20;
        double a2 = 10;
        double V_l = V_ref * (1 + exp(-t/a1) * sin(a2*t));

        // compute v
        double temp = 1 / a * (y[0] - mu_ref - b * log(V_ref * y[1] / dc));
        double v = V_ref * exp(temp);

        // time derivative of theta
        dydt[1] = 1. - v * y[1] / dc;

        // time derivative of mu
        dydt[0] = kprime * V_l - kprime * v;

        // time derivative of velocity
        dydt[2] = v / a * (dydt[0] - b / y[1] * dydt[1]);

        // add radiation damping term if specified
        if (RadiationDamping) {
            // time derivative of mu with radiation damping
            dydt[0] = dydt[0] - k1 * dydt[2];
            // time derivative of velocity with radiation damping
            dydt[2] = v / a * (dydt[0] - b / y[1] * dydt[1]);
        }

        return GSL_SUCCESS;
    }

    void evaluate(vector<double>& t, vector<double>& acc, vector<double>& acc_noise) {
        vector<double> mu(num_tsteps), theta(num_tsteps), velocity(num_tsteps);
        t.resize(num_tsteps);
        acc.resize(num_tsteps);
        acc_noise.resize(num_tsteps);

        t[0] = t_start;
        mu[0] = mu_ref;
        theta[0] = Dc / V_ref;
        velocity[0] = V_ref;
        acc[0] = 0.0;

        random_device rd;
        mt19937 gen(rd());
        normal_distribution<> dist(0.0, 1.0);

        // Create GSL workspace and ODE system
        const gsl_odeiv2_step_type* step_type = gsl_odeiv2_step_rkf45;
        gsl_odeiv2_system sys = {friction, nullptr, 3, &V_ref};

        gsl_odeiv2_driver* driver = gsl_odeiv2_driver_alloc_y_new(&sys, step_type, 1e-6, 1e-10, 0.0);

        double t_current = t_start;
        double y[3] = {mu_ref, Dc / V_ref, V_ref};
        double h = delta_t;

        for (int k = 1; k < num_tsteps; ++k) {
            double t_next = t_current + h;

            int status = gsl_odeiv2_driver_apply(driver, &t_current, t_next, y);

            if (status != GSL_SUCCESS) {
                // Handle integration error
                cout << "Integration error occurred at time: " << t_current << endl;
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

private:
    double a;
    double b;
    double mu_ref;
    double V_ref;
    double k1;
    double t_start;
    double t_final;
    int num_tsteps;
    double delta_t;
    double mu_t_zero;
    double Dc;
    bool RadiationDamping;
};

int main() {
    int number_slip_values = 5;
    double lowest_slip_value = 10.0;
    double largest_slip_value = 1000.0;

    vector<double> dc_list(number_slip_values);
    double dc_step = (largest_slip_value - lowest_slip_value) / (number_slip_values - 1);
    for (int i = 0; i < number_slip_values; ++i)
        dc_list[i] = lowest_slip_value + i * dc_step;

    bool plotfigs = true;

    Gnuplot gp;
    gp << "set xlabel 'Time (sec)'\n";
    gp << "set ylabel 'Acceleration (um/s^2)'\n";
    gp << "set grid\n";

    for (double dc : dc_list) {

        RateStateModel model(500, 0.0, 50.0);
        model.setA(0.011);
        model.setB(0.014);
        model.setMuRef(0.6);
        model.setVRef(1);
        model.setK1(1e-7);
        model.setTStart(0.0);
        model.setTFinal(50.0);
        model.setMuTZero(0.6);
        model.setRadiationDamping(true);

        model.setDc(dc);

        vector<double> t, acc, acc_noise;
        model.evaluate(t, acc, acc_noise);

        if (plotfigs) {
            vector<pair<double, double>> data(t.size());
            for (int i = 0; i < t.size(); ++i)
                data[i] = make_pair(t[i], acc[i]);

            string title = "$d_c$=" + to_string(dc) + " um";
            gp << "set title '" + title + "'\n";
            gp << "set yrange [-1:1]\n";  // Set y-axis range
            gp << "plot '-' with lines title 'True'\n";
            gp.send1d(data);  // Use send1d instead of send
            gp.flush();

            gp << "unset title\n";
            gp << "reset\n";
        }

    }

    return 0;
}
