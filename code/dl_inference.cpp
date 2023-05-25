#include "/Users/saumikdana/gnuplot-iostream/gnuplot-iostream.h"

#include <iostream>
#include <cmath>
#include <vector>
#include <random>
#include "fileIO.h"
#include "RateStateModel.h"

// #include "MCMC.h"

using namespace std;

int main() {
    int number_slip_values = 5;
    double lowest_slip_value = 10.0;
    double largest_slip_value = 1000.0;

    // Generate a list of slip values
    vector<double> dc_list(number_slip_values);
    double dc_step = (largest_slip_value - lowest_slip_value) / (number_slip_values - 1);
    for (int i = 0; i < number_slip_values; ++i)
        dc_list[i] = lowest_slip_value + i * dc_step;

    vector<double> acc_appended_noise;

    bool plotfigs = true;

    // Create a RateStateModel instance
    RateStateModel model(500, 0.0, 50.0);

    // Set the parameters of the model
    model.setA(0.011);
    model.setB(0.014);
    model.setMuRef(0.6);
    model.setVRef(1);
    model.setK1(1e-7);
    model.setTStart(0.0);
    model.setTFinal(50.0);
    model.setMuTZero(0.6);
    model.setRadiationDamping(true);

    // Loop through each slip value
    for (double dc : dc_list) {

        Gnuplot gp;
        gp << "set xlabel 'Time (sec)'\n";
        gp << "set ylabel 'Acceleration (um/s^2)'\n";
        gp << "set grid\n";

        // Set the dc value in the model
        model.setDc(dc);

        vector<double> t, acc, acc_noise;
        model.evaluate(t, acc, acc_noise);

        // Loop to push elements from acc_noise to acc_appended_noise
        for (const auto& value : acc_noise) {
            acc_appended_noise.push_back(value);
        }

        if (plotfigs) {
            vector<pair<double, double>> data(t.size());
            for (int i = 0; i < t.size(); ++i)
                data[i] = make_pair(t[i], acc[i]);

            string title = "$d_c$=" + to_string(dc) + " um";
            gp << "set title '" + title + "'\n";

            // Calculate y-axis range based on minimum and maximum acceleration values
            double minAcc = *min_element(acc.begin(), acc.end());
            double maxAcc = *max_element(acc.begin(), acc.end());
            double yMin = minAcc - 0.1 * fabs(minAcc);
            double yMax = maxAcc + 0.1 * fabs(maxAcc);
            gp << "set yrange [" << yMin << ":" << yMax << "]\n";

            gp << "plot '-' with lines title 'True'\n";
            gp.send1d(data);  // Send data using send1d
            gp.flush();
        }
    }

    // // Bayesian MCMC starts here ..
    // const char* filename = "data.bin";

    // // Dump the data to a file
    // dumpData(filename, acc_appended_noise);

    // // Load the data from the file
    // vector<double> noisy_acc = loadData(filename);

    // vector<double> qpriors; // Declare and initialize qpriors if needed
    // double qstart; // Declare and initialize qstart if needed
    // int nsamples; // Declare and initialize nsamples if needed

    // for (int index = 0; index < dc_list.size(); ++index) {
    //     int start = index * model.getNumTimesteps();
    //     int end = start + model.getNumTimesteps();
    //     vector<double> noisy_data(noisy_acc.begin() + start, noisy_acc.begin() + end);
    //     cout << "--- Dc is " << dc_list[index] << " ---" << endl;

    //     // Perform MCMC sampling without reduction
    //     MCMC MCMCobj(model, noisy_data, qpriors, qstart, nsamples);
    //     vector<double> qparams = MCMCobj.sample();
    //     // plot_dist(qparams, dc);

    // }
    // // Bayesian MCMC ends here ..

    return 0;
}
