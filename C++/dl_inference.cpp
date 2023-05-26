#include "/Users/saumikdana/gnuplot-iostream/gnuplot-iostream.h"

#include <iostream>
#include <cmath>
#include <vector>
#include <random>
#include "RateStateModel.h"
#include "MCMC.h"

using namespace std;

#include <fstream>

void dumpData(const char* filename, const vector<double>& data)
{
    ofstream file(filename, ios::binary);
    if (file)
    {
        const char* charData = reinterpret_cast<const char*>(data.data());
        size_t dataSize = data.size() * sizeof(double);

        file.write(charData, dataSize);
        file.close();
    }
    else
    {
        cerr << "Error opening file: " << filename << endl;
        // Handle the error: display an error message, log the error, or take appropriate action
    }
}

vector<double> loadData(const char* filename)
{
    ifstream file(filename, ios::binary | ios::ate);
    if (file)
    {
        size_t dataSize = static_cast<size_t>(file.tellg());
        size_t numDoubles = dataSize / sizeof(double);
        vector<double> data(numDoubles);

        file.seekg(0, ios::beg);
        file.read(reinterpret_cast<char*>(data.data()), dataSize);
        file.close();

        return data;
    }
    else
    {
        cerr << "Error opening file: " << filename << endl;
        // Handle the error: display an error message, log the error, or take appropriate action
        return {};  // Return an empty vector to indicate failure
    }
}

bool areVectorsIdentical(const vector<double>& vector1, const vector<double>& vector2) {
    if (vector1.size() != vector2.size()) {
        return false;  // Different sizes, not identical
    }

    for (size_t i = 0; i < vector1.size(); ++i) {
        if (vector1[i] != vector2[i]) {
            return false;  // Different elements at the same index, not identical
        }
    }

    return true;  // Sizes are the same and all elements are identical
}

int main() {
    int number_slip_values = 5;
    double lowest_slip_value = 100.0;
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

    // Bayesian MCMC starts here ..
    const char* filename = "data.bin";

    // Dump the data to a file
    dumpData(filename, acc_appended_noise);

    // Load the data from the file
    vector<double> noisy_acc = loadData(filename);

    // we use std::random_device to obtain a random seed 
    // for the pseudo-random number generator (std::mt19937). 
    // We then define a std::uniform_real_distribution<double> with a range from 0.0 to 10000.0.
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<double> qpriors(0.0, 10000.0);

    double qstart = 10.0;

    int nsamples = 500; // Declare and initialize nsamples if needed

    for (int index = 0; index < dc_list.size(); ++index) {
        int start = index * model.getNumTimesteps();
        int end = start + model.getNumTimesteps();
        vector<double> noisy_data(noisy_acc.begin() + start, noisy_acc.begin() + end);
        cout << "--- Dc is " << dc_list[index] << " ---" << endl;

        // Perform MCMC sampling without reduction
        MCMC MCMCobj(model, noisy_data, qpriors, qstart, nsamples);
        // Perform MCMC sampling without reduction
        Eigen::MatrixXd qparams = MCMCobj.sample();
    }
    // Bayesian MCMC ends here ..

    return 0;
}
