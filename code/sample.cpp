#include <iostream>
#include <vector>
#include "/Users/saumikdana/gnuplot-iostream/gnuplot-iostream.h"

int main() {
    // Generate some sample data for the time series
    std::vector<double> time;
    std::vector<double> values;
    for (double t = 0.0; t < 10.0; t += 0.1) {
        time.push_back(t);
        values.push_back(sin(t));
    }

    // Create a Gnuplot object
    Gnuplot gp;

    // Set the labels for the x-axis and y-axis
    gp << "set xlabel 'Time'\n";
    gp << "set ylabel 'Value'\n";

    // Plot the time series
    gp << "plot '-' with lines title 'Time Series'\n";
    gp.send1d(std::make_tuple(time, values));

    // Wait for a key press to close the plot window
    std::cout << "Press enter to exit.";
    std::cin.get();

    return 0;
}
