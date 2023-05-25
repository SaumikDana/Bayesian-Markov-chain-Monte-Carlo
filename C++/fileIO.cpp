#include <iostream>
#include <fstream>
#include <vector>

using namespace std;

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

