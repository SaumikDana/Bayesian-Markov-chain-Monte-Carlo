#include <iostream>
#include <fstream>
#include <vector>

using namespace std;

void dumpData(const char* filename, const void* data)
{
    ofstream file(filename, ios::binary);
    if (file)
    {
        const char* charData = static_cast<const char*>(data);
        size_t dataSize = 0;
        while (charData[dataSize])
        {
            ++dataSize;
        }

        file.write(charData, dataSize);
        file.close();
    }
    else
    {
        cerr << "Error opening file: " << filename << endl;
        // Handle the error: display an error message, log the error, or take appropriate action
    }
}

vector<char> loadData(const char* filename)
{
    ifstream file(filename, ios::binary | ios::ate);
    if (file)
    {
        size_t dataSize = static_cast<size_t>(file.tellg());
        vector<char> data(dataSize);

        file.seekg(0, ios::beg);
        file.read(data.data(), dataSize);
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
