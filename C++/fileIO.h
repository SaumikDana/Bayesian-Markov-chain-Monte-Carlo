#ifndef FILEIO_H
#define FILEIO_H

#include <vector>

void dumpData(const char* filename, const std::vector<double>& data);
std::vector<double> loadData(const char* filename);
bool areVectorsIdentical(const std::vector<double>& vector1, const std::vector<double>& vector2);

#endif
