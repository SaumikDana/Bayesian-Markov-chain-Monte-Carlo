#ifndef FILEIO_H
#define FILEIO_H

#include <vector>

void dumpData(const char* filename, const void* data);
std::vector<char> loadData(const char* filename);

#endif
