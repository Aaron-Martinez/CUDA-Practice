
#ifndef CPP_UTILS_H
#define CPP_UTILS_H

#include <iostream>
#include <cstdlib>
#include <string>
#include <math.h>
#include <chrono>

using std::cout;
using std::endl;
using std::string;

double getMilliseconds(std::chrono::high_resolution_clock::time_point startTime, std::chrono::high_resolution_clock::time_point endTime);
void printMilliseconds(std::chrono::high_resolution_clock::time_point startTime, std::chrono::high_resolution_clock::time_point endTime);
void printMilliseconds(std::chrono::high_resolution_clock::time_point startTime, std::chrono::high_resolution_clock::time_point endTime, string caption);

#endif