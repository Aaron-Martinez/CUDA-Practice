#include "cpp_utils.h"

double getMilliseconds(std::chrono::high_resolution_clock::time_point startTime, std::chrono::high_resolution_clock::time_point endTime) {
    double time = std::chrono::duration_cast<std::chrono::nanoseconds>(endTime - startTime).count();
    time *= 1e-6;
    return time;
}

void printMilliseconds(std::chrono::high_resolution_clock::time_point startTime, std::chrono::high_resolution_clock::time_point endTime) {
    printf("%0.4f milliseconds \n", getMilliseconds(startTime, endTime));
}

void printMilliseconds(std::chrono::high_resolution_clock::time_point startTime, std::chrono::high_resolution_clock::time_point endTime, string caption) {
    printf("%s: %0.4f milliseconds \n", caption.c_str(), getMilliseconds(startTime, endTime));
}