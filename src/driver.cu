#include "linAlgFcns.cuh"
#include "cpp_utils.h"
#include "examples_linAlg.cuh"

int main(void) {
    // addComparison(512);
    // addComparison(256);
    // addComparison(128);
    // addComparison(1);
    matrixMultiplyComparison();
    printf("\ndone\n\n");
    return 0;
}