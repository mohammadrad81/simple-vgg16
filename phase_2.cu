#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <omp.h>

float rand_float(){
    return (2 * (float)(rand()) / (float)(RAND_MAX)) - 1.0;
}

void fill_array(float* array, int length){
    #pragma omp parallel for
    for(int i = 0; i < length; i++){
        array[i] = rand_float();
    }
}

void vgg_16(float* images, int image_count, int channels, int width, int height){
    
}

int main(){
    srand(42);
    int length = 100000000;
    float* array = (float*) malloc(length * sizeof(float));
    fill_array(array, length);
    return 0;
}