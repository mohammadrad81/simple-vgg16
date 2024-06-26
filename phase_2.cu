#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <omp.h>

#define MALLOC_ERROR "could not allocate memory";
#define CNN_ERROR "error in CNN!";
#define FC_ERROR "error in FC!";

void handle(cudaError_t status, char* message){
    if(status != cudaSuccess){
        printf("message: %s\n", message);
        printf("error string: %s\n", cudaGetErrorString(status));
        exit(-1);
    }
}

float* pad(float* input, )

float rand_float(){
    return (2 * (float)(rand()) / (float)(RAND_MAX)) - 1.0;
}

void fill_array(float* array, int length){
    #pragma omp parallel for
    for(int i = 0; i < length; i++){
        array[i] = rand_float();
    }
}

float* vgg_16(float* host_images, int channels, int width, int height){
    float* dev_images = 0;
    int image_vector_length = channels * width * height * sizeof(float);
    handle(cudaMalloc((void**)&dev_images, image_vector_length, MALLOC_ERROR));
    handle(cudaMemcpy(dev_images, host_images, image_vector_length));


}

int main(){
    srand(42);
    int length = 100000000;
    float* array = (float*) malloc(length * sizeof(float));
    fill_array(array, length);
    return 0;
}