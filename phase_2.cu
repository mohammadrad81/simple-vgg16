#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <omp.h>

#define MALLOC_ERROR "could not allocate memory"
#define PAD_ERROR "error in pading!"
#define POOL_ERROR "error in pooling!"
#define CNN_ERROR "error in CNN!"
#define FC_ERROR "error in FC!"
#define CEIL_DIV(a, b) ((a + b - 1) / b)
#define MIN(a, b) (a < b? a : b)

void handle(cudaError_t status, char* message){
    if(status != cudaSuccess){
        printf("message: %s\n", message);
        printf("error string: %s\n", cudaGetErrorString(status));
        exit(-1);
    }
}

__global__ void pad_kernel(float* dev_input,
                           float* dev_output,
                           int width,
                           int height,
                           int channels,
                           int pad_w,
                           int pad_h){

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int channel = blockIdx.z;
    int position = (channel * width * height) + (row * width) + col;
    float value = 0.0;
    if((row < height + 2 * pad_h) && (col < width + 2 * pad_w) && (channel < channels)){
        if((row < pad_h) || (row >= height + pad_h) || (col < pad_w) || (col >= width + pad_w)){ // in pad
                value = 0;
        }
        else{ // not in pad
            value = dev_output[position - ((pad_h * width) + pad_w)];
        }
    }
    dev_output[position] = value;
}

float* pad_with_cuda(float* dev_input_image,
                     int width,
                     int height,
                     int channels, 
                     int pad_w, 
                     int pad_h){

    int input_image_size = width * height * channels;
    int output_image_width = (width + 2 * pad_w);
    int output_image_height = (height + 2 * pad_h);
    int output_image_size = output_image_width * output_image_height * channels * sizeof(float);
    float* dev_output_image = 0;
    handle(cudaMalloc((void**)&dev_output_image, output_image_size), MALLOC_ERROR);
    dim3 gridDim(CEIL_DIV(width, 32), CEIL_DIV(height, 32), channels);
    dim3 blockDim(32, 32, 1);
    pad_kernel<<<gridDim, blockDim>>>(dev_input_image, dev_output_image, width, height, channels, pad_w, pad_h);
    handle(cudaDeviceSynchronize(), PAD_ERROR);
    return dev_output_image;
}

void max_pool_2D_kernel(float* dev_input_image,
                        float* dev_output_image, 
                        int width,
                        int height, 
                        int channels, 
                        int output_width, 
                        int output_height){

    __shared__ float part[64][16];
    int row_in_block = threadIdx.y;
    int col_in_block = threadIdx.x;
    int block_row = blockIdx.y;
    int block_col = blockIdx.x;
    int row = block_row * blockDim.y + row_in_block;
    int col = block_col * blockDim.x + col_in_block;
    int channel = blockIdx.z;
    int load_position = (channel * width * height) + (row * width) + col;
    int in_input_image = row < height && col < width;
    float value = 0.0;
    if(in_input_image){
        value = dev_input_image[load_position];
    }
    part[row_in_block][col_in_block] = value;
    __syncthreads();
    int block_row_step = blockDim.y / 2;
    int block_col_step = blockDim.x / 2;
    int store_row = block_row_step * block_row + row_in_block;
    int store_col = block_col_step * block_col + col_in_block;
    if(row_in_block < block_row_step &&
       col_in_block < block_col_step &&
       store_row < output_height &&
       store_col < output_width){
        int store_position = (channel * output_width * output_height) + (store_row * output_width) + store_col;
        int row_in_part = 2 * row;
        int col_in_part = 2 * col;
        value = MIN(part[row_in_part][col_in_part], part[row_in_part][col_in_part + 1]);
        value = MIN(value, part[row_in_part + 1][col_in_part]);
        value = MIN(value, part[row_in_part + 1][col_in_part + 1]); 
        dev_output_image[store_position] = value;
    }
}

float* max_pool_2D_with_cuda(float* dev_input_image, 
                             int width, 
                             int height, 
                             int channels){

    float* dev_output_image = 0;
    int output_image_width = width / 2;
    int output_image_height = height / 2;
    int output_image_size = output_image_width * output_image_height * channels * sizeof(float);
    handle(cudaMalloc((void**)&dev_output_image, output_image_size), MALLOC_ERROR);
    dim3 gridDim(CEIL_DIV(width, 64), CEIL_DIV(height, 16), channels);
    dim3 blockDim(64, 16, 1);
    max_pool_2D_kernel<<<gridDim, blockDim>>>(dev_input_image, dev_output_image, width, height, channels, output_image_width, output_image_height);
    handle(cudaDeviceSynchronize(), POOL_ERROR);
    return dev_output_image;
}

__global__ void conv2D_kernel(float* dev_input_image, // dev_input_image[channels][heigth][width]
                              int width,
                              int height,
                              int channels,
                              float* kernel, // kernel[output_channels][channels][k_height][k_width]
                              int k_width,
                              int k_height,
                              float* dev_output_image,
                              int output_width,
                              int output_height,
                              int output_channels){ 
    __shared__ float part[32][32];
    __shared__ float channel_kernel[10][10];
    int row_in_block = threadIdx.y;
    int col_in_block = threadIdx.x;
    int block_row = blockIdx.y;
    int block_col = blockIdx.x;
    int block_row_step = blockDim.y - k_height + 1;
    int block_col_step = blockDim.x - k_width + 1;
    int load_position_row = block_row * block_row_step + row_in_block;
    int load_position_col = block_col * block_col_step + col_in_block;
    int in_block_load_margin = row_in_block >= block_row_step || col_in_block >= block_col_step;
    int in_input_image = load_position_row < height && load_position_col < width;
    int in_output_image = (!in_block_load_margin) && (load_position_row < (output_height)) && (load_position_col < (output_width));
    int thread_output_channel = blockDim.z;
    float value = 0.0;
    for(int current_input_channel = 0; current_input_channel < channels; current_input_channel++){
        if(row_in_block < k_height && col_in_block < k_width){
            channel_kernel[row_in_block][col_in_block] = kernel[(thread_output_channel * channels * k_width * k_height) + (current_input_channel * k_width * k_height) + (row_in_block * k_width) + col_in_block];
        }   
        int load_position = (current_input_channel * width * height) + (load_position_row * width) + load_position_col;
        part[row_in_block][col_in_block] = dev_input_image[load_position];
        __syncthreads();
        if(in_output_image){
            for(int i = 0; i < k_width; i++){
                for(int j = 0; j < k_height; j++){
                    value += channel_kernel[i][j] * part[row_in_block + i][col_in_block + j];
                }
            }
        }
    }
    if(in_output_image){
        dev_output_image[(thread_output_channel * output_width * output_height) + (load_position_row * output_width) + load_position_col] = value;
    }
}

float* conv2D_with_cuda(float* dev_input_image, 
                        int width, 
                        int height, 
                        int channels, 
                        float* dev_kernel,
                        int k_width, 
                        int k_height, 
                        int output_channels){

    float* dev_output_image = 0;
    int output_image_width = width - k_width + 1;
    int output_image_height = height - k_height + 1;
    int output_image_size = output_image_width * output_image_height * output_channels * sizeof(float);
    handle(cudaMalloc((void**)dev_output_image, output_image_size), MALLOC_ERROR);
    dim3 gridDim(CEIL_DIV(width, 32 - k_width + 1), CEIL_DIV(height, 32 - k_height + 1), output_channels);
    dim3 blockDim(32, 32, 1);
    conv2D_kernel<<<gridDim, blockDim>>>(dev_input_image,
                                         width,
                                         height,
                                         channels,
                                         dev_kernel,
                                         k_width,
                                         k_height,
                                         dev_output_image,
                                         output_image_width,
                                         output_image_height,
                                         output_channels);
    handle(cudaDeviceSynchronize(), CNN_ERROR);
    return dev_output_image;

}

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
    int image_vector_size = channels * width * height * sizeof(float);
    handle(cudaMalloc((void**)&dev_images, image_vector_size), MALLOC_ERROR);


}

int main(){
    srand(42);
    int length = 100000000;
    float* array = (float*) malloc(length * sizeof(float));
    fill_array(array, length);
    return 0;
}