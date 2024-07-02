#include <stdio.h>
#include <stdlib.h>
// #include <cuda_runtime.h>
#include <omp.h>
#include <time.h>

#define MALLOC_ERROR "could not allocate memory"
#define CPY_ERROR "could not copy"
#define FREE_ERROR "could not free cuda memory"
#define PAD_ERROR "error in pading!"
#define POOL_ERROR "error in pooling!"
#define CNN_ERROR "error in CNN!"
#define ZERO_ERROR "error in zero!"
#define FC_ERROR "error in FC!"
#define RELU_ERROR "error in RELU!"
#define ADD_ERROR "error in add"
#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))
#define MIN(a, b) ((a) < (b)? a : b)
#define MAX(a, b) ((a) < (b)? (b) : (a))

void handle(cudaError_t status, char* message, int line){
    if(status != cudaSuccess){
        printf("message: %s\n", message);
        printf("error string: %s\n", cudaGetErrorString(status));
        printf("line: %d\n", line);
        exit(-1);
    }
}

double rand_double(){
    // return (2 * (double)(rand()) / (double)(RAND_MAX)) * 2 - 1;
    return 0.001;
}

void fill_array(double* array, int length){
    for(int i = 0; i < length; i++){
        array[i] = rand_double();
    }
}

double* random_vector(int length){
    double* params = (double*) malloc(length * sizeof(double));
    fill_array(params, length);
    return params;
}

double* copy_to_cuda(double* vector, int length){
    double* dev_vector = 0;
    handle(cudaMalloc((void**)&dev_vector, length * sizeof(double)), MALLOC_ERROR, __LINE__);
    handle(cudaMemcpy(dev_vector, vector, length * sizeof(double), cudaMemcpyHostToDevice), CPY_ERROR, __LINE__);
    return dev_vector;
}

double* copy_from_cuda(double* dev_vector, int length){
    double* vector = (double*)malloc(length * sizeof(double));
    handle(cudaMemcpy(vector, dev_vector, length * sizeof(double), cudaMemcpyDeviceToHost), CPY_ERROR, __LINE__);
    return vector;
}

double* random_vector_cuda(int length){
    double* params = random_vector(length);
    return copy_to_cuda(params, length);
}

__global__ void pad_kernel(double* dev_input,
                           double* dev_output,
                           int width,
                           int height,
                           int channels,
                           int pad_w,
                           int pad_h,
                           int output_width,
                           int output_height){

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int channel = blockIdx.z;
    int output_position = (channel * output_width * output_height) + (row * output_width) + col;
    int input_position = (channel * width * height) + ((row - pad_h) * width) + (col - pad_w);
    double value = 0.0;
    if((row < output_height) && (col < output_height) && (channel < channels)){
        if((row < pad_h) || (row >= height + pad_h) || (col < pad_w) || (col >= width + pad_w)){ // in pad
                value = 0.0;
        }
        else{ // not in pad
            value = dev_input[input_position];
        }
        dev_output[output_position] = value;
    }
}

double* pad_with_cuda(double* dev_input_image,
                     int width,
                     int height,
                     int channels, 
                     int pad_w, 
                     int pad_h){
    
    int output_image_width = (width + 2 * pad_w);
    int output_image_height = (height + 2 * pad_h);
    int output_image_length = output_image_width * output_image_height * channels;
    int output_image_size = output_image_length * sizeof(double);
    double* dev_output_image;
    handle(cudaMalloc((void**)&dev_output_image, output_image_width * output_image_height * channels * sizeof(double)), MALLOC_ERROR, __LINE__);
    dim3 gridDim(CEIL_DIV(output_image_width, 32), CEIL_DIV(output_image_height, 32), channels);
    dim3 blockDim(32, 32, 1);
    pad_kernel<<<gridDim, blockDim>>>(dev_input_image, dev_output_image, width, height, channels, pad_w, pad_h, output_image_width, output_image_height);
    handle(cudaDeviceSynchronize(), PAD_ERROR, __LINE__);
    return dev_output_image;
}

__global__ void max_pool_2D_kernel(double* dev_input_image,
                        double* dev_output_image, 
                        int width,
                        int height, 
                        int channels, 
                        int output_width, 
                        int output_height){
    __shared__ double part[16][64]; // 16 rows, 64 columns
    int row_in_block = threadIdx.y;
    int col_in_block = threadIdx.x;
    int block_row = blockIdx.y;
    int block_col = blockIdx.x;
    int row = block_row * blockDim.y + row_in_block;
    int col = block_col * blockDim.x + col_in_block;
    int channel = blockIdx.z;
    int load_position = (channel * width * height) + (row * width) + col;
    int in_input_image = row < height && col < width;
    double value = 0.0;
    if(in_input_image){
        value = dev_input_image[load_position];
        part[row_in_block][col_in_block] = value;
        __syncthreads();
        int block_row_step = blockDim.y / 2; // 8
        int block_col_step = blockDim.x / 2; // 32
        int store_row = block_row_step * block_row + row_in_block;
        int store_col = block_col_step * block_col + col_in_block;
        if(row_in_block < block_row_step &&
            col_in_block < block_col_step &&
            store_row < output_height &&
            store_col < output_width){

            int store_position = (channel * output_width * output_height) + (store_row * output_width) + store_col;
            int row_in_part = 2 * row_in_block;
            int col_in_part = 2 * col_in_block;
            value = MAX(part[row_in_part][col_in_part], part[row_in_part][col_in_part + 1]);
            value = MAX(value, part[row_in_part + 1][col_in_part]);
            value = MAX(value, part[row_in_part + 1][col_in_part + 1]);
            dev_output_image[store_position] = value;
        }
    }
}

double* max_pool_2D_with_cuda(double* dev_input_image, 
                             int width, 
                             int height, 
                             int channels){

    double* dev_output_image = 0;
    int output_image_width = width / 2;
    int output_image_height = height / 2;
    int output_image_length = output_image_width * output_image_height * channels;
    int output_image_size = output_image_length * sizeof(double);
    handle(cudaMalloc((void**)&dev_output_image, output_image_size), MALLOC_ERROR, __LINE__);
    dim3 gridDim(CEIL_DIV(width, 64), CEIL_DIV(height, 16), channels);
    dim3 blockDim(64, 16, 1);
    max_pool_2D_kernel<<<gridDim, blockDim>>>(dev_input_image, dev_output_image, width, height, channels, output_image_width, output_image_height);
    handle(cudaDeviceSynchronize(), POOL_ERROR, __LINE__);
    handle(cudaFree(dev_input_image), FREE_ERROR, __LINE__);
    return dev_output_image;
}

__global__ void conv2D_kernel(double* dev_input_image, // dev_input_image[channels][heigth][width]
                              int width,
                              int height,
                              int channels,
                              double* kernel, // kernel[output_channels][channels][k_height][k_width]
                              int k_width,
                              int k_height,
                              double* dev_bias, // bias[output_channels]
                              double* dev_output_image,
                              int output_width,
                              int output_height,
                              int output_channels){ 
    __shared__ double part[32][32];
    __shared__ double channel_kernel[10][10];
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
    int thread_output_channel = blockIdx.z;
    double value = 0.0;
    for(int current_input_channel = 0; current_input_channel < channels; current_input_channel++){
        if(row_in_block < k_height && col_in_block < k_width){
            channel_kernel[row_in_block][col_in_block] = kernel[(thread_output_channel * channels * k_width * k_height) + (current_input_channel * k_width * k_height) + (row_in_block * k_width) + col_in_block];
        }
        if(in_input_image){
            int load_position = (current_input_channel * width * height) + (load_position_row * width) + load_position_col;
            part[row_in_block][col_in_block] = dev_input_image[load_position];
        }
        __syncthreads();
        if(in_output_image){
            for(int i = 0; i < k_width; i++){
                for(int j = 0; j < k_height; j++){
                    value += channel_kernel[i][j] * part[row_in_block + i][col_in_block + j];
                }
            }
        }
        __syncthreads();
    }
    if(in_output_image){
        value += dev_bias[thread_output_channel];
        //relu
        value = MAX(value, 0);
        dev_output_image[(thread_output_channel * output_width * output_height) + (load_position_row * output_width) + load_position_col] = value;
    }
}

double* conv2D_with_cuda(double* dev_input_image, 
                        int width, 
                        int height, 
                        int channels, 
                        double* dev_kernel,
                        int k_width, 
                        int k_height,
                        double* dev_bias,
                        int output_channels){

    double* dev_output_image = 0;
    int output_image_width = width - k_width + 1;
    int output_image_height = height - k_height + 1;
    int output_image_length = output_image_width * output_image_height * output_channels;
    //printf("*********************************************\n");
    //printf("in conv2D_with_cuda\n");
    //printf("width: %d\n", width);
    //printf("height: %d\n", width);
    //printf("channels: %d\n", width);
    //printf("output_image_width: %d\n", output_image_width);
    //printf("output_image_height: %d\n", output_image_height);
    //printf("output_image_length: %d * %d * %d = %d\n", output_image_width, output_image_height, channels, output_image_length);
    int output_image_size = output_image_length * sizeof(double);
    handle(cudaMalloc((void**)&dev_output_image, output_image_size), MALLOC_ERROR, __LINE__);
    int step_x = 32 - k_width + 1;
    int step_y = 32 - k_height + 1;
    int grid_dim_x = CEIL_DIV(width, step_x);
    int grid_dim_y = CEIL_DIV(height, step_y);
    dim3 gridDim(grid_dim_x, grid_dim_y, output_channels);
    dim3 blockDim(32, 32, 1);
    conv2D_kernel<<<gridDim, blockDim>>>(dev_input_image,
                                         width,
                                         height,
                                         channels,
                                         dev_kernel,
                                         k_width,
                                         k_height,
                                         dev_bias,
                                         dev_output_image,
                                         output_image_width,
                                         output_image_height,
                                         output_channels);
    handle(cudaDeviceSynchronize(), CNN_ERROR, __LINE__);
    return dev_output_image;

}

__global__ void zero_all_kernel(double* dev_vector, int length){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < length){
        dev_vector[i] = 0.0;
    }
}

void zero_with_cuda(double* dev_vector, int length){
    dim3 gridDim(CEIL_DIV(length, 1024), 1, 1);
    dim3 blockDim(1024, 1, 1);
    zero_all_kernel<<<gridDim, blockDim>>>(dev_vector, length);
    handle(cudaDeviceSynchronize(), ZERO_ERROR, __LINE__);
}

__global__ void add_to_vector_1_kernel(double* dev_vector_1, double* dev_vector_2, int length){
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if(i < length){
        dev_vector_1[i] += dev_vector_2[i];
    }
}

void add_to_vector_1_with_cuda(double* dev_vector_1, double* dev_vector_2, int length){
    dim3 gridDim(CEIL_DIV(length, 1024), 1, 1);
    dim3 blockDim(1024, 1, 1);
    add_to_vector_1_kernel<<<gridDim, blockDim>>>(dev_vector_1, dev_vector_2, length);
    handle(cudaDeviceSynchronize(), ADD_ERROR, __LINE__);
}

__global__ void FC_kernel_improved(double* dev_input, int input_length, double* dev_output, int output_length, double* parameters){
    __shared__ double partial_results[1024];
    int block_row = blockIdx.y;
    int block_col = blockIdx.x;
    int thread_col_in_block = threadIdx.x;
    int thread_col = blockIdx.x * blockDim.x + threadIdx.x;
    double value = 0.0;
    if(thread_col < input_length){
        value = dev_input[thread_col] * parameters[block_row * input_length + thread_col];
    }
    partial_results[thread_col_in_block] = value;
    __syncthreads();
    for(int stride = 512; stride >= 1; stride /= 2){
        if(thread_col_in_block < stride){
            partial_results[thread_col_in_block] += partial_results[thread_col_in_block + stride];
        }
        __syncthreads();
    }
    if(thread_col_in_block == 0){
        atomicAdd(&dev_output[block_row], partial_results[0]);
    }
}

__global__ void RELU_kernel(double* dev_input, int length){
    int thread_point = blockIdx.x * blockDim.x + threadIdx.x;
    if(thread_point < length){
        double value = dev_input[thread_point];
        dev_input[thread_point] = MAX(value, 0);
    }
}

void RELU_with_cuda(double* dev_input, int length){
    dim3 gridDim(CEIL_DIV(length, 1024), 1, 1);
    dim3 blockDim(1024, 1, 1);
    RELU_kernel<<<gridDim, blockDim>>>(dev_input, length);
    handle(cudaDeviceSynchronize(), RELU_ERROR, __LINE__);
}

double* FC_relu(double* dev_input, int input_length, int output_length, double* parameters, double* dev_bias){
    int parameters_height = output_length;
    int parameters_width = input_length;
    double* dev_output = 0;
    handle(cudaMalloc((void**)&dev_output, parameters_height * parameters_width * sizeof(double)), MALLOC_ERROR, __LINE__);
    dim3 gridDim(CEIL_DIV(parameters_width, 1024), parameters_height, 1);
    dim3 blockDim(1024, 1, 1);
    zero_with_cuda(dev_output, output_length);
    FC_kernel_improved<<<gridDim, blockDim>>>(dev_input, input_length, dev_output, output_length, parameters);
    handle(cudaDeviceSynchronize(), FC_ERROR, __LINE__);
    add_to_vector_1_with_cuda(dev_output, dev_bias, output_length);
    RELU_with_cuda(dev_output, output_length);
    return dev_output;
}

double* conv2D_and_pad_with_cuda(double* dev_input_image, int width, int height, int channels, int kernel_width, int kernel_height, int output_channels, int pad_w, int pad_h){
    int image_vector_size = channels * width * height * sizeof(double);
    double* padded_image = pad_with_cuda(dev_input_image, width, height, channels, pad_w, pad_h);
    // printf("padding ended\n");
    width = width + 2 * pad_w;
    height = height + 2 * pad_h;
    double* dev_kernel = random_vector_cuda(output_channels * channels * kernel_width * kernel_height);
    double* dev_bias = random_vector_cuda(output_channels);
    double* dev_output_image = conv2D_with_cuda(padded_image, width, height, channels, dev_kernel, kernel_width, kernel_height, dev_bias, output_channels);
    handle(cudaFree(dev_input_image), FREE_ERROR, __LINE__);
    handle(cudaFree(dev_kernel), FREE_ERROR, __LINE__);
    handle(cudaFree(padded_image), FREE_ERROR, __LINE__);
    handle(cudaFree(dev_bias), FREE_ERROR, __LINE__);
    return dev_output_image;
}

double* FC_relu_with_cuda(double* dev_input, int input_length, int output_length){
    double* dev_parameters = random_vector_cuda(input_length * output_length);
    double* dev_bias = random_vector_cuda(output_length);
    double* result = FC_relu(dev_input, input_length, output_length, dev_parameters, dev_bias);
    handle(cudaFree(dev_input), FREE_ERROR, __LINE__);
    handle(cudaFree(dev_parameters), FREE_ERROR, __LINE__);
    handle(cudaFree(dev_bias), FREE_ERROR, __LINE__);
    return result;
}

void print_duration(char* message, clock_t start, clock_t end){
    printf("%s: \t%lf MS\n", message, 1000.0 * ((double)(end - start)) / ((double)CLOCKS_PER_SEC));
}


void vgg_16(double* h_input_image){
    clock_t total_start = clock();
    int width = 224;
    int height = 224;
    int channels = 3;
    double* dev_input_image = 0;
    // 224 * 224 * 3
    int image_vector_size = channels * width * height * sizeof(double);
    handle(cudaMalloc((void**)&dev_input_image, image_vector_size), MALLOC_ERROR, __LINE__);
    handle(cudaMemcpy(dev_input_image, h_input_image, image_vector_size, cudaMemcpyHostToDevice), CPY_ERROR, __LINE__);
    clock_t move_to_gpu = clock();

    print_duration("move image to global memory", total_start, move_to_gpu);
    

    // ===============  1 =====================

    clock_t start = clock();
    int kernel_width = 3;
    int kernel_height = 3;
    int kernel_channels = 3;
    int output_channels = 64;
    double* dev_output_image = conv2D_and_pad_with_cuda(dev_input_image, width, height, channels, kernel_width, kernel_height, output_channels, 1, 1);
    // //224 * 224 * 64
    clock_t end = clock();
    print_duration("CONV: 224 * 224 * 64", start, end);

    start = clock();
    width = 224;
    height = 224;
    channels = 64;
    kernel_width = 3;
    kernel_height = 3;
    kernel_channels = 64;
    output_channels = 64;
    dev_output_image = conv2D_and_pad_with_cuda(dev_output_image, width, height, channels, kernel_width, kernel_height, output_channels, 1, 1);
    end = clock();
    print_duration("CONV: 224 * 224 * 64", start, end);

    
    // //224 * 224 * 64

    start = clock();
    width = 224;
    height = 224;
    channels = 64;
    dev_output_image = max_pool_2D_with_cuda(dev_output_image, width, height, channels);
    end = clock();
    print_duration("POOLMAX: 112 * 112 * 64", start, end);


    // printf("phase 1 ended\n");

    // ===============  2 =====================

    //112 * 112 * 64
    start = clock();
    width = 112;
    height = 112;
    channels = 64;
    kernel_width = 3;
    kernel_height = 3;
    kernel_channels = 64;
    output_channels = 128;
    dev_output_image = conv2D_and_pad_with_cuda(dev_output_image, width, height, channels, kernel_width, kernel_height, output_channels, 1, 1);
    end = clock();
    print_duration("CONV: 112 * 112 * 128", start, end);


    //112 * 112 * 128
    start = clock();
    width = 112;
    height = 112;
    channels = 128;
    kernel_width = 3;
    kernel_height = 3;
    kernel_channels = 128;
    output_channels = 128;
    dev_output_image = conv2D_and_pad_with_cuda(dev_output_image, width, height, channels, kernel_width, kernel_height, output_channels, 1, 1);
    end = clock();
    print_duration("CONV: 112 * 112 * 128", start, end);


    //112 * 112 * 128
    start = clock();
    width = 112;
    height = 112;
    channels = 128;
    dev_output_image = max_pool_2D_with_cuda(dev_output_image, width, height, channels);
    end = clock();
    print_duration("POOLMAX: 56 * 56 * 128", start, end);

    // printf("phase 2 ended\n");

    // ===============  3 =====================

    //56 * 56 * 128
    start = clock();
    width = 56;
    height = 56;
    channels = 128;
    kernel_width = 3;
    kernel_height = 3;
    kernel_channels = 128;
    output_channels = 256;
    dev_output_image = conv2D_and_pad_with_cuda(dev_output_image, width, height, channels, kernel_width, kernel_height, output_channels, 1, 1);
    end = clock();
    print_duration("CONV: 56 * 56 * 256", start, end);

    //56 * 56 * 256
    start = clock();
    width = 56;
    height = 56;
    channels = 256;
    kernel_width = 3;
    kernel_height = 3;
    kernel_channels = 256;
    output_channels = 256;
    dev_output_image = conv2D_and_pad_with_cuda(dev_output_image, width, height, channels, kernel_width, kernel_height, output_channels, 1, 1);
    end = clock();
    print_duration("CONV: 56 * 56 * 256", start, end);


    //56 * 56 * 256
    start = clock();
    width = 56;
    height = 56;
    channels = 256;
    dev_output_image = max_pool_2D_with_cuda(dev_output_image, width, height, channels);
    end = clock();
    print_duration("POOLMAX: 28 * 28 * 256", start, end);

    // printf("phase 3 ended\n");

    // ===============  4 =====================

    //28 * 28 * 256
    start = clock();
    width = 28;
    height = 28;
    channels = 256;
    kernel_width = 3;
    kernel_height = 3;
    kernel_channels = 256;
    output_channels = 512;
    dev_output_image = conv2D_and_pad_with_cuda(dev_output_image, width, height, channels, kernel_width, kernel_width, output_channels, 1, 1);
    end = clock();
    print_duration("CONV: 28 * 28 * 512", start, end);


    //28 * 28 * 512
    start = clock();
    width = 28;
    height = 28;
    channels = 512;
    kernel_width = 3;
    kernel_height = 3;
    kernel_channels = 512;
    output_channels = 512;
    dev_output_image = conv2D_and_pad_with_cuda(dev_output_image, width, height, channels, kernel_width, kernel_width, output_channels, 1, 1);
    end = clock();
    print_duration("CONV: 28 * 28 * 512", start, end);

    start = clock();
    width = 28;
    height = 28;
    channels = 512;
    dev_output_image = max_pool_2D_with_cuda(dev_output_image, width, height, channels);
    end = clock();
    print_duration("POOLMAX: 14 * 14 * 512", start, end);

    // ===============  5 =====================

    //14 * 14 * 512
    start = clock();
    width = 14;
    height = 14;
    channels = 512;
    kernel_width = 3;
    kernel_height = 3;
    kernel_channels = 512;
    output_channels = 1024;
    dev_output_image = conv2D_and_pad_with_cuda(dev_output_image, width, height, channels, kernel_width, kernel_height, output_channels, 1, 1);
    end = clock();
    print_duration("CONV: 14 * 14 * 1024", start, end);


    //14 * 14 * 1024
    start = clock();
    width = 14;
    height = 14;
    channels = 1024;
    kernel_width = 3;
    kernel_height = 3;
    kernel_channels = 1024;
    output_channels = 1024;
    dev_output_image = conv2D_and_pad_with_cuda(dev_output_image, width, height, channels, kernel_width, kernel_height, output_channels, 1, 1);
    end = clock();
    print_duration("CONV: 14 * 14 * 1024", start, end);

    //14 * 14 * 1024
    start = clock();
    width = 14;
    height = 14;
    channels = 1024;
    dev_output_image = max_pool_2D_with_cuda(dev_output_image, width, height, channels);
    end = clock();
    print_duration("POOLMAX: 7 * 7 * 1024", start, end);

    // ===============  6 =====================

    //7 * 7 * 1024
    start = clock();
    dev_output_image = FC_relu_with_cuda(dev_output_image, 7 * 7 * 1024, 4096);
    end = clock();
    print_duration("FC", start, end);
    // 4096

    start = clock();
    dev_output_image = FC_relu_with_cuda(dev_output_image, 4096, 4096);
    end = clock();
    print_duration("FC", start, end);
    // 4096
    start = clock();
    dev_output_image = FC_relu_with_cuda(dev_output_image, 4096, 1000);
    end = clock();
    print_duration("FC", start, end);
    // 1000

    double* result = (double*)malloc(1000 * sizeof(double));
    handle(cudaMemcpy(result, dev_output_image, 1000 * sizeof(double), cudaMemcpyDeviceToHost), CPY_ERROR, __LINE__);
    // for(int i = 0; i < 1000; i++){
        // printf("result[%d] = %lf\n", i, result[i]);
    // }
    clock_t total_end = clock();
    print_duration("copied to cpu memory, total GPU time: ", total_start, total_end);
}

int main(){
    clock_t start = clock();
    srand(42);
    int width = 224;
    int heigth = 224;
    int channels = 3;
    double* random_image = random_vector(width * heigth * channels);
    vgg_16(random_image);
    clock_t end = clock();
    printf("total execution: %f seconds\n", ((double)(end - start)) / ((double)(CLOCKS_PER_SEC)));

}
