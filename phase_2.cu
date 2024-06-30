#include <stdio.h>
#include <stdlib.h>
// #include <cuda_runtime.h>
#include <omp.h>

#define MALLOC_ERROR "could not allocate memory"
#define CPY_ERROR "could not copy"
#define FREE_ERROR "could not free cuda memory"
#define PAD_ERROR "error in pading!"
#define POOL_ERROR "error in pooling!"
#define CNN_ERROR "error in CNN!"
#define FC_ERROR "error in FC!"
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

__global__ void pad_kernel(float* dev_input,
                           float* dev_output,
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
    float value = 0.0;
    if((row < output_height) && (col < output_height) && (channel < channels)){
        if((row < pad_h) || (row >= height + pad_h) || (col < pad_w) || (col >= width + pad_w)){ // in pad
                value = 0;
        }
        else{ // not in pad
            value = dev_input[input_position];
        }
        dev_output[output_position] = value;
    }
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
    int output_image_length = output_image_width * output_image_height * channels;
    printf("*********************************************\n");
    printf("in pad_with_cuda\n");
    printf("width: %d\n", width);
    printf("height: %d\n", width);
    printf("channels: %d\n", width);
    printf("pad_w: %d\n", pad_w);
    printf("pad_h: %d\n", pad_h);
    printf("output_image_width: %d\n", output_image_width);
    printf("output_image_height: %d\n", output_image_height);
    printf("output_image_length: %d * %d * %d = %d\n", output_image_width, output_image_height, channels, output_image_length);


    int output_image_size = output_image_length * sizeof(float);

    float* dev_output_image;
    // printf("before cuda malloc!\n");
    handle(cudaMalloc((void**)&dev_output_image, output_image_width * output_image_height * channels * sizeof(float)), MALLOC_ERROR, __LINE__);
    // printf("after cuda malloc!\n");
    dim3 gridDim(CEIL_DIV(output_image_width, 32), CEIL_DIV(output_image_height, 32), channels);
    dim3 blockDim(32, 32, 1);
    pad_kernel<<<gridDim, blockDim>>>(dev_input_image, dev_output_image, width, height, channels, pad_w, pad_h, output_image_width, output_image_height);
    handle(cudaDeviceSynchronize(), PAD_ERROR, __LINE__);
    return dev_output_image;
}

__global__ void max_pool_2D_kernel(float* dev_input_image,
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
        int row_in_part = 2 * row_in_block;
        int col_in_part = 2 * col_in_block;
        value = MAX(part[row_in_part][col_in_part], part[row_in_part][col_in_part + 1]);
        value = MAX(value, part[row_in_part + 1][col_in_part]);
        value = MAX(value, part[row_in_part + 1][col_in_part + 1]); 
        dev_output_image[store_position] = value;
        if(value > 0.0){
            printf("max_pooling_vaue: %f\n", value);

        }
    }
}

float* max_pool_2D_with_cuda(float* dev_input_image, 
                             int width, 
                             int height, 
                             int channels){

    float* dev_output_image = 0;
    int output_image_width = width / 2;
    int output_image_height = height / 2;
    int output_image_length = output_image_width * output_image_height * channels;
    printf("*********************************************\n");
    printf("in max_pool_2D_with_cuda\n");
    printf("width: %d\n", width);
    printf("height: %d\n", width);
    printf("channels: %d\n", width);
    printf("output_image_width: %d\n", output_image_width);
    printf("output_image_height: %d\n", output_image_height);
    printf("output_image_length: %d * %d * %d = %d\n", output_image_width, output_image_height, channels, output_image_length);
    int output_image_size = output_image_length * sizeof(float);
    handle(cudaMalloc((void**)&dev_output_image, output_image_size), MALLOC_ERROR, __LINE__);
    dim3 gridDim(CEIL_DIV(width, 64), CEIL_DIV(height, 16), channels);
    dim3 blockDim(64, 16, 1);
    max_pool_2D_kernel<<<gridDim, blockDim>>>(dev_input_image, dev_output_image, width, height, channels, output_image_width, output_image_height);
    handle(cudaDeviceSynchronize(), POOL_ERROR, __LINE__);
    handle(cudaFree(dev_input_image), FREE_ERROR, __LINE__);
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
    int thread_output_channel = blockIdx.z;
    float value = 0.0;
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
        //relu
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
    int output_image_length = output_image_width * output_image_height * output_channels;
    printf("*********************************************\n");
    printf("in conv2D_with_cuda\n");
    printf("width: %d\n", width);
    printf("height: %d\n", width);
    printf("channels: %d\n", width);
    printf("output_image_width: %d\n", output_image_width);
    printf("output_image_height: %d\n", output_image_height);
    printf("output_image_length: %d * %d * %d = %d\n", output_image_width, output_image_height, channels, output_image_length);
    int output_image_size = output_image_length * sizeof(float);
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
                                         dev_output_image,
                                         output_image_width,
                                         output_image_height,
                                         output_channels);
    handle(cudaDeviceSynchronize(), CNN_ERROR, __LINE__);
    return dev_output_image;

}

__global__ void FC_relu_kernel(float* dev_input, int input_length, float* dev_output, int output_length, float* parameters){
    __shared__ float block_value;
    float thread_value = 0.0;
    int block_row = blockIdx.x;
    int thread_col = threadIdx.x;
    int step = blockDim.x;
    if(thread_col == 0){
        block_value = 0;
    }
    __syncthreads();
    for(int i = thread_col; i < input_length; i += step){
        thread_value += dev_input[i] * parameters[block_row * input_length + i];
    }
    atomicAdd(&block_value, thread_value);
    __syncthreads();
    if(thread_col == 0){
        dev_output[block_row] = MAX(block_value, 0);
    }
}

float* FC_relu(float* dev_input, int input_length, int output_length, float* parameters){
    int parameters_height = output_length;
    int parameters_width = input_length;
    float* dev_output = 0;
    handle(cudaMalloc((void**)&dev_output, parameters_height * parameters_width * sizeof(float)), MALLOC_ERROR, __LINE__);
    dim3 gridDim(parameters_height, 1, 1);
    dim3 blockDim(1024, 1, 1);
    FC_relu_kernel<<<gridDim, blockDim>>>(dev_input, input_length, dev_output, output_length, parameters);
    handle(cudaDeviceSynchronize(), FC_ERROR, __LINE__);
    return dev_output;
}

float rand_float(){
    // return (2 * (float)(rand()) / (float)(RAND_MAX)) * 2 - 1;
    return 1.0;
}

void fill_array(float* array, int length){
    for(int i = 0; i < length; i++){
        array[i] = rand_float();
    }
}

float* random_vector(int length){
    float* params = (float*) malloc(length * sizeof(float));
    fill_array(params, length);
    return params;
}

float* copy_to_cuda(float* vector, int length){
    float* dev_vector = 0;
    handle(cudaMalloc((void**)&dev_vector, length * sizeof(float)), MALLOC_ERROR, __LINE__);
    handle(cudaMemcpy(dev_vector, vector, length * sizeof(float), cudaMemcpyHostToDevice), CPY_ERROR, __LINE__);
    return dev_vector;
}

float* copy_from_cuda(float* dev_vector, int length){
    printf("length: %d\n", length);
    float* vector = (float*)malloc(length * sizeof(float));
    printf("after vector malloc!\n");
    handle(cudaMemcpy(vector, dev_vector, length * sizeof(float), cudaMemcpyDeviceToHost), CPY_ERROR, __LINE__);
    printf("after memcpy!\n");
    return vector;
}

float* random_vector_cuda(int length){
    float* params = random_vector(length);
    return copy_to_cuda(params, length);
}

int are_same(float* a, float* b, int length){
    for(int i = 0; i < length; i++){
        if (a[i] != b[i]){
            printf("%f == a[%d] != b[%d] == %f\n", a[i], i, i, b[i]);
            return 0;
        }
    }
    return 1;
}

float* pad_with_cpu(float* input, int channels, int input_width, int input_height, int pad_w, int pad_h){
    int output_width = input_width + pad_w - 1;
    int output_height = input_height + pad_h - 1;
    float* output = (float*) malloc(output_width * output_height * sizeof(float));
    // printf("before parallel region!\n");
    for(int c = 0; c < channels; c++){
        for(int i = 0; i < output_height; i++){
            for(int j = 0; j < output_width; j++){
                // printf("%d\n", j);
                if(i < pad_h || i >= output_height + pad_h || j < pad_w || j >= output_width + pad_w){
                    output[(c * output_width * output_height) + i * output_width + j] = input[(i - pad_h) * output_width + j - pad_w];
                }
                else{
                    output[(c * output_width * output_height) + i * output_width + j] = 0;
                }
            }
        }
    }
    // printf("after parallel region!\n");
    return output;
}

float* conv2D_and_pad_with_cuda(float* dev_input_image, int width, int height, int channels, int kernel_width, int kernel_height, int output_channels, int pad_w, int pad_h){
    int image_vector_size = channels * width * height * sizeof(float);
    float* padded_image = pad_with_cuda(dev_input_image, width, height, channels, pad_w, pad_h);
    handle(cudaFree(dev_input_image), FREE_ERROR, __LINE__);
    width = width + 2 * pad_w;
    height = height + 2 * pad_h;
    float* dev_kernel = random_vector_cuda(output_channels * channels * kernel_width * kernel_height);
    float* dev_output_image = conv2D_with_cuda(padded_image, width, height, channels, dev_kernel, kernel_width, kernel_height, output_channels);
    handle(cudaFree(dev_kernel), FREE_ERROR, __LINE__);
    handle(cudaFree(padded_image), FREE_ERROR, __LINE__);
    float* h_output_image = (float*) malloc(224 * 224 * 64 * sizeof(float));
    handle(cudaMemcpy(h_output_image, dev_output_image, 224 * 224 * 64 * sizeof(float), cudaMemcpyDeviceToHost), CPY_ERROR, __LINE__);
    for(int c = 0; c < output_channels; c++){
        for(int i = 0; i < 224; i++){
            for(int j = 0; j < 224; j++){
                float value = h_output_image[c * 224 * 224 + i * 224 + j];
                // if(value > 0.0){
                    printf("output[%d][%d][%d] = %f\n", c, i, j, value);
                // }
            }
        }
    }
    return NULL;
    return dev_output_image;
}

float* FC_relu_with_cuda(float* dev_input, int input_length, int output_length){
    float* parameters = random_vector_cuda(input_length * output_length);
    return FC_relu(dev_input, input_length, output_length, parameters);
}


void vgg_16(float* h_input_image){
    int width = 224;
    int height = 224;
    int channels = 3;
    float* dev_input_image = 0;
    // 224 * 224 * 3
    int image_vector_size = channels * width * height * sizeof(float);
    handle(cudaMalloc((void**)&dev_input_image, image_vector_size), MALLOC_ERROR, __LINE__);
    handle(cudaMemcpy(dev_input_image, h_input_image, image_vector_size, cudaMemcpyHostToDevice), CPY_ERROR, __LINE__);

    // ===============  1 =====================

    int kernel_width = 3;
    int kernel_height = 3;
    int kernel_channels = 3;
    int output_channels = 64;
    float* dev_output_image = conv2D_and_pad_with_cuda(dev_input_image, width, height, channels, kernel_width, kernel_height, output_channels, 1, 1);

    // //224 * 224 * 64

    // width = 224;
    // height = 224;
    // channels = 64;
    // kernel_width = 3;
    // kernel_height = 3;
    // kernel_channels = 64;
    // output_channels = 64;
    // dev_output_image = conv2D_and_pad_with_cuda(dev_output_image, width, height, channels, kernel_width, kernel_height, 64, 1, 1);
    
    // //224 * 224 * 64

    // width = 224;
    // height = 224;
    // channels = 64;
    // dev_output_image = max_pool_2D_with_cuda(dev_output_image, width, height, channels);

    // printf("phase 1 ended\n");

    // // ===============  2 =====================

    // //112 * 112 * 64
    // width = 112;
    // height = 112;
    // channels = 64;
    // kernel_width = 3;
    // kernel_height = 3;
    // kernel_channels = 64;
    // output_channels = 128;
    // dev_output_image = conv2D_and_pad_with_cuda(dev_output_image, width, height, channels, kernel_width, kernel_height, output_channels, 1, 1);
    
    // //112 * 112 * 128
    // width = 112;
    // height = 112;
    // channels = 128;
    // kernel_width = 3;
    // kernel_height = 3;
    // kernel_channels = 128;
    // output_channels = 128;
    // dev_output_image = conv2D_and_pad_with_cuda(dev_output_image, width, height, channels, kernel_width, kernel_height, output_channels, 1, 1);

    // //112 * 112 * 128

    // width = 112;
    // height = 112;
    // channels = 128;
    // dev_output_image = max_pool_2D_with_cuda(dev_output_image, width, height, channels);

    // printf("phase 2 ended\n");

    // // ===============  3 =====================

    // //56 * 56 * 128
    // width = 56;
    // height = 56;
    // channels = 128;
    // kernel_width = 3;
    // kernel_height = 3;
    // kernel_channels = 128;
    // output_channels = 256;
    // dev_output_image = conv2D_and_pad_with_cuda(dev_output_image, width, height, channels, kernel_width, kernel_height, output_channels, 1, 1);

    // //56 * 56 * 256
    // width = 56;
    // height = 56;
    // channels = 256;
    // kernel_width = 3;
    // kernel_height = 3;
    // kernel_channels = 256;
    // output_channels = 256;
    // dev_output_image = conv2D_and_pad_with_cuda(dev_output_image, width, height, channels, kernel_width, kernel_height, output_channels, 1, 1);

    // //56 * 56 * 256
    // width = 56;
    // height = 56;
    // channels = 256;
    // dev_output_image = max_pool_2D_with_cuda(dev_output_image, width, height, channels);

    // printf("phase 3 ended\n");

    // // ===============  4 =====================

    // //28 * 28 * 256
    // width = 28;
    // height = 28;
    // channels = 256;
    // kernel_width = 3;
    // kernel_height = 3;
    // kernel_channels = 256;
    // output_channels = 512;
    // dev_output_image = conv2D_and_pad_with_cuda(dev_output_image, width, height, channels, kernel_width, kernel_width, channels, 1, 1);

    // //28 * 28 * 512
    // width = 28;
    // height = 28;
    // channels = 512;
    // kernel_width = 3;
    // kernel_height = 3;
    // kernel_channels = 512;
    // output_channels = 512;
    // dev_output_image = conv2D_and_pad_with_cuda(dev_output_image, width, height, channels, kernel_width, kernel_width, channels, 1, 1);

    // width = 28;
    // height = 28;
    // channels = 512;
    // dev_output_image = max_pool_2D_with_cuda(dev_output_image, width, height, channels);

    // printf("phase 4 ended\n");

    // // ===============  5 =====================

    // //14 * 14 * 512

    // width = 14;
    // height = 14;
    // channels = 512;
    // kernel_width = 3;
    // kernel_height = 3;
    // kernel_channels = 512;
    // output_channels = 1024;
    // dev_output_image = conv2D_and_pad_with_cuda(dev_output_image, width, height, channels, kernel_width, kernel_height, output_channels, 1, 1);
    
    // //14 * 14 * 1024
    // width = 14;
    // height = 14;
    // channels = 1024;
    // kernel_width = 3;
    // kernel_height = 3;
    // kernel_channels = 1024;
    // output_channels = 1024;
    // dev_output_image = conv2D_and_pad_with_cuda(dev_output_image, width, height, channels, kernel_width, kernel_height, output_channels, 1, 1);
    
    // //14 * 14 * 1024
    // width = 14;
    // height = 14;
    // channels = 1024;
    // dev_output_image = max_pool_2D_with_cuda(dev_output_image, width, height, channels);

    // printf("phase 5 ended\n");

    // // ===============  6 =====================

    // //7 * 7 * 1024

    // dev_output_image = FC_relu_with_cuda(dev_output_image, 7 * 7 * 1024, 4096);
    
    // // 4096

    // dev_output_image = FC_relu_with_cuda(dev_output_image, 4096, 4096);

    // // 4096

    // dev_output_image = FC_relu_with_cuda(dev_output_image, 4096, 1000);
    
    // // 1000

    // printf("phase 6 ended\n");

    // float* result = (float*)malloc(1000 * sizeof(float));
    // handle(cudaMemcpy(result, dev_output_image, 1000 * sizeof(float), cudaMemcpyDeviceToHost), CPY_ERROR, __LINE__);
    // printf("result:\n\n");
    // for(int i = 0; i < 1000; i++){
    //     printf("%f\n", result[i]);
    // }
}

int main(){
    srand(42);
    int width = 224;
    int heigth = 224;
    int channels = 3;
    float* random_image = random_vector(width * heigth * channels);
    // for(int i = 0; i < width * heigth * channels; i++){
    //     printf("%f\n", random_image[i]);
    // }
    vgg_16(random_image);
    // int length = 100000000;
    // float* array = (float*) malloc(length * sizeof(float));
    // fill_array(array, length);
    // return 0;
    // float* x;
    // cudaMalloc((void**)&x, 1 * sizeof(float));
    // pad_with_cuda(0, 0, 0, 0, 0, 0);
    // float* x = 0;
    // cudaMalloc((void**)&x, 224 * 224 * 3 * sizeof(float));
    // return 0;
}
