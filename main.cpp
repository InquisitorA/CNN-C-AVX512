#include <immintrin.h>
#include <iostream>
#include <omp.h>
#include <chrono>
#include <ctime>
#include <cmath>

using namespace std;
using namespace chrono;

void convolve(float* input, float* kernel, float* output,
              const int input_height, const int input_width,
              const int kernel_height, const int kernel_width,
              const int output_height, const int output_width)
{
    const int padded_input_height = input_height + kernel_height - 1;
    const int padded_input_width = input_width + kernel_width - 1;
    const int padded_input_size = padded_input_height * padded_input_width;
    const int kernel_size = kernel_height * kernel_width;
    const int output_size = output_height * output_width;

    float* padded_input = new float[padded_input_size];
    float* padded_kernel = new float[padded_input_size];

    for (int i = 0; i < input_height; ++i)
    {
        for (int j = 0; j < input_width; ++j)
        {
            padded_input[(i + kernel_height / 2) * padded_input_width + (j + kernel_width / 2)] = input[i * input_width + j];
        }
    }
    for (int i = 0; i < kernel_height; ++i)
    {
        for (int j = 0; j < kernel_width; ++j)
        {
            padded_kernel[i * kernel_width + j] = kernel[(kernel_height - 1 - i) * kernel_width + (kernel_width - 1 - j)];
        }
    }

    #pragma omp parallel for
    for (int i = 0; i < output_height; ++i)
    {
        for (int j = 0; j < output_width; ++j)
        {
            float sum = 0.0f;

            for (int k = 0; k < kernel_size; ++k)
            {
                const int input_idx = (i + k / kernel_width) * padded_input_width + (j + k % kernel_width);
                const int kernel_idx = k;

                sum += padded_input[input_idx] * padded_kernel[kernel_idx];
            }

            output[i * output_width + j] = sum;
        }
    }

    delete[] padded_input;
    delete[] padded_kernel;
}

void convolve_avx512(float* input, float* kernel, float* output,
                     const int input_height, const int input_width,
                     const int kernel_height, const int kernel_width,
                     const int output_height, const int output_width)
{
    const int padded_input_height = input_height + kernel_height - 1;
    const int padded_input_width = input_width + kernel_width - 1;
    const int padded_input_size = padded_input_height * padded_input_width;
    const int kernel_size = kernel_height * kernel_width;
    const int output_size = output_height * output_width;

    float* padded_input = new float[padded_input_size];
    float* padded_kernel = new float[padded_input_size];

    for (int i = 0; i < input_height; ++i)
    {
        for (int j = 0; j < input_width; ++j)
        {
            padded_input[(i + kernel_height / 2) * padded_input_width + (j + kernel_width / 2)] = input[i * input_width + j];
        }
    }
    for (int i = 0; i < kernel_height; ++i)
    {
        for (int j = 0; j < kernel_width; ++j)
        {
            padded_kernel[i * kernel_width + j] = kernel[(kernel_height - 1 - i) * kernel_width + (kernel_width - 1 - j)];
        }
    }
    __m512 input_vec, kernel_vec, output_vec, sum_vec;
    __m512i index_vec;
    const int vector_size = 16;

    #pragma omp parallel for private(input_vec, kernel_vec, output_vec, sum_vec, index_vec)
    for (int i = 0; i < output_height; ++i)
    {
        for (int j = 0; j < output_width; j += vector_size)
        {
            sum_vec = _mm512_setzero_ps();

            for (int k = 0; k < kernel_size; ++k)
            {
                input_vec = _mm512_loadu_ps(padded_input + (i + k / kernel_width) * padded_input_width + (j + k % kernel_width));
                kernel_vec = _mm512_set1_ps(padded_kernel[k]);

                output_vec = _mm512_fmadd_ps(input_vec, kernel_vec, sum_vec);

                sum_vec = output_vec;
            }
            _mm512_storeu_ps(output + i * output_width + j, sum_vec);
        }
    }

    delete[] padded_input;
    delete[] padded_kernel;
}

void relu(float* input, int size) {
    const float zero = 0.0f;
    #pragma omp parallel for
    for (int i = 0; i < size; ++i) {
        input[i] = input[i] > zero ? input[i] : zero;
    }
}

void relu_avx512(float* input, int size) {
    __m512 zero = _mm512_setzero_ps();
    #pragma omp parallel for
    for (int i = 0; i < size; i += 16) {
        __m512 x = _mm512_loadu_ps(input + i);
        __mmask16 mask = _mm512_cmp_ps_mask(x, zero, _CMP_GT_OS);
        __m512 result = _mm512_mask_blend_ps(mask, zero, x);
        _mm512_storeu_ps(input + i, result); 
    }
}

void max_pool(float* input, float* output,
              const int input_height, const int input_width,
              const int pool_size, const int stride,
              const int output_height, const int output_width)
{
    const int input_size = input_height * input_width;
    const int output_size = output_height * output_width;

    const int vector_size = 16;

    #pragma omp parallel for
    for (int i = 0; i < output_height; ++i)
    {
        for (int j = 0; j < output_width; ++j)
        {
            float max_val = -std::numeric_limits<float>::infinity();
            int max_index = 0;

            for (int k = 0; k < pool_size; ++k)
            {
                for (int l = 0; l < pool_size; ++l)
                {
                    int row = i * stride + k;
                    int col = j * stride + l;

                    if (row < input_height && col < input_width)
                    {
                        float input_val = input[row * input_width + col];

                        if (input_val > max_val)
                        {
                            max_val = input_val;
                            max_index = row * input_width + col;
                        }
                    }
                }
            }

            output[i * output_width + j] = max_val;
            output[output_size + i * output_width + j] = static_cast<float>(max_index);
        }
    }
}

void max_pool_avx512(float* input, float* output,
                     const int input_height, const int input_width,
                     const int pool_size, const int stride,
                     const int output_height, const int output_width)
{
    const int input_size = input_height * input_width;
    const int output_size = output_height * output_width;

    __m512 max_vec;
    __m512i index_vec;
    const int vector_size = 16;

    #pragma omp parallel for private(max_vec, index_vec)
    for (int i = 0; i < output_height; ++i)
    {
        for (int j = 0; j < output_width; j += vector_size)
        {
            max_vec = _mm512_set1_ps(-std::numeric_limits<float>::infinity());
            index_vec = _mm512_setzero_si512();

            for (int k = 0; k < pool_size; ++k)
            {
                for (int l = 0; l < pool_size; ++l)
                {
                    int row = i * stride + k;
                    int col = j * stride + l;

                    if (row < input_height && col < input_width)
                    {
                        __m512 input_vec = _mm512_loadu_ps(input + row * input_width + col);
                        __mmask16 mask = _mm512_cmp_ps_mask(input_vec, max_vec, _CMP_GT_OQ);
                        max_vec = _mm512_mask_blend_ps(mask, max_vec, input_vec);

                        __m512i current_index_vec = _mm512_set1_epi32(row * input_width + col);
                        index_vec = _mm512_mask_blend_epi32(mask, index_vec, current_index_vec);
                    }
                }
            }
            for (int k = vector_size / 2; k > 0; k /= 2)
            {
                max_vec = _mm512_max_ps(max_vec, _mm512_permutexvar_ps(_mm512_set1_epi32(2*k-1), max_vec));
            }
            _mm512_storeu_ps(output + i * output_width + j, max_vec);
            _mm512_storeu_si512((__m512i*)(output + output_size + i * output_width + j), index_vec);
        }
    }
}

void fully_connected_layer(float* input, float* output,
                           const int input_size, const int output_size,
                           float* weights, float* biases)
{
    const int vector_size = 16;
    #pragma omp parallel for
    for (int i = 0; i < output_size; ++i)
    {
        float sum = 0.0f;

        for (int j = 0; j < input_size; j += vector_size)
        {
            for (int k = 0; k < vector_size; ++k)
            {
                float input_val = input[j + k];
                float weight_val = weights[i * input_size + j + k];
                sum += input_val * weight_val;
            }
        }

        sum += biases[i];

        output[i] = sum > 0.0f ? sum : 0.0f;
    }
}

void fully_connected_layer_avx512(float* input, float* output,
const int input_size, const int output_size,
float* weights, float* biases)
{
    const int vector_size = 16;
    #pragma omp parallel for
    for (int i = 0; i < output_size; ++i)
    {
        __m512 sum_vec = _mm512_setzero_ps();

        for (int j = 0; j < input_size; j += vector_size)
        {
            __m512 input_vec = _mm512_loadu_ps(input + j);
            __m512 weight_vec = _mm512_loadu_ps(weights + i * input_size + j);
            sum_vec = _mm512_fmadd_ps(input_vec, weight_vec, sum_vec);
        }

        __m512 bias_vec = _mm512_set1_ps(biases[i]);
        sum_vec = _mm512_add_ps(sum_vec, bias_vec);

        __mmask16 mask = _mm512_cmp_ps_mask(sum_vec, _mm512_setzero_ps(), _CMP_GT_OQ);
        sum_vec = _mm512_mask_blend_ps(mask, _mm512_setzero_ps(), sum_vec);

        for (int j = vector_size / 2; j > 0; j /= 2)
        {
            sum_vec = _mm512_add_ps(sum_vec, _mm512_permutexvar_ps(_mm512_set1_epi32(2*j-1), sum_vec));
        }
        output[i] = _mm512_reduce_add_ps(sum_vec);
    }
}

int main(int argc, char* argv[])
{
    string dim = argv[1];
    int n = stoi(dim.substr(dim.find("=")+1));
    
    const int input_height = n;
    const int input_width = n;
    const int kernel_height = 3;
    const int kernel_width = 3;

    const int pool_size = 2;
    const int stride = 2;

    float input[input_height * input_width];
    float kernel[kernel_height * kernel_width];
    for(int i = 0; i < input_height * input_width; i++){
        input[i] = i;
    }
    for(int j = 0; j < kernel_height * kernel_width; j++){
  	    kernel[j] = j;
    }

    float input_512[input_height * input_width];
    float kernel_512[kernel_height * kernel_width];
    for(int i = 0; i < input_height * input_width; i++){
        input_512[i] = i;
    }
    for(int j = 0; j < kernel_height * kernel_width; j++){
  	    kernel_512[j] = j;
    }

    const int output_height = input_height - kernel_height + 1;
    const int output_width = input_width - kernel_width + 1;
    float output[output_height * output_width];
    float output_512[output_height * output_width];

    const int new_output_height = (output_height - pool_size) / stride + 1;
    const int new_output_width = (output_width - pool_size) / stride + 1;
    float new_output[new_output_height * new_output_width];
    float new_output_512[new_output_height * new_output_width];

    const int final_output_dim = 10;

    float final_output[final_output_dim];
    float final_output_512[final_output_dim];

    float biases[final_output_dim] = {0.1f};
    float weights[new_output_height * new_output_width * final_output_dim] = {0.5f};    

    auto start_0 = high_resolution_clock::now();
    for(int i = 0; i < 1000; i++){
        convolve(input_256, kernel_256, output_256, input_height, input_width,
                kernel_height, kernel_width, output_height, output_width);
        relu(output_256, output_height * output_width);
        max_pool(output_256, new_output_256,
                        output_height, output_width,
                        pool_size, stride,
                        new_output_height, new_output_width);
        fully_connected_layer(new_output_256, final_output_256,
                                    new_output_height * new_output_width, final_output_dim,
                                    weights, biases);
    }
    auto stop_0 = high_resolution_clock::now();
    auto duration_0 = duration_cast<milliseconds>(stop_0 - start_0);

    auto start_1 = high_resolution_clock::now();
    for(int i = 0; i < 1000; i++){
        convolve_avx512(input_512, kernel_512, output_512, input_height, input_width,
                kernel_height, kernel_width, output_height, output_width);
        relu_avx512(output_512, output_height * output_width);
        max_pool_avx512(output_512, new_output_512,
                        output_height, output_width,
                        pool_size, stride,
                        new_output_height, new_output_width);
        fully_connected_layer_avx512(new_output_512, final_output_512,
                                    new_output_height * new_output_width, final_output_dim,
                                    weights, biases);
    }
    auto stop_1 = high_resolution_clock::now();
    auto duration_1 = duration_cast<milliseconds>(stop_1 - start_1);

    cout << "CNN program execution took " << duration_0.count() << "ms" << endl;
    cout << "CNN program execution accelerated using avx512 took " << duration_1.count() << "ms" << endl;

    return 0;
}
