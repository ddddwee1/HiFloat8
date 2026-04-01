#include <cmath>
#include <torch/all.h>
#include <torch/python.h>


using bf16 = at::BFloat16;
using athalf = at::Half;


template <typename T>
__global__ void hif8_quant_cuda_kernel(const T* x_ori, T* res_ori, const int n_total){
    const int thread_idx = threadIdx.x;
    const int block_idx = blockIdx.x;
    const int block_size = blockDim.x;

    const int offset = block_idx * block_size;
    const T* x_offset = x_ori + offset;
    T* res_mem = res_ori + offset;

    if (offset + thread_idx < n_total){
        const float x = (float) x_offset[thread_idx];
        const float absx = fabsf(x);
        if (isnan(x) || isinf(x)){
            res_mem[thread_idx] = x;
        }else if (absx < exp2f(-23.0f)){
            res_mem[thread_idx] = 0;
        }else if (x <= -40960.0f){
            const uint neginf_int = 0xFF800000;
            const float neginf = *(float*) &neginf_int;
            res_mem[thread_idx] = neginf;
        }else if (x >= 40960.0f){
            const uint posinf_int = 0x7F800000;
            const float posinf = *(float*) &posinf_int;
            res_mem[thread_idx] = posinf;
        }else{
            float e = floorf(log2f(absx));
            if (e == -23.0f){
                e = -22.0f;
            }

            const float abs_e = fabsf(e);
            float mant_bits = 0.0f;
            if (abs_e <= 15.0f){
                mant_bits = 1.0f;
            }
            if (abs_e <= 7.0f){
                mant_bits = 2.0f;
            }
            if (abs_e <= 3.0f){
                mant_bits = 3.0f;
            }

            const float scale = exp2f(-e + mant_bits);
            float res = floorf(absx * scale + 0.5f) * exp2f(e - mant_bits);
            res = copysignf(res, x);
            res_mem[thread_idx] = (T) res;
        }
    }
}


void hif8_quant_cuda(torch::Tensor x, torch::Tensor result){
    const int threads = 1024;
    const int blocks = (x.numel() + 1023) / 1024;
    hif8_quant_cuda_kernel<float><<<blocks, threads>>>(x.data_ptr<float>(), result.data_ptr<float>(), x.numel());
}


void hif8_quant_cuda_fp16(torch::Tensor x, torch::Tensor result){
    const int threads = 1024;
    const int blocks = (x.numel() + 1023) / 1024;
    hif8_quant_cuda_kernel<athalf><<<blocks, threads>>>(x.data_ptr<athalf>(), result.data_ptr<athalf>(), x.numel());
}


void hif8_quant_cuda_bf16(torch::Tensor x, torch::Tensor result){
    const int threads = 1024;
    const int blocks = (x.numel() + 1023) / 1024;
    hif8_quant_cuda_kernel<bf16><<<blocks, threads>>>(x.data_ptr<bf16>(), result.data_ptr<bf16>(), x.numel());
}
