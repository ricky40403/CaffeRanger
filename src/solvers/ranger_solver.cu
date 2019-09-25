#include "caffe/util/math_functions.hpp"


namespace caffe {

template <typename Dtype>
__global__ void RangerUpdate(int N,
    const Dtype* g_gpu_data, Dtype* g_mut_gpu_diff,
    Dtype* m_mut_gpu_data,
    Dtype* v_mut_gpu_data,
    const Dtype* slow_gpu_data,  Dtype* slow_mut_gpu_data, 
    Dtype beta1, Dtype beta2, Dtype eps_hat, Dtype corrected_local_rate,
    Dtype N_sma, const Dtype N_sma_threshhold,
    const int t, const int k_thres, const Dtype alpha
    ) {
  CUDA_KERNEL_LOOP(i, N) {
    float gdiff = g_mut_gpu_diff[i];
    float mi = m_mut_gpu_data[i] = m_mut_gpu_data[i]*beta1 + gdiff*(1-beta1);
    float vi = v_mut_gpu_data[i] = v_mut_gpu_data[i]*beta2 + gdiff*gdiff*(1-beta2);
    
    if (N_sma > N_sma_threshhold){
      g_mut_gpu_diff[i] = corrected_local_rate * mi / (sqrt(vi) + eps_hat);
    }
    else{
      g_mut_gpu_diff[i] = corrected_local_rate * mi;
    }

    
    
    if ((t%k_thres) == 0){
      
      // set slow 
      float slow = slow_mut_gpu_data[i] = alpha  * ( g_gpu_data[i] - slow_gpu_data[i] );

      //p => slow_p
      //p_diff  = p - slow_p 
      g_mut_gpu_diff[i] = g_gpu_data[i] - slow;
      
    }
    


  }
}


template <typename Dtype>
void ranger_update_gpu(int N,
    const Dtype* g_gpu_data, Dtype* g_mut_gpu_diff,
    Dtype* m_mut_gpu_data,
    Dtype* v_mut_gpu_data,
    const Dtype* slow_gpu_data,  Dtype* slow_mut_gpu_data, 
    Dtype beta1, Dtype beta2, Dtype eps_hat, Dtype corrected_local_rate,
    const Dtype N_sma, const Dtype N_sma_threshhold,
    const int t, const int k_thres, const Dtype alpha
    ) {

  
  RangerUpdate<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)
      <<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>( N,
      g_gpu_data, g_mut_gpu_diff,
      m_mut_gpu_data,
      v_mut_gpu_data,
      slow_gpu_data, slow_mut_gpu_data,
      beta1, beta2, eps_hat, corrected_local_rate,
      N_sma, N_sma_threshhold,
      t, k_thres, alpha
      );
  
  CUDA_POST_KERNEL_CHECK;
}
template void ranger_update_gpu<float>(int,
    const float*, float*,
    float*,
    float*,
    const float*, float*,
    float, float, float, float,
    const float, const float,
    const int, const int, const float
    );

template void ranger_update_gpu<double>(int,
    const double*, double*,
    double*,
    double*,
    const double*, double*,
    double, double, double, double,
    const double, const double,
    const int, const int, const double
    );

}  // namespace caffe

