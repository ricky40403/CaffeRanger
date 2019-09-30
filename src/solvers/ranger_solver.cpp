#include <vector>

#include "caffe/sgd_solvers.hpp"

namespace caffe {

template <typename Dtype>
void RangerSolver<Dtype>::RangerPreSolve() {  
  
  const vector<Blob<Dtype>*>& net_params = this->net_->learnable_params();
  for (int i = 0; i < net_params.size(); ++i) {
    const vector<int>& shape = net_params[i]->shape();
    this->history_.push_back(
            shared_ptr<Blob<Dtype> >(new Blob<Dtype>(shape)));
  }
}

#ifndef CPU_ONLY  
template <typename Dtype>
void ranger_update_gpu(int N,
    const Dtype* g_gpu_data, Dtype* g_mut_gpu_diff,
    Dtype* m_mut_gpu_data,
    Dtype* v_mut_gpu_data,
    const Dtype* slow_gpu_data,  Dtype* slow_mut_gpu_data, 
    Dtype beta1, Dtype beta2, Dtype eps_hat, Dtype corrected_local_rate,
    const Dtype N_sma, const Dtype N_sma_threshold,
    const int t, const int k_thres, const Dtype alpha,
    const bool use_lookahead
    );
    
#endif

template <typename Dtype>
void RangerSolver<Dtype>::ComputeUpdateValue(int param_id, Dtype rate) {
  const vector<Blob<Dtype>*>& net_params = this->net_->learnable_params();
  const vector<float>& net_params_lr = this->net_->params_lr();
  Dtype local_rate = rate * net_params_lr[param_id];
  const Dtype beta1 = this->param_.momentum();
  const Dtype beta2 = this->param_.momentum2();
  const Dtype alpha = this->param_.ranger_alpha();
  // Dtype => will be float and % can not use 
  const int k_thres = this->param_.ranger_k_thres();
  const Dtype N_sma_threshold = this->param_.ranger_n_sma_threshold();
  // has not decide put where
  const bool use_radam = this->param_.ranger_use_radam();
  const bool use_lookahead = this->param_.ranger_use_lookahead();
  //const Dtype alpha = 0.5;
  // const int k_thres = 5;
  // const Dtype N_sma_threshold = 5;
    
  // we create aliases for convenience
  size_t update_history_offset = net_params.size();
  Blob<Dtype>* val_m = this->history_[param_id].get();
  Blob<Dtype>* val_v = this->history_[param_id + update_history_offset].get();
  Blob<Dtype>* val_t = this->temp_[param_id].get();
  // use update_ to store slow_buffer
  // should use in other script when using ranger  
  Blob<Dtype>* val_slow = this->update_[param_id].get();

  const int t = this->iter_ + 1;
  // const Dtype correction = std::sqrt(Dtype(1) - pow(beta2, t)) /
  //     (Dtype(1.) - pow(beta1, t));
  const int N = net_params[param_id]->count();
  const Dtype eps_hat = this->param_.delta();
  
  
  const Dtype beta1_t = pow(beta1, t); 
  const Dtype beta2_t = pow(beta2, t); 
  const Dtype N_sma_max = (Dtype(2) / (Dtype(1) - beta2)) - Dtype(1);
  const Dtype N_sma = N_sma_max - (2 * t * beta2_t) / (1 - beta2_t);

  Dtype correction = Dtype(1.) / (Dtype(1.) -  beta1_t);
  if (N_sma > N_sma_threshold){
      // (r_t)
      Dtype tmp = (Dtype(1) - beta2_t) * ((N_sma - Dtype(4)) * (N_sma - Dtype(2)) * N_sma_max) / ((N_sma_max - Dtype(4)) * (N_sma_max - Dtype(2)) * N_sma);
      tmp = pow(tmp, 0.5) ;
      // correction 
      correction = correction * tmp * std::sqrt(Dtype(1) - pow(beta2, t));      
  }


  switch (Caffe::mode()) {
    case Caffe::CPU: {
    
    // update m <- \beta_1 m_{t-1} + (1-\beta_1)g_t    
    caffe_cpu_axpby(N, Dtype(1)-beta1,
        net_params[param_id]->cpu_diff(), beta1,
        val_m->mutable_cpu_data());  

    // update v <- \beta_2 m_{t-1} + (1-\beta_2)g_t^2
    caffe_mul(N,
        net_params[param_id]->cpu_diff(),
        net_params[param_id]->cpu_diff(),
    val_t->mutable_cpu_data());
    caffe_cpu_axpby(N, Dtype(1)-beta2,
        val_t->cpu_data(), beta2,
        val_v->mutable_cpu_data());   
    
    
    
    // val_m = val_m / (sqrt(val_v)+eplison)
    if (N_sma > N_sma_threshold){
      //  v ^1/2
      caffe_powx(N,
          val_v->cpu_data(), Dtype(0.5),
          val_t->mutable_cpu_data());
      // v^1/2 + eplison
      caffe_add_scalar(N, eps_hat, val_t->mutable_cpu_data());
      // val_m / (sqrt(val_v)+eplison)
      caffe_div(N,
        val_m->cpu_data(),
        val_t->cpu_data(),
        val_t->mutable_cpu_data()
      );
    }
    else{
      // copy m to v
      caffe_copy(N, val_m->cpu_data(), val_t->mutable_cpu_data());
    }
    // param = val_m' * local_rate * correction
    // (N_sma > 5) => local_rate * (val_m)/(sqrt(val_v)+eplison) * (r_t * sqrt(1-beta2_t) / (1-beta1_t))
    //               = alpha * (m^) * r_t / (v^)
    // (N_sma <= 5) => local_rate * (val_m) * (1 / (1-beta1_t))
    //               = alpha * (m^)
    caffe_cpu_scale(N, local_rate*correction,
        val_m->cpu_data(),
        net_params[param_id]->mutable_cpu_diff());

    // look ahead
    if (use_lookahead && ((t % k_thres) == 0)){
      // slow = alpha * (p - slow_p) = alpha * p - alpha * slow_p
      caffe_cpu_axpby(N,
        Dtype(alpha), net_params[param_id]->cpu_data(),
        Dtype(alpha), val_slow->mutable_cpu_data()        
      );
      // p => slow_p
      // p_diff  = p - slow_p 
      caffe_sub(N, 
        net_params[param_id]->cpu_data(),
        val_slow->cpu_data(),
        net_params[param_id]->mutable_cpu_diff()
      );
    }

    break;
  }
  case Caffe::GPU: {
#ifndef CPU_ONLY
    
    ranger_update_gpu(N,
        net_params[param_id]->gpu_data(), net_params[param_id]->mutable_gpu_diff(),
        val_m->mutable_gpu_data(),
        val_v->mutable_gpu_data(),
        val_slow->gpu_data(), val_slow->mutable_gpu_data(),
        beta1, beta2, eps_hat, local_rate*correction,
        N_sma, N_sma_threshold,
        t, k_thres, alpha,
        use_lookahead
    );
    

#else
    NO_GPU;
#endif
    break;
  }
  default:
    LOG(FATAL) << "Unknown caffe mode: " << Caffe::mode();
  }
}

INSTANTIATE_CLASS(RangerSolver);
REGISTER_SOLVER_CLASS(Ranger);

}  // namespace caffe
