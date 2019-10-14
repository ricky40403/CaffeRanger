# CaffeRanger
Implementation of Solver(Optimizer) Ranger
(Radam + look ahead)

****
Radam : [On the Variance of the Adaptive Learning Rate and Beyond](https://arxiv.org/abs/1904.11486)  
Look ahead : [Lookahead Optimizer: k steps forward, 1 step back](https://arxiv.org/abs/1907.08610)  
Ptorch Version : <https://github.com/lessw2020/Ranger-Deep-Learning-Optimizer> 

****

caffe.proto

```
optional float ranger_alpha = 43 [default = 0.5];
optional int32 ranger_k_thres = 44 [default = 6];
optional float ranger_n_sma_threshold = 45 [default = 5.0];
optional bool ranger_use_radam = 45 [default = true];
optional bool ranger_use_lookahead = 45 [default = true];
```

Here use ranger_use_lookahead (ranger_use_radam has not decided where to use) to switch between radam and ranger
because when using l1 training, the training error will increase.  
It should because the lookahead is not a soft gradient when using l1. The loss between fast_move and slow_move may get a high loss and the model will confuse where to go.

