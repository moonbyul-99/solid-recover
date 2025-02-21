sr rna 模型实验报告

|log目录|config|动机|基本结果|
|---|---|---|---|
|config_1_2025-02-19-05-16|config1|构建一个AE base 模型| |
|config_2_2025-02-19-05-16|config2|构建一个VAE base 模型 | 无结果，当时代码有误，造成后验坍缩|
|config_3_2025-02-19-05-16|config3|构建一个AE + CE base 模型||
|config_4_2025-02-19-05-16|config4|构建一个VAE + CE base 模型|无结果，当时代码有误，造成后验坍缩|
|config_5_2025-02-19-05-16|config5|使用最早版本的模型一致的参数，构建一个VAE模型|无结果，当时代码有误，造成后验坍缩|
|config_6_2025-02-19-06-08|config6|复现先前版本的AE+CE模型| |
|config_7_2025-02-19-09-42|config7|使用更小的vae权重避免后验坍缩|解决后验坍缩问题，但是代码中的错误仍未修正|
|config_8_2025-02-19-10-23|config8|使用稍大的vae权重|再次后验坍缩，代码错误未修正|
|config_9_2025-02-19-11-14|config9|修正代码错误后，vae权重1检验后验坍缩 |意外中断 |
|config_2_2025-02-19-12-45|config2|修正错误后，构建一个VAE base 模型||
|config_4_2025-02-19-12-45|config4|修正错误后，构建一个VAE+CE base 模型||
|config_5_2025-02-19-12-45|config5|修正错误后，构建一个与先前版本同尺度的VAE模型||


- 2025-02-19 实验总结：
  1. 复现了先前版本的AE+CE模型
  2. config (1,3,6) 检验了AE模型的隐藏层维度以及CE loss的影响。从结果看，三次实验的重构Loss基本一致，使用更小隐层维度的 1,3 的误差略小一点，从 (1,3)看，ce_loss对重构效果影响不大，三次实验的ce_loss基本一致，与隐层维度关系不大。
  3. 修正了vae模型loss的错误，vae模型loss包含两部分，重构mse_loss和kl_loss，其中mse_loss计算时，对每个细胞，计算重构误差，之后在Batch维度上取均值，kl_loss计算时也应如此，但错误的写成了直接对样本对特征求和，缺失了batch维度取均值。这导致初始时kl_loss过大，训练时过快的减小Kl_loss，最终坍缩成0。实验2,4,5,7,8全部无意义。
  4. 修正loss错误后，config9 正常进行，kl_loss先上升之后稳定下来。


|log目录|config|动机|基本结果|
|---|---|---|---|
|config_9_2025-02-19-13-41|config9|上一个config9任务意外中断||
|ga_config1_2025-02-20-07-19|ga_config1|构建AE base | |
|ga_config2_2025-02-20-07-19|ga_config2|VAE base| |
|ga_config3_2025-02-20-07-19|ga_config3|AE+CE base| |
|ga_config4_2025-02-20-07-19|ga_config4|VAE+CE base| |
|ga_config5_2025-02-20-07-19|ga_config5|AE+CE large| |
|ga_config6_2025-02-20-07-19|ga_config6|VAE + CE large| |

- 2025-02-20 实验总结:
    1. 成功复现先前GA模型
    2. ga_config (1,3,5)(2,4,6) 隐层size影响较小，重构误差基本一致，large版本重构误差最小，base+ce 次之，但变化幅度不大。vae模型下，重构误差几乎完全一致。
    3. ga_config (1,2) (3,4) (5,6) VAE 模型的重构误差略大于 AE 模型，CE loss也略大于 AE 模型。
    4. ga_config (2,4,6) kl_loss都呈现先上升后下降最后平稳的趋势，使用CE loss会增大kl_loss 
    5. config (2,4,5,9) (1,3,6) 对VAE 模型，重构误差上 2,4 接近，小于5小于9，重构效果和隐层维度成非线性关系。CE loss上相同的现象。对AE 模型，重构误差上，有无CE差距不大，维度低一些的模型误差略小。
    6. config (1,2) (3,4) (6,5) 相同尺度下，AE模型重构误差略小于VAE模型，但差距不大。但CE loss上，小型的VAE模型loss小于AE模型，大型的VAE模型loss 大于 AE 模型
    7. config (4,5,9) kl_loss都呈现先上升后下降最后平稳的状态，使用CE loss会增大KL_loss。 
    8. 从训练loss上来看，VAE,AE 区别不大，CE loss有无对重构影响很小。维度的增加对模型的收益很小。考虑优先在小模型上对齐。