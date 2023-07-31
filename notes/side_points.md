# 简介：
- 这里分享 NLP 的理论基础知识

# Language Model 其他
- 为什么模型输入有最大的 Token 限制，比如 gpt-3.5-turbo 是 （4096 - max_gen_len）， LLaMA 是 （2048 - max_gen_len）？
  - 这和模型构造有关，这个限制了 K 和 V 的长度
- Temperature 机制是什么？怎么实现的？
  - Temperature 被抽象为：“越高模型输出就越有想象力越不按套路出牌”
  - 在模型输出的 logits 输入 softmax 进行 probabilities（probs） 计算时，将 logits 除以 temperature
  - 一般模型会设置判断条件 "if temperature > 0:", 所以几乎所有模型设置 Temperature = 0 就不影响其输出分布
- Top-P 机制是什么？怎么实现的？
  - 不选择 probs 最大的 next_token，而是从一簇累积概率大于 P 的 next_token 待选组中抽样
    - 通常和 Temperature > 0 合并使用（如LLaMA），会降低输出速度
  - 先排序，使 next_token 的预测组 降序排列
  - 选择 len 最小的 next_token 预测组，并采样
- 为什么 Prompt 的长度对 Runtime 的影响微乎其微，但生成长度 的影响这么大？
  - 因为第一次生成的时候，prompt 的所有 token 是同时输入进模型的，而后续继续生成的时候，只会一个一个（Batch） token 输入进去，所以每次生成都要进行一次模型推理
- 通常用什么来评估一个 Autoregressive Language Model 的性能？为什么？
  - 优秀的文章：Evaluation Metrics for Language Modeling
  - "Autoregressive Language Model" 指基于 preceding token 预测 next token 从而生成 text 的 model，如 GPT。另一种是 ”Masked LM“，如 BERT。
  - 有 Perplexity，Cross-Entropy, Bits-per-Character (BPC)
    - Perplexity（PPL），PPL 不适用于像 BERT 一样的 Masked Language Model
      - ”A measurement of how well a probability distribution or probability model predicts a sample"
      - 就如其中文 “困惑度“ 表示的一样，就是说当语言模型预测下一个词时，会有几种可能预测，可能的预测越多，且每个可能的概率越平均，模型就越不知道该预测哪个（就如同三个bit，完全未知情况下，其困惑度就是 8）
    - Cross-Entropy 用于 data 和 model predictions 之间
      - 计算式就是 {输入 x 经过 corpus P(.) 得到的 P(x) 的 信息量} 加上 {模型输出 Q(x) 和 P(x) 的 KL 散度} 【corpus 可以理解为一个语言的概率分布】
        -  H(P(x)) + KL_Divergence (P(x)|Q(x))
      - 直觉上就是通过降低 Cross-Entropy Loss，我们将模型 Q(.) 的概率分布逐渐贴近真实语料库的概率分布 P(.)
    - Bits-per-character（BPC) or Bits-per-word
      - 借用香农的一句话：if the language is translated into binary digits (0 or 1) in the most efficient way, the entropy is the average number of binary digits required per letter of the original language.
      - 所以 Average number of BPC 就是 entropy
        - 在 Pytorch，Tensorflow 中，entropy 中的对数运算不是以 2 为底，而是以 e 为底
- Causal Language Model 和 Mask Language Model 的区别
  - CLM 就是 Autoregressive LM，它是不断地预测下一个词，典之GPT
  - MLM 就是在训练的时候会将一些词给 mask 掉，让其预测，典之BERT
   