# 简介:
- 这里分享模型部署的理论基础知识

# 引荐
- 分享一篇博客，由于担心下架所以将其许多内容转载到了这里，非常推荐阅读
    - https://www.cnblogs.com/rossiXYZ/p/15172665.html
- 精彩的文章，非常推荐阅读：
    - https://blog.csdn.net/OneFlow_Official/article/details/116781168
---
# 模型训练并行机制
## 1. 前情提要
深度学习框架本质上是一个基于张量（Tensor）之间的计算（Operator）表达式所组成的计算图（Graph）编译执行引擎，提供了一系列张量的定义、一元操作、二元操作等数学原语，并根据反向传播算法（Back Propagation）进行梯度自动求导以及模型更新。在大量数据分批次流入计算图进行模型训练之后，使得模型学习到数据中的内在关联关系，从而获得对应场景中的“智能”感知与判断能力。
DNN训练的目标是在尽可能短的时间内获得一个高精度的模型。这一目标可以通过两个指标来实现：
- 统计效率，达到预期准确度所需的历元数；
- 硬件效率，完成单个历元所需的时间。达到期望精度水平的总训练时间只是这两个指标的乘积；
GPU最主要提供的是两种资源：计算资源 和 显存带宽资源。所以训练大型模型有两个基本挑战：显存效率和计算效率。
深度学习框架性能优化的最终目标是深度学习模型训练最快，从而使得完成训练的时间最短，节省模型训练开发周期和用户的时间成本。
## 2. 三种机制
在 "Efficient Large-Scale Language Model Training on GPU Clusters" 论文中， NVIDIA 介绍了分布式训练超大规模模型的三种并行技术：
- 数据并行（Data Parallelism）
- 模型并行（Tensor Model Parallelism）
- 流水并行（Pipeline Model Parallelism）
### 2.1 数据并行
数据并行（Data Parallelism）是最常见的方法。其特点如下：
- 模型在多个worker机器上复制，每个GPU都维护模型的完整副本。
- 输入数据集可以跨多个gpu进行分区。每批输入的训练数据都在数据并行的 worker 之间划分。每个worker处理训练数据的一个子集。
- 使用集合通信原语或参数服务器定期与其他GPU同步权重。
- 反向传播后需要通信并规约梯度，以保证优化器在各个 worker 上进行相同的更新。即，对单个worker计算的权重更新进行聚合，以获得反映所有输入更新的最终权重更新。
- 每个聚合传递的数据量与模型的大小成比例。
数据并行性具有几个明显的优势，包括计算效率高和实现起来工作量小，这使得数据并行训练在一些流行的具有高计算通信比的模型上运行良好，但有几个重要的趋势威胁着它的有效性：
- 显存效率：数据并行会在所有 worker 之间进行模型和优化器的复制，因此显存效率不高。
- 计算效率：随着我们提高并行度，每个 worker 执行的计算量是恒定的。数据并行可以在小规模上实现近乎线性扩展。但是，在 worker 之间规约梯度的通信开销跟模型大小成正相关，所以当模型很大或通信带宽很低时，计算效率会受限。而GPU计算能力的快速增长进一步将训练的瓶颈转移到跨模型的通信上。另外，参数同步的频率影响统计和硬件效率。
- 伸缩性：不断增长的模型大小增加了每次聚合的通信量。事实上，一些广泛使用的模型足够大，使得通信超过了计算时间，限制了伸缩性并支配了总的训练时间。而且，数据并行的 batch 大小随 worker 数量提高，而我们往往无法在不影响收敛性的情况下一直增加 batch 大小。
### 2.2 模型并行
模型并行在传统上用于训练过程中太大而无法保存在工作者内存或缓存中的模型。其特点如下：
- 模型并行涉及到将模型在worker之间进行划分，以便每个worker仅对模型参数的一个子集进行评估和更新。这样就可以分为层间并行和层内模型并行。
- 层间模型并行会在多个 worker 之间划分模型的各个层。
- 层内模型并行把每层的模型参数切分到多个设备。层内模型并行在有的论文里被叫做 "Tensor 级别的模型并行" ，是对某一层（如 Linear/Dense Layer 里的 Variable ）的模型 Tensor 切分，从而将大的模型 Tensor 分成多个相对较小的 Tensor 进行并行计算；
- 层间值（激活和梯度）往往是需要跨机器通信的唯一参数。
就其本质而言，模型并行性的计算和通信因模型结构而异，因此在实现上有很大的工作量。
然而，即使模型并行能够训练非常大的模型，传统的模型并行也会导致计算资源的严重利用率不足，因为它一次只主动使用一个worker（如果每个层被分配给一个worker），或者不能重叠计算和通信（如果每个层被分区）。
- 显存效率：模型并行DNN训练导致GPU资源的严重利用不足。模型并行通过在模型并行 worker 之间划分激活显存，会根据 worker 数量成比例地减少显存使用量。至关重要的是，这是减少单个网络层的激活显存的唯一方法。
- 计算效率：由于每次前向和反向传播中都需要额外通信激活值，模型并行的计算效率很低。模型并行需要高通信带宽，并且不能很好地扩展到通信带宽受限的节点。此外，每个模型并行worker 都会减少每个通信阶段之间执行的计算量，从而影响计算效率。模型并行性通常与数据并行性结合使用，以在内存和计算效率之间进行权衡。
- 开发效率：跨多个GPU划分模型的负担留给了程序员，即使对于最有经验的机器学习实践者来说，确定如何在工作者中最好地划分DNN模型也是一项具有挑战性的任务，这往往会导致额外的效率低下。最近有些工作探索了如何使用增强学习来自动确定模型并行性的设备位置。不幸的是，这样的在线决策技术是时间和资源密集型的；它们也不能无缝地结合流水线、数据并行和模型并行。
2.3 流水线并行
流水并行（Pipeline Model Parallelism）在有的论文里叫做流水线级别的模型并行，其特点是：
- 将整个网络分段（stage），不同段在不同的设备上，前后阶段流水分批工作，通过一种“接力”的方式并行。
- 流水线并行将模型的各层划分为可以并行处理的阶段。当一个阶段完成一个 micro-batch 的正向传递时，激活内存将被通信至流水线的下一个阶段。类似地，当下一阶段完成反向传播时，将通过管道反向通信梯度。必须同时计算多个 micro-batch 以确保流水线的各个阶段能并行计算。
- 流水线并行训练有可能在数据并行性困难时提供较高的DNN训练性能。特别是，工作人员之间的通信可以限制在分配给不同工作人员的相邻层之间的激活（在前向通道上）和梯度（后向）上。
但是流水线并行依然有一些问题：
- 显存效率：流水线并行减少的显存与流水线的阶段数成正比，使模型的大小可以随 worker 的数量线性扩展。但是，流水线并行不会减少每一层的激活函数的显存占用量。此外，每个 worker 必须存储同时运行的各个 micro-batch 的激活值。这导致流水线第一阶段的激活内存与单个 mirco batch 的总激活内存大致相同。
- 计算效率：流水线并行具有最低的通信量，因为它的通信量只和在各阶段边界的各层的激活值大小成正比。但是，它不能无限扩展。像模型并行一样，增加流水线大小会减少每个流水线阶段的计算量，这会降低计算与通信的比率。如果要实现好的计算效率，流水线并行还要求其每个阶段的计算负载完美的均衡。此外，流水线并行性会在每个 batch 的开始和结束时因为需要重新填充或排空流水线而产生 bubble overhead。
- 开发效率：DNN的双向性（正向传递后反向传递相同的层）使得流水线具有挑战性，更重要的是，一个简单的流水线机制引入了过时权重的最新计算，导致最终模型的精度低于数据并行训练。
## 3. 如何使用
给定一个特定的神经网络模型和一批计算资源，从任务到设备之间的映射有多种方式，但不同的映射方案运行效率不同。哪种方案最优既取决于作业本身的特性，也取决于底层硬件的拓扑。
神经网络由很多局部计算搭建组成，一般来说，同一个神经网络的不同算子可能适合不同的并行模式。一个局部计算是采用数据并行，还是模型并行取决于这个局部任务的计算传输比。比如：
- 某些算子（比如卷积） 运算参数量很小，但中间结果量大，为了较少传输量，所以最划算的方法是数据并行。即对数据进行切分，不同的设备处理不同的数据，在设备之间偶尔进行参数同步；
- 某些算子，中间计算结果相对于参数量更少，就适合模型并行。
- 还有一些算子，网络参数量/中间计算结果都很大，可能采用流水并行（也就是接力的形式）是最优的。
- 相比于一个算子只使用单一的并行模式，一个算子也可以同时使用多样的并行模式可能进一步地减少传输量，譬如在隐藏层比较大的地方，就可能同时对数据矩阵切割以及对模型矩阵切割。
因此，对于每一个任务选择最优的并行模式是一个非常复杂的问题，需要具体情况具体分析。

---
# Pytorch 的并行加速支持
## 1. 基础知识
PyTorch以Tensor为基本单元，更符合算法工程师写Python脚本的直觉，以面向对象的方式进行模型搭建和训练。对Tensor进行赋值、切片，就像numpy一样易用。
PyTorch 是单卡视角，一个设备上的 Tensor、模型脚本跟另一个设备上的 Tensor、模型脚本并无直接关系，对于每个设备上的模型脚本都完全对称的（Mirror）最简单的数据并行来说，PyTorch 这样的设计没有什么明显的缺陷。每个设备上的脚本运行到相同 batch 的模型更新部分（Optimizer），统一做一次模型同步（AllReduce 操作）就完成了数据并行，这就是 PyTorch 的 DDP（DistributedDataParallel）模块。
但在分布式情况下想把一个Tensor切分到不同机器上，需要手动构建传输过程，相当于直接对物理编程，所以对分布式使用的门槛更高。
## 2. 特点
PyTorch 模型并行将单个模型拆分到不同的GPU上，而不是在每个GPU上复制整个模型（具体来说，假设模型 m包含10层。如果使用 DataParallel，则每个GPU都具有这10层中每个层的副本，而如果在两个GPU上使用模型并行时，每个GPU可以托管5层）。
模型并行的高级思想是将模型的不同子网络放置在不同的设备上，并相应地实现该forward方法以跨设备移动中间输出。由于模型的一部分只在任何单个设备上运行，因此一组设备可以共同服务于一个更大的模型。
## 3. 基本用法
让我们从包含两个线性层的玩具模型（toy model）开始。要在两个GPU上运行此模型，只需将每个线性层放在不同的GPU上，然后移动输入（input）和中间输出（intermediate outputs）以匹配层设备（layer devices）。

---
```python
import torch
import torch.nn as nn
import torch.optim as optim

class ToyModel(nn.Module):
  def __init__(self):
    super(ToyModel, self).__init__()
    self.net1 = torch.nn.Linear(10, 10).to('cuda:0')  # 将net1放置在第1个GPU上
    self.relu = torch.nn.ReLU()
    self.net2 = torch.nn.Linear(10, 5).to('cuda:1')   # 将net2放置在第2个GPU上
    
    def forward(self, x):
    x = self.relu(self.net1(x.to('cuda:0')))  # 从第一个设备中拿到relu后的数据
    return self.net2(x.to('cuda:1'))  # 将上一层的输出（在CPU中？）输入第二个设备
```
---

请注意对于 ToyModel ，除了五个用于将线性层（linear layers）和张量（tensors）放置在适当的设备上的to(device)调用之外，以上内容与在单个GPU上实现该功能非常相似。这是模型中唯一需要更改地方（即to(device) ）。 backward()和 torch.optim 会自动关注梯度（gradients），就好像模型是一个GPU一样。调用损失函数时，只需确保标签（label）与输出（output）在同一设备（on the same device）上。

---
```python
model = ToyModel()
loss_fn = nn.MSELoss()
optimizer = optim.SGD(model.paraeters(), lr=0.001)

optimizer.zero_grad()
outputs = model(torch.randn(20, 10))
labels = torch.randn(20, 5).to('cuda:1') # ToyMode 的 output 是在 'cuda:1' 上，此处的 label 也应该置于 'cuda:1' 上
loss_fn(outputs,labels).backward()
optimizer.step()
```
---
## 4. 将模型并行化应用于现有模块
只需更改几行，就可以在多个GPU上运行现有的单GPU模块。以下代码显示了如何分解 torchvision.models.reset50() 为两个GPU。思想是从现有 ResNet模块继承，并在构建过程中将层拆分为两个GPU。然后，覆盖 forward方法来缝合两个子网，通过相应地移动中间输出。

---
```python
from torchvision.models.resnet import ResNet, Bottleneck

num_classes = 1000
class ModelParallelResNet50(ResNet):
    def __init__(self, *args, kwargs):
        super(ModelParallelResNet50, self).__init__(
            Bottleneck, [3, 4, 6, 3], num_classes=num_classes, *args, kwargs)

        self.seq1 = nn.Sequential(
            self.conv1,
            self.bn1,
            self.relu,
            self.maxpool,

            self.layer1,
            self.layer2
        ).to('cuda:0')  # 放置在第1个GPU上

        self.seq2 = nn.Sequential(
            self.layer3,
            self.layer4,
            self.avgpool,
        ).to('cuda:1')  # 放置在第2个GPU上

        self.fc.to('cuda:1')

    def forward(self, x):
        x = self.seq2(self.seq1(x).to('cuda:1'))
        return self.fc(x.view(x.size(0), -1))
```
---
对于模型太大而无法放入单个GPU的情况，上述实现解决了该问题。但是，你可能已经注意到，如果模型合适，它（model parallel）将比在单个GPU上运行要慢。这是因为在任何时间点，两个GPU中只有一个在工作，而另一个在那儿什么也没做。在 layer2 和 layer3之间，中间输出需要从 cuda:0 复制到 cuda:1，这使得性能进一步恶化。
在整个执行过程中，两个GPU中的一个会处于空闲状态。为了解决这个问题，有一种选择是将每个批次进一步划分为拆分流水线，以便当一个拆分到达第二子网时，可以将下一个拆分馈入第一子网。这样，两个连续的拆分可以在两个GPU上同时运行。
## 5. 通过流水线输入（Pipelining Inputs）加速
在以下实验中，我们将每批次 120-image 进一步划分为 20-image 。当PyTorch异步启动CUDA操作时，该实现无需生成多个线程即可实现并发。

---
```python
class PipelineParallelResNet50(ModelParallelResNet50):
    def __init__(self, split_size=20, *args, kwargs):
        super(PipelineParallelResNet50, self).__init__(*args, kwargs)
        self.split_size = split_size

    def forward(self, x):
        splits = iter(x.split(self.split_size, dim=0))  # iter() 生成一个iterator
        s_next = next(splits)  # 通过next() 轮询地拿这个iterator的值
        s_prev = self.seq1(s_next).to('cuda:1')
        ret = []

        for s_next in splits:
            # s_prev runs on cuda:1
            s_prev = self.seq2(s_prev)
            ret.append(self.fc(s_prev.view(s_prev.size(0), -1)))

            # s_next runs on cuda:0, which can run concurrently with A
            s_prev = self.seq1(s_next).to('cuda:1')

        s_prev = self.seq2(s_prev)
        ret.append(self.fc(s_prev.view(s_prev.size(0), -1)))

        return torch.cat(ret)


setup = "model = PipelineParallelResNet50()"
pp_run_times = timeit.repeat(
    stmt, setup, number=1, repeat=num_repeat, globals=globals())
pp_mean, pp_std = np.mean(pp_run_times), np.std(pp_run_times)

plot([mp_mean, rn_mean, pp_mean],
     [mp_std, rn_std, pp_std],
     ['Model Parallel', 'Single GPU', 'Pipelining Model Parallel'],
     'mp_vs_rn_vs_pp.png')
```
---
请注意，设备到设备的张量复制操作在源设备和目标设备上的当前流（current streams）上同步。如果创建多个流，则必须确保复制操作正确同步。在完成复制操作之前写入源张量或读取/写入目标张量可能导致不确定的行为。上面的实现仅在源设备和目标设备上都使用默认流，因此没有必要强制执行其他同步。

## 6. 关键技术
因为每个模型的并行策略候选集合是指数级的，纯手工从中挑出一种合适的并行策略，需要耗费算法工程师大量的时间以及计算资源，而且算法工程师需要考虑的相关事宜太多，比如：如何分配内存，层之间如何交互，如何减少通信代价，分割的张量不能破坏原有数学模型，如何确定确定张量shape，如何确定输入输出等等。
所以自动并行技术（如何从框架层次自动解决并行策略选择问题）成为一个研究热点。
自动并行通过建立代价模型来预测并挑选一个较优的并行策略（暂时无法保证是最优的策略，因为挑出最优的策略是个NP-Hard的问题），有希望将算法工程师从并行策略的选择和配置中解放出来。
因此，目前分布式模型训练有几个必要并行技术：
- 流水并行，尤其是如何自动设定流水；
- 梯度累加；
- 后向重计算；
- 1F1B 策略（我们将采用PipeDream分析）；
内存计算
1. 总体分析
我们主要参考了 ZeRO: Memory Optimization Towards Training A Trillion Parameter Models 这篇论文的思路。
在模型训练期间，大部分内存被以下三种情况之一消耗：
- i）激活。
- ii）OGP状态，即由优化器状态、参数梯度和参数本身组成的张量。
- iii）临时缓冲区。
输入数据所占用的显存其实并不大，这是因为我们往往采用迭代器的方式读取数据，这意味着我们其实并不是一次性的将所有数据读入显存，而这保证每次输入所占用的显存与整个网络参数来比是微不足道的。
我们逐一分析。
2. 激活函数
对于激活函数，有如下特点：
- 激活函数额外消耗的显存随 batch 大小而增加，batch 设置为1的情况下，训练万亿参数模型就会产生超过 1 TB 的激活函数用的显存。
- 业界已有方案如论文 Training deep nets with sublinear memory cost ，可以以33%的重新计算开销为代价，几乎可以消除激活所需的所有内存。这个技巧叫做 gradient checkpointing, 有时还叫做 rematerialization, re-forward。
3. OGP状态
3.1 模型自身参数
模型自身的参数指的就是各个网络层的 Weight 和Bias，这部分显存在模型加载完成之后就会被占用。另外需要注意到的是，有些层是有参数的，如CNN， RNN； 而有些层是无参数的， 如激活层， 池化层等。
3.2 优化器参数
优化器参数指的是模型在优化过程即反向传播中所产生的参数， 这部分参数主要指的就是 dw， 即梯度，在SGD中， 其大小与参数一样， 因此在优化期间， 模型参数所占用的显存会翻倍。
值得注意的是，不同的优化器其所需保存的优化参数不同， 对于 Adam， 由于其还需要保存其余参数， 模型的参数量会在优化区间翻 4 倍。
3.3 样例
对于OGP状态，让我们以ADAM为例。使用ADAM对具有Ψ个参数的模型进行混合精度训练。
- 模型参数：参数的fp16副本，内存要求2Ψ字节。
- 梯度 ：梯度的fp16副本，内存要求2Ψ字节。
- 优化器状态：参数、Momentum和Variance的fp32副本，内存需求分别为4Ψ、4Ψ和4Ψ字节。让我们用K来表示优化器状态的内存乘数，也就是说，存储它们所需的额外内存是KΨ字节。
总的来说，OGP状态有2Ψ+2Ψ+KΨ=16Ψ字节（混合精度ADAM的K=12）的内存需求。
具体如下：
[图片]
蓝色是参数，橙色是梯度，绿色是优化器状态。
在内存消耗公式中，Ψ表示模型大小（参数个数），K表示优化器状态的内存乘数，Nd表示数据并行度。在这个例子中，我们假设模型大小为Ψ=75亿，基于Adam优化器的混合精度训练，数据并行度为Nd=64（即64个GPU），K=12。
对于一个拥有15亿个参数的GPT-2这样的模型，这导致了至少24gb的内存需求，远远高于单独保存fp16参数所需的3gb内存。
3.4 临时缓冲区
临时缓冲区是用于存储临时结果的缓冲区，例如，对于参数为15亿的GPT-2模型， fp32缓冲区将需要6GB的内存。

---
算力计算
1. 背景知识
- FLOPS：注意全大写，是floating point operations per second的缩写，意指每秒浮点运算次数，理解为计算速度。是一个衡量硬件性能的指标。
- FLOPs：注意s小写，是floating point operations的缩写（s表复数），意指浮点运算数，理解为计算量。可以用来衡量算法/模型的复杂度。
前向传播时所需的计算力就是由FLOPs体现，那么FLOPs该怎么计算呢？
我们知道，在一个模型进行前向传播的时候，会进行卷积、池化、BatchNorm、Relu、Upsample等操作。这些操作的进行都会有其对应的计算力消耗产生，其中，卷积所对应的计算力消耗是所占比重最高的。所以，我们以卷积操作为例，看看卷积所对应的计算力。
推导过程：卷积层 wx + b 需要计算两部分，首先考虑前半部分 wx 的计算量：
令 :
- k 表示卷积核大小;
- c 表示输入 feature map 的数量;
则对于输出 feature map 上的单个 Unit 有：
k * k * c 次乘法，以及 k * k * c - 1 次加法
如果输出 feature map 的分辨率是 H * W ，且输出 o 个 feature map，则输出 feature map 包含 Unit的总数就是 H * W * o。
因此，该卷积层在计算 wx 时有:
k * k * c * H * W * o 次乘法          --（1）
(k * k * c - 1) * H * W * o 次加法    --（2）
再考虑偏置项 b 包含的计算量：
由于 b 只存在加法运算，输出 feature map 上的每个 Unit 做一次偏置项加法。因此，该卷积层在计算偏置项时总共包含：
H * W * o 次加法      --（3）
将该卷积层的 wx 和 b 两部分的计算次数累计起来就有：
式(1) 次乘法:
k * k * c * H * W * o 次乘法
式(2) + 式(3) 次加法:
(k * k * c - 1) * H * W * o  + H * W * o  = k * k * c * H * W * o
可见，式(2) + 式(3) = 式 (1)
因此，对于带偏置项的卷积层，该层的计算力消耗 为：
k * k * c * H * W * o

---

# 流水线
## 1. 背景知识
### 1.1 问题点
通信问题
无论是数据并行还是模型并行，都会在相应的机器之间进行全连接的通信，当机器数量增大时，通信开销和时延会大到难以忍受。
比如参数服务器使用中，三段式流程如下：Pull weight ---> Compute new weight ---> Push new weight。
如果三段式流程串行的进行通信和计算，无论这个通信是快是慢，这个时间开销都会导致在分布式环境下每个iteration的时间比单机版要长（Ethernet的带宽高低或者latency大小只会影响这个时间的长短，但并不能把这个时间降到零）。所以，把通信和计算重叠（overlap）起来以便 “掩盖” 通信时间几乎是一个必须的步骤。如何重叠计算和传输从而提高设备利用率就非常有挑战。
无法充分利用
原则上我们可以通过并行计算在GPU或者TPU上训练更大的DNN模型。但是由于DNN的顺序性，这种方法可能导致在计算期间只有一个加速器处于活动状态，不能充分利用设备的计算能力。
### 1.2 如何设计系统
回到神经网络的训练过程上，怎么设计系统来重叠计算和通信？
在后向传播之中有两个特点可以利用：
- 首先，神经网络的计算是一层接着一层完成的，不管是前向还是后向传播，算完本层才能算下一层；
- 另一方面，在后向传播的过程中，一旦后一层拿到前一层的输入，这一层的计算就不再依赖于前一层了。
因此，根据这个特点，人们引入了流水线并行。
数据并行与模型并行都是让设备执行同一个层次的计算，流水并行则是把任务划分为几个有明确先后顺序的阶段，把不同的阶段分给不同的计算设备，使得单设备只负责网络中部分层的计算。模型网络分布在各个设备上是非对称的，各个设备“接力”执行网络的一部分。每个阶段（stage） 和下一个阶段之间仅有相邻的某一个 Tensor 数据需要传输，每台机器的数据传输量跟总的网络大小、机器总数、并行规模无关。
在这种多设备接力完成一个网络计算的模式下，可以支持更大的模型或者支持更大的 Batch Size。如果通信和计算重叠（overlap）好，又可以解决了机器之间的通信开销的问题，
总的来说，流水并行在通常大模型训练情况下具有优势。流水并行的数据传输量少，仅为阶段之间需要传输的数据量之和，不像数据并行与模型并行那样大，传输量与整个计算图都有关，因此对于带宽较小的机器，会趋于使用流水并行。但某些情况下，流水并行与模型并行的结合则会优于单一的模型并行与流水并行。同时，在数据并行与模型并行中也存在计算时间掩盖传输时间的优化。
## 2. Gpipe流水线综述
### 2.1 关键要点
从概念上讲，GPipe是一个分布式机器学习库，它使用同步随机梯度下降和流水线并行的方式进行训练，适用于任何由多个有序的层组成的深度神经网络(Deep Neural Networks, DNN)。 Gpipe通过跨不同的加速器来分割模型，并自动将一小批训练示例划分成更小的批量。 该模型允许GPipe的加速器进行并行操作，最大限度地提高了训练过程中的可扩展性。
GPipe 有几个关键要点:
- Network partition（网络分片）：将一个N层的网络划分成K个partition, 每个partition在单独的TPU上执行，partition之间需要插入一些网络通信操作。
- Pipeline parallelism（流水线并行）：把CPU里的流水线并发技术用在了深度学习上，主要是把计算和网络通信两种操作，更好地重排列。即自动将mini-batch的训练样本分成更小的micro-batch，并在pipeline中运行，使TPU核心能够并行操作。
- Gradient Accumulation（梯度累积） ：梯度一直在micro-batch中累积，因此分区数量不会影响模型质量。
- Re-Materialization（重计算) ：Re-Materialization具体是指在前向计算过程中，GPipe只记录stage划分处的输出，在计算梯度时，GPipe会重新执行前向计算逻辑，从而得到各个算子的前向结果，然后再计算梯度结果。跟OpenAI开源的gradient-checkpointing一样，只不过GPipe是在TPU上实现的，OpenAI的只能运行在GPU上。
### 2.2 图示
- 下图左端是原始模型。
- 右端说明了具有多个有序层的神经网络的GPipe模型被划分到了四个加速器上。 Fk是kth分区的复合正向计算函数。 Bk是其相对应的反向传播函数。 Bk依赖于来自上层的Bk+1和Fk的中间激活函数。
  - 上面的模型中，我们可以看到网络的顺序性是如何导致资源利用不足的。
  - 下面的模型则显示了GPipe方法，在该方法中，输入的小批量示例被划分为更小的微批量，这些微批量可以由加速器同时处理。

### 2.3 问题
我们根据论文思路，提出了几个问题，争取以后按图索骥。
- 如何划分 stage？
  - 将模型划分为连续的几个stage，每个stage各自对应一个设备。这样就使得模型的大小可以突破单个设备内存的大小，因为一台设备只需要能够容纳部分模型的参数和计算；
  - 因为划分了stage，所以整个系统中，处理最慢的stage会成为瓶颈。所以应该平均分配算力。
- 依据什么分割做流水？
  - 如何将mini-batch进一步划分成更小的micro-batch，同时利用pipipline方案，每次处理一个micro-batch的数据，得到结果后，将该micro-batch的结果发送给下游设备，同时开始处理后一个 micro-batch的数据，通过这套方案减小设备中的Bubble（设备空闲的时间称为 Bubble）。
- 如何做Re-Materialization？
## 3. 依据算力分割Stage
神经网络有一个特点：对不同的输入，其运行时间相差不大，因此可以预估其算力，时间，参数大小等等。Gpipe就是依据算力对图进行了分割，从而把不同层分配到不同的设备上。
### 3.1 PartitionSequentialLayers
PartitionSequentialLayers 把一个包括sequential layers的层分解，目的是让每一个partition都大致拥有同样的 flops。最终目的是让每个 GPU 都拥有尽量同样的算力。
- 输入是：一个layer param 或者 一个 layer param 列表；
- 输出是：一个 FeatureExtractionLayer params 列表；
逻辑是：
- 如果params只是一个layer，那么就把这个layer构建成一个包含sub-layers的列表 subs；
- 利用 FPropMeta 计算出来这个 subs 列表的shapes和总flops，赋值给了 histo；
- 利用 histo 计算出来一个层代价（layer's cost）的归一化累积直方图；
- 构建一个parts变量；
  - 该变量是一个num_partitions大小的数组；数组每个item也是一个数组；
  - 依据直方图把subs分到parts中的每个item之中，这样每个parts[i]都拥有部分layers，一些算力小的算子被合并到一个 part 之中，目的是让最终 parts 每个item的算力尽量相同；
- 把parts转换成一个 FeatureExtractionLayer param 列表；

```python
def PartitionSequentialLayers(params, num_partitions, *shapes):
  r"""Partition a layer composed of sequential layers.

  This routine strives to partition layers so that each partition costs roughly
  the same flops given the input shapes.

  Args:
    params: A layer param or a list of layer param.
    num_partitions: The desired number of partitions.
    *shapes: A tuple of tshape.Shape representing input tensors to the first
      layer.

  Returns:
    A list of FeatureExtractionLayer params.
  """# Recursively concatenate SequentialLayer into a list.# SequentialLayer 是一个层，其作用是把若干层按顺序连接起来def FlattenSeq(p):
    if isinstance(p, list): # 已经是列表则返回return p
    if p.cls not in [builder_layers.SequentialLayer, FeatureExtractionLayer]:
      return [p.Copy()]
    subs = []
    for _ in range(p.repeat): # 把p包含的所有层都组装成一个层列表for s in p.sub:
        subs += FlattenSeq(s)
    return subs

  # 如果params是一个layer，那么就依据这个layer，构建一个包含sub-layers的新列表subs，如果是列表则直接返回
  subs = FlattenSeq(params)

  assert len(shapes) == 1
  tf.logging.info('num_partitions: {} input_shape: {}'.format(
      num_partitions, shapes[0]))

  # 利用 FPropMeta 计算出来这个 subs 列表的shapes和总flops，赋值给了 histo# Computes the estimate cost for each sub layer.# 假设有7个sub-layers，其flops分别是 10，40，30，10，20，50,10
  total, histo, output_shapes = 0, [], []
  for i, s in enumerate(subs):
    s.name = 'cell_%03d' % i
    meta = s.cls.FPropMeta(s, shapes) # 
    total += meta.flops
    histo.append(total)
    output_shapes.append(meta.out_shapes)
    shapes = meta.out_shapes
  tf.logging.vlog(1, 'len %d histogram = %s', len(subs), histo)
  # 则对应的histo 为：[10，50，80，90，110，160, 170]，total为170# 利用 histo 计算出来一个层代价（layer's cost）的归一化累积直方图# Computes the normalized cumulative histogram of the layer's cost.
  histo_pct = [float(x / total) for x in histo]
  tf.logging.vlog(1, 'cost pct = %s', histo_pct)
  # histo_pct 为 [1/17,5/17,8/17,9/17,11/17,16/17, 1], # 假设 num_partitions = 3# 构建一个parts变量，该变量是一个num_partitions大小的数组，数组每个item也是一个数组# 依据直方图把subs分到parts中的每个item之中，这样每个parts[i]都拥有部分layers，目的是让最终 parts 每个item的算力尽量相同# i-th sub layer is put into partition j, where j is roughly i-th cumulative# histogram times num_partitions.

  parts = [[] for _ in range(num_partitions)]
  parts_cost = [0]  num_partitions
  pre_hist_cost = 0for i, s in enumerate(subs):
    # 从histogram数组中找出s对应cost的index，j也就是s对应的partition# 对于i,s，则 histo_pct[i] * num_partitions 分别为: [3/17, 15/17, 24/17, 27/17, 33/17, 48/17，3]，j分别为[0,0,1,1,1,2,2]
    j = min(int(histo_pct[i] * num_partitions), num_partitions - 1)
    # The boundary at parts[j] where j > 0if j > 0 and not parts[j]:
      parts_cost[j - 1] = histo_pct[i - 1] - pre_hist_cost
      pre_hist_cost = histo_pct[i - 1]
    parts[j].append(s) # 把s加入到对应的partition# 三个桶内容分别为:[1,2],[3,4,5],[6,7]# 对应每个桶的flops为: [60,280,330]# 把parts转换成一个 FeatureExtractionLayer 列表
  parts_cost[num_partitions - 1] = 1.0 - pre_hist_cost
  seqs = []
  for i, pa in enumerate(parts):
    tf.logging.info('Partition %d #subs %d #cost %.3f', i, len(pa),
                         parts_cost[i])
    seqs.append(FeatureExtractionLayer.Params().Set(name='d%d' % i, sub=pa))
  return seqs
```
---

## 3.2 FeatureExtractionLayer
上面代码中使用了FeatureExtractionLayer，其功能就是返回一个层的序列。
FeatureExtractionLayer 从一个层序列中提取特征，具体特点是：
- 把一些层连接成一个序列；
- 可以 得到&传递 激活点；

```python
class FeatureExtractionLayer(base_layer.BaseLayer):
  """A layer that extrac features from a sequence of layers.

  FeatureExtractionLayer is a layer which connects a few layers in a sequence.
  It is also capable of fetching and forwarding activation endpoints.
  # TODO(huangyp): Make it a sublayer of builder_layers.SequentialLayer
  """  @classmethoddef Params(cls):
    p = super().Params()
    p.Define('variable_name_prefix', '',
             'Prefix for variable names in sub layers')
    p.Define('sub', [], 'A list of layers\' params.')
    p.Define('num_act_inputs', 0, 'Number of activation inputs.')
    p.Define('num_act_outputs', 0, 'Number of activation outputs.')
    p.Define('act_fetch_layers', [],
             'Names of fetch layers that cached extra activations')
    return p

  def __init__(self, params):
    super().__init__(params)
    p = self.params
    assert p.num_act_inputs >= 0assert p.num_act_outputs >= 0
    p.act_fetch_layers = p.act_fetch_layers or []
    assert p.num_act_outputs == p.num_act_inputs + len(p.act_fetch_layers)
    self._seq = []
    for sub in p.sub:
      assert sub.name
      sub.name = p.variable_name_prefix + sub.name
      self.CreateChild(sub.name, sub)
      self._seq.append((sub.name, self.children[sub.name])) # 把一些层连接成一个序列def FProp(self, theta, *args): # 实现该层的前向传播，在计算的前向step时将会被调用
    p = self.params
    assert len(args) > p.num_act_inputs
    out_args = args[:-p.num_act_inputs] if p.num_act_inputs > 0 else args
    extra_args = args[-p.num_act_inputs:] if p.num_act_inputs > 0 else ()
    for (name, ch) in self._seq:
      th = theta[name]
      out_args = _ToTuple(out_args)
      out_args = ch.FProp(th, *out_args)
    # Append fetched activations to fprop outputs.for fetch_layer in p.act_fetch_layers:
      assert fetch_layer in self.children
      activation = self.children[fetch_layer].activation # 子层激活点if isinstance(activation, (tuple, list)):
        activation = activation[0] # 如果是list，得到相应激活点
      extra_args += (activation,) # 把激活点添加进来if extra_args:
      out_args = _ToTuple(out_args) + extra_args # 最终返回所有激活点return out_args

  @classmethoddef FPropMeta(cls, p, *args): # 返回该层关于`FProp`计算的元数据assert len(args) > p.num_act_inputs
    seq_args = args[:-p.num_act_inputs] if p.num_act_inputs > 0 else args
    extra_args = args[-p.num_act_inputs:] if p.num_act_inputs > 0 else ()
    total = 0
    act_fetch_metas = {}
    for sub in p.sub:
      meta = sub.cls.FPropMeta(sub, *seq_args)
      if sub.name in p.act_fetch_layers:
        act_fetch_metas[sub.name] = meta.out_shapes[0]
      total += meta.flops
      seq_args = meta.out_shapes
    for fetch_layer in p.act_fetch_layers:
      extra_args += (act_fetch_metas[fetch_layer],)
    return py_utils.NestedMap(flops=total, out_shapes=seq_args + extra_args)
```

```
计算过程如下图，里面具体数值请参见上面几段代码之中的举例：
  +--------------+   +--------------+   +---------------+
  |              |   |              |   |               |
  |  sub-layer 1 |   |  sub-layer 2 |   |  sub-layer n  |
  |              |   |              |   |               |
  +-------+------+   +-------+------+   +--------+------+
          |                  |                   |
          |FPropMeta         |FPropMeta          |FPropMeta
          |                  |                   |
          v                  v                   v
       flops 1            flops 2             flops n
          +                  +                   +
          |                  |                   |
          |                  |                   |
          +--------------------------------------+
                             |
                             |
                             v
                  for i, s in enumerate(subs):
                     total += meta.flops
                     histo.append(total)
                  histo=[10,50,80,90,110,160,170]
                             +
                             |
                             |
                             v
Computes the normalized cumulative histogram of the layer's cost
        histo_pct = [float(x / total) for x in histo]
       histo_pct=[1/17,5/17,8/17,9/17,11/17,16/17,1]
                             +
                             |
                             |
                             +
           Assign layers to partition based on histogram
                   [1,2],[3,4,5],[6,7]
                             +
                             |
                             |
                             v
      +----------------------+----------------------------+
      | parts                                             |
      |                                                   |
      | +--------------+  +------------+  +-------------+ |
      | | sub-layer 1  |  |sub-layer 3 |  | sub-layer 6 | |
      | |              |  |            |  |             | |
      | | sub-layer 2  |  |sub-layer 4 |  | sub-layer 7 | |
      | |              |  |            |  |             | |
      | |              |  |sub-layer 5 |  |             | |
      | +--------------+  +------------+  +-------------+ |
      +---------------------------------------------------+
```

# 4. 流水线分配
## 4.1 基础类 SeqLayer
该层的目的是：用 Round-robin 策略把 cell_tpl之中的每个children cell 部署到 工作设备之上。
Params 包括：
- before_tpl ：配置在流水线之前运行的CNN层；
- cell_tpl ：FeatureExtractionLayer 列表；
### 4.1.1 初始化
初始化函数的逻辑是：
- 遍历before_tpl，对于每个item调用CreateChild构建其子层，把item添加到 _before_layers 之中；
- 遍历cell_tpl，对于每个item调用CreateChild构建其子层，把item添加到 _cells 之中；

```python
  def __init__(self, params):
    super().__init__(params)
    p = self.params
    self._before_layers = []
    self._cells = []
    # 遍历before_tpl，对于每个item调用CreateChild构建其子层，把item添加到 _before_layers 之中for l in p.before_tpl:
      self.CreateChild(l.name, l)
      self._before_layers.append((l.name, self.children[l.name]))
    # 遍历cell_tpl，对于每个item调用CreateChild构建其子层，把item添加到 _cells 之中  for l in p.cell_tpl:
      self.CreateChild(l.name, l)
      self._cells.append((l.name, self.children[l.name]))
```

### 4.1.2 _CreateChildrenVariables
构建变量。逻辑如下：
- 如果使用 tpu，则
  - 利用 cluster.WorkerDeviceInModelSplit(0) 来构建 before_tpl_device，即用集群的第一个设备作为 before_tpl_device；
  - 遍历集群的其他设备，分配给cell_devices；
- 遍历 _before_layers，把其中每个变量部署在 before_tpl_device；
- 遍历 _cells，把其中每个变量部署在 cell_devices；

```python
  def _CreateChildrenVariables(self):
    p = self.params

    num_cells = len(p.cell_tpl)
    before_tpl_device = ''
    cell_devices = [''] * num_cells
    if py_utils.use_tpu(): # 如果使用 tpu# 利用 `cluster.WorkerDeviceInModelSplit(0)` 来构建 before_tpl_device，即用集群的第一个设备作为 before_tpl_device
      cluster = self.cluster
      before_tpl_device = cluster.WorkerDeviceInModelSplit(0)
      # 遍历集群的其他设备，分配给cell_devices
      cell_devices = [
          cluster.WorkerDeviceInModelSplit(i) for i in range(num_cells)
      ]

    # 遍历 _before_layers，把其中每个变量部署在 before_tpl_devicefor unused_name, l in self._before_layers:
      with tf.device(before_tpl_device):
        l.InstantiateVariables()

    # 遍历 _cells，把其中每个变量部署在 cell_devicesfor i, (unused_name, l) in enumerate(self._cells):
      with tf.device(cell_devices[i]):
        l.InstantiateVariables()

    super()._CreateChildrenVariables()
```
### 4.1.3 FProp
前向传播代码，具体逻辑如下：
- 遍历 _before_layers，对于其中每层调用其FProp；
- 遍历 _cells，对于其中每层，在cluster.WorkerDeviceInModelSplit(i)之上调用其FProp；

```python
  def FProp(self, theta, *args):
    """Round-robin every children cells in cell_tpl among worker devices.

    Args:
      theta: A NestedMap object containing weights' values of this layer and its
        children layers.
      *args: Input args

    Returns:
      A list contains one tensor of [batch_size, feature_height, feature_width,
        channel].
    """
    num_layers = len(self.params.cell_tpl)
    cluster = self.cluster

    # 遍历 _before_layers，对于其中每层调用其FProp        for (name, l) in self._before_layers:
      l_theta = theta[name]
      args = _ToTuple(args)
      args = l.FProp(l_theta, *args)
    # 遍历 _cells，对于其中每层，在`cluster.WorkerDeviceInModelSplit(i)`之上调用其FProp  for i in range(num_layers):
      with tf.device(cluster.WorkerDeviceInModelSplit(i)):
        cell_name, cell = self._cells[i]
        args = _ToTuple(args)
        args = cell.FProp(theta[cell_name], *args)

    return args
```

### 4.1.4 具体实现
SeqLayer 全部代码如下：

```python
class SeqLayer(base_layer.BaseLayer):
  """Round-robin every children cells in cell_tpl among worker devices."""  @classmethoddef Params(cls):
    p = super().Params()
    p.Define('before_tpl', [],
             'Config for the CNN layers that runs before pipelining.')
    p.Define('cell_tpl', [], 'A list of FeatureExtractionLayer layers.')
    return p

  def __init__(self, params):
    super().__init__(params)
    p = self.params
    self._before_layers = []
    self._cells = []
    for l in p.before_tpl:
      self.CreateChild(l.name, l)
      self._before_layers.append((l.name, self.children[l.name]))
    for l in p.cell_tpl:
      self.CreateChild(l.name, l)
      self._cells.append((l.name, self.children[l.name]))

  def _CreateChildrenVariables(self):
    p = self.params

    num_cells = len(p.cell_tpl)
    before_tpl_device = ''
    cell_devices = [''] * num_cells
    if py_utils.use_tpu():
      cluster = self.cluster
      before_tpl_device = cluster.WorkerDeviceInModelSplit(0)
      cell_devices = [
          cluster.WorkerDeviceInModelSplit(i) for i in range(num_cells)
      ]

    for unused_name, l in self._before_layers:
      with tf.device(before_tpl_device):
        l.InstantiateVariables()

    for i, (unused_name, l) in enumerate(self._cells):
      with tf.device(cell_devices[i]):
        l.InstantiateVariables()

    super()._CreateChildrenVariables()

  def FProp(self, theta, *args):
    """Round-robin every children cells in cell_tpl among worker devices.

    Args:
      theta: A NestedMap object containing weights' values of this layer and its
        children layers.
      *args: Input args

    Returns:
      A list contains one tensor of [batch_size, feature_height, feature_width,
        channel].
    """
    num_layers = len(self.params.cell_tpl)
    cluster = self.cluster

    for (name, l) in self._before_layers:
      l_theta = theta[name]
      args = _ToTuple(args)
      args = l.FProp(l_theta, *args)
    for i in range(num_layers):
      with tf.device(cluster.WorkerDeviceInModelSplit(i)):
        cell_name, cell = self._cells[i]
        args = _ToTuple(args)
        args = cell.FProp(theta[cell_name], *args)

    return args
```

## 4.2 具体分配 PipeliningLayer
PipeliningLayer 是 SeqLayer 的派生类。
- 在流水线最前面是device[0]，负责处理前置条件。
- 流水线中间是一系列 device，负责处理具体的 micro batches。
- 流水线最后是 device[-1]，负责整理形状，最后输出一个最终张量。
### 4.2.1 得到中间层输出形状
_CalculateOutputShapes 计算出中间层的output shape。具体逻辑如下：
- 遍历 _before_layers，对其中每层调用其FPropMeta，得到 output shapes，插入 state_shapes 数组之中；
- 遍历 _cells，对其中每层调用其FPropMeta，得到 output shapes，插入 state_shapes 数组之中；

```python
  def _CalculateOutputShapes(self, input_shapes):
    """Calcuate the output shape of intermediate layers.

    Given the FPropMeta function in each FeatureExtractionLayer, calcuates
    the shapes of outputs of that layer. This is used to recover the shape
    information in StackedRecurrent.

    Args:
      input_shapes: NestedMap or tuple of input TensorShapes.

    Returns:
      Return a list of K + 1 NestedMaps or lists of tShape where K is
      the number of partitions.
    """
    p = self.params
    shapes = []

    # Converts TensorShape to tshape.Shape.def _ToTShape(x):
      if x is None:
        return Nonereturn tshape.Shape(x.as_list())

    shapes = py_utils.Transform(_ToTShape, input_shapes)
    shapes = _ToTuple(shapes)

    state_shapes = []
    # 遍历_before_layers，对其中每层调用其FPropMeta，得到 output shapes，插入 state_shapes 数组之中for (_, cell) in self._before_layers:
      shapes = cell.FPropMeta(cell.params, *shapes).out_shapes

    state_shapes.append(shapes[0] if p.nested_map_fprop else shapes)

    # 遍历 _cells，对其中每层调用其FPropMeta，得到 output shapes，插入 state_shapes 数组之中for (_, cell) in self._cells:
      shapes = cell.FPropMeta(cell.params, *shapes).out_shapes
      state_shapes.append(shapes[0] if p.nested_map_fprop else shapes)

    return state_shapes
```

### 4.2.2 得到数据类型
_get_state_dtype 的作用是得到数据类型。

```python
  def _get_state_dtype(self, *args):
    if self.params.state_dtype:
      return self.params.state_dtype
    if self.params.nested_map_fprop:
      inputs = args[0].Filter(lambda x: x is not None)
      return py_utils.Flatten(inputs)[0].dtype
    return args[0].dtype
```

### 4.2.3 得到输入形状
Gpipe 会首先将一个小批量的训练样本（mini-batch）分割成更小的小批量（micro-batches），然后将每组小批量的执行通过管道传送到单元上。
_get_input_shapes作用是得到输入的shapes，具体逻辑如下：
- 从 args 得到输入 input_tensors；
- 遍历 input_tensors，找出第一个不为空的张量，获取这个张量的 batch size，赋给 mini_batch_size；
- 从参数中得到 micro_batch_size，设置到 micro_batch_size；
- 如果 micro_batch_size 没有意义，则：
  - 如果 p.num_micro_batches 大于 mini_batch_size，则 p.num_micro_batches 为 mini_batch_size；
  - 把 micro_batch_size 设置为 mini_batch_size // p.num_micro_batches；
- 建立一个 input_shapes 集合，遍历 input_tensors，对于每个张量，得到其shapes列表 input_shape，并且设置 input_shape 的 batch_dim 为 micro_batch_size；
- 如果设置了 p.nested_map_fprop，则把 input_shapes 构建成一个递归嵌套的结构；
- 返回 input_shapes；

```python
  def _get_input_shapes(self, args):
    p = self.params
    if p.nested_map_fprop:
      assert len(args) == 1assert isinstance(args[0], py_utils.NestedMap)
      input_tensors = py_utils.Flatten(args[0])
    else:
      input_tensors = _ToTuple(args)
    
    # 遍历 input_tensors，找出第一个不为空的张量，获取这个张量的 batch size，赋给 mini_batch_size# Get batch size from the first tensor which is not None.
    mini_batch_size = Nonefor input_tensor in input_tensors:
      if input_tensor is not None:
        mini_batch_size = input_tensor.get_shape().as_list()[p.batch_dim]
    assert mini_batch_size is not None
    micro_batch_size = p.micro_batch_size
    
    if not micro_batch_size: # 如果 micro_batch_size 没有意义# 如果 p.num_micro_batches 大于 mini_batch_size，则 p.num_micro_batches 为 mini_batch_sizeif p.num_micro_batches > mini_batch_size:
        p.num_micro_batches = mini_batch_size
      # 把 micro_batch_size 设置为 mini_batch_size // p.num_micro_batches  
      micro_batch_size = mini_batch_size // p.num_micro_batches
    if mini_batch_size is not None:
      if micro_batch_size  p.num_micro_batches != mini_batch_size:
        raise ValueError('micro_batch_size * num_micro_batches != batch_size.')

    # 遍历 input_tensors，对于每个张量，得到其shapes列表 input_shape，并且设置 input_shape 的 batch_dim 为 micro_batch_size
    input_shapes = ()
    for input_tensor in input_tensors:
      if input_tensor is not None:
        input_shape = input_tensor.get_shape().as_list()
        input_shape[p.batch_dim] = micro_batch_size
        input_shapes += (tf.TensorShape(input_shape),)
      else:
        input_shapes += (None,)

    # 如果设置了 p.nested_map_fprop，则把 input_shapes 构建成一个递归嵌套的结构    if p.nested_map_fprop:
      input_shapes = py_utils.Pack(args[0], input_shapes)
    return input_shapes
```

### 4.2.4 FProp
前向传播函数，用流水线方式在多个设备上运行多个 children cells。具体逻辑如下：
- 做一些准备工作，比如：
  - 得到 children cell个数；
  - 得到集群；
  - 得到 输入shapes，dtypes；
  - 利用 输入shapes 计算出 输出shapes；
- 遍历处理中间层：
  - 对于具体cell，把cell加入到累积层中，对于每个cell，得到对应的function；
  - 为后续的 StackedRecurrent 运行设置其初始状态；
  - 把cell_idx对应的设备加入到devices列表；
- 为流水线中间（去除头尾）的各个设备设定一些变量；
- 在第一个设备上执行如下操作：
  - 遍历_before_layers，运行每层的FProp，最终得到 previous;
  - 对于 previous 继续操作，构建出 inputs，即利用_StackAndSplit分割张量;
  - 为流水线后续设备设置其输入;
- 在中间设备上执行recurrent.StackedRecurrent操作 ;
- 在最后一个设备上把micro_batches的形状聚合，最终得到输出张量:
  - 如果嵌套，则返回最后一个形状；
  - 否则遍历输出，聚合各个输出的形状；

```python
  def FProp(self, theta, *args):
    """Run multiple cells in different devices in a pipelining manner.

    Args:
      theta: A NestedMap object containing weights' values of this layer and its
        children layers.
      *args: Non-keyworded variable length argument list of input tensors.

    Returns:
      A list of output tensors
    """# TODO(huangyp): handle optional None inputs.
    p = self.params
    if self.do_eval and self.cluster.num_devices_per_split == 1: # 如果设置了 do_eval 并且集群的 num_devices_per_split 为 1
      outputs = copy.copy(args)
      # 就直接串行执行for (name, l) in self._before_layers + self._cells:
        outputs = _ToTuple(outputs)
        outputs = l.FProp(theta[name], *outputs)
      return outputs

    num_cells = len(p.cell_tpl) # 得到 children cell个数
    cluster = self.cluster # 得到集群# Compute shapes of input and output tensors.# 得到 输入shapes，dtypes
    input_shapes = self._get_input_shapes(*args)
    state_dtype = self._get_state_dtype(*args)
    # 利用 输入shapes 计算出 输出shapes
    state_shapes = self._CalculateOutputShapes(input_shapes)
    tf.logging.info('state_shapes={}'.format(state_shapes))

    def GetCellFn(i): # 对于第 i 个层，返回一个对应的函数，这个函数将在 StackedRecurrent 内部执行"""Get the ith feature extraction layer."""def CellFn(theta, state0, inputs):
        """A cell fn is exectued inside of StackedRecurrent."""# 没有深入研究StackedRecurrent，只从此函数看，作用是利用cell.FProp计算输出，并且得到一个state，其中包括输出和micro batch tensordel state0

        def _FPropInputSetShape(name, t_shape): # 给输入设置shapeif t_shape is None:
            return None
          inputs[name].set_shape(t_shape.ToTensorShape().as_list())
          return inputs[name]

        if p.nested_map_fprop:
          # pylint: disable=protected-access
          fprop_inputs = state_shapes[i]._RecursiveMap(_FPropInputSetShape)
          # pylint: enable=protected-accesselse:
          fprop_inputs = []
          for input_idx, input_shape in enumerate(state_shapes[i]):
            name = 's{}'.format(input_idx)
            fprop_inputs.append(_FPropInputSetShape(name, input_shape))

        with py_utils.RemoveAssertContext(remove=True):
          with CellFnFPropOpReplacementWrapper():
            tf.logging.info('cell {} input {}'.format(i, fprop_inputs))
            mb_tensor = inputs[_MICRO_BATCH_STATE_NAME] # 得到输入的 micro batch tensor
            SetOverWriteGlobalStep(mb_tensor)
            _, cell = self._cells[i]
            fprop_inputs = _ToTuple(fprop_inputs)
            outputs = cell.FProp(theta, *fprop_inputs) # 计算输出if p.nested_map_fprop:
          assert py_utils.IsCompatible(outputs, state_shapes[i + 1])
          state1 = outputs.Filter(lambda x: x is not None)
        else:
          state1 = py_utils.NestedMap()
          outputs = _ToTuple(outputs)
          assert len(outputs) == len(state_shapes[i + 1])
          for output_idx in range(len(outputs)):
            if outputs[output_idx] is not None:
              name = 's{}'.format(output_idx)
              state1[name] = outputs[output_idx]
        state1[_MICRO_BATCH_STATE_NAME] = mb_tensor
        return state1, py_utils.NestedMap()

      return CellFn

    cell_fns = []
    accumulator_layers = [] # 为了梯度累积
    thetas = []
    init_states = []
    devices = []
    # 遍历，把cell_idx对应的设备加入到devices列表for cell_idx in range(num_cells): # 遍历 children cell
      cell_name, cell = self._cells[cell_idx] # 得到具体一个 cell
      accumulator_layers.append(cell) # 把cell加入到累积层中
      cell_fns.append(GetCellFn(cell_idx)) # 对于每个cell，得到对应的function
      thetas.append(theta[cell_name]) # 添加 theta# 返回一个带有形状t_shape的，类型为state_dtype的张量,并且所有元素都设为零.def _TfZeros(t_shape):
        if t_shape is None:
          return Nonereturn tf.zeros(t_shape.ToTensorShape().as_list(), dtype=state_dtype)

      # 为后续的 StackedRecurrent 运行设置其初始状态if p.nested_map_fprop:
        init_state = py_utils.Transform(_TfZeros, state_shapes[cell_idx + 1])
        init_state = init_state.Filter(lambda x: x is not None)
      else:
        init_state = py_utils.NestedMap()
        for output_idx, state in enumerate(state_shapes[cell_idx + 1]):
          state = _TfZeros(state)
          if state is not None:
            name = 's{}'.format(output_idx)
            init_state[name] = state
      init_state[_MICRO_BATCH_STATE_NAME] = tf.cast(0, dtype=state_dtype)
      init_states.append(init_state)

      # 把cell_idx对应的设备加入到devices列表
      devices.append(cluster.WorkerDeviceInModelSplit(cell_idx))

    # 为流水线中间（去除头尾）的各个设备设定一些变量
    cell_grads = [None] * num_cells
    cell_outs = [lambda x: x] * num_cells
    cell_out_grads = [lambda x: x] * num_cells

    # 在第一个设备上执行如下操作with tf.device(devices[0]): 
      previous = _ToTuple(args)
      for (name, l) in self._before_layers: # 遍历_before_layers，运行每层的FProp，最终得到 previous
        previous = l.FProp(theta[name], previous)
        previous = _ToTuple(previous)

      def _StackAndSplit(x): # 把张量分割成# Split tensors into microbatches.if x is None:
          return None# tf.split按照行或者列分割一个矩阵return tf.stack(tf.split(x, p.num_micro_batches, axis=p.batch_dim))

      # 对于 previous 继续操作，构建出 inputs，即利用_StackAndSplit分割张量if p.nested_map_fprop: # 嵌套情况，只选取previous[0]做处理
        inputs = py_utils.Transform(_StackAndSplit, previous[0]) #利用_StackAndSplit分割张量
        inputs = inputs.Filter(lambda x: x is not None)
      else: # 非嵌套
        inputs = py_utils.NestedMap()
        for output_idx, output_tensor in enumerate(previous): # 遍历第一层的输出
          output_tensor = _StackAndSplit(output_tensor) # 利用_StackAndSplit分割张量if output_tensor is not None:
            name = 's{}'.format(output_idx)
            inputs[name] = output_tensor
      gs_tensor = py_utils.GetGlobalStep()
      # 为流水线后续设备设置其输入
      inputs[_MICRO_BATCH_STATE_NAME] = tf.stack([
          tf.cast(gs_tensor  p.num_micro_batches + t, dtype=state_dtype)
          for t in range(p.num_micro_batches)
      ])
      
    # 在中间设备上执行操作    
    tf.logging.info('pipeline input = {}'.format(inputs))
    output_state, _ = recurrent.StackedRecurrent( 
        devices=devices,
        cell_fns=cell_fns,
        cell_grads=cell_grads,
        cell_outs=cell_outs,
        cell_out_grads=cell_out_grads,
        thetas=thetas,
        init_states=init_states,
        inputs=inputs,
        accumulator_layers=accumulator_layers,
        unused_acc_state=True)

    # 在最后一个设备上执行如下操作，最终得到输出张量with tf.device(devices[-1]):
      def _ReshapeRetVal(name, t_shape): # 把micro_batches的形状聚合，得到最终输出"""Restore shape for tensors in microbatches."""if t_shape is None:
          return None
        output_tensor = output_state[name]
        if p.batch_dim != 0:
          perm = list(range(1, p.batch_dim + 1)) + [0]
          perm += list(range(p.batch_dim + 1, t_shape.rank + 1))
          output_tensor = tf.transpose(output_tensor, perm=perm)
        output_shape = t_shape.ToTensorShape().as_list()
        output_shape[p.batch_dim] *= p.num_micro_batches
        output_tensor = tf.reshape(output_tensor, output_shape)
        return output_tensor

      # Construct the final return values from output_state.if p.nested_map_fprop: # 如果嵌套，则返回最后一个形状# pylint: disable=protected-access
        output_tensors = state_shapes[-1]._RecursiveMap(_ReshapeRetVal) # 聚合形状# pylint: enable=protected-accesselse:
        output_tensors = []
        # 遍历输出，聚合各个输出的形状for output_idx, state_shape in enumerate(state_shapes[-1]): 
          output_name = 's{}'.format(output_idx)
          output_tensor = _ReshapeRetVal(output_name, state_shape) # 聚合形状
          output_tensors.append(output_tensor)
        if len(output_tensors) == 1:
          output_tensors = output_tensors[0]
        else:
          output_tensors = tuple(output_tensors)
        
      tf.logging.info('pipeline output = {}'.format(output_tensors))
      return output_tensors
```

### 4.2.5 类定义
具体代码如下：

```python
class PipeliningLayer(SeqLayer):
  """Pipelining a sequence of layers on multiple devices."""  @classmethoddef Params(cls):
    p = super().Params()
    p.Define('num_micro_batches', 1, 'Number of micro batches.')
    p.Define('micro_batch_size', None, 'Size of a micro batch.')
    p.Define('batch_dim', 0, 'The batch dimension.')
    p.Define('state_dtype', None, 'Externally specify dtype for states.')
    p.Define(
        'nested_map_fprop', False, 'Whether arguments and returns of ''cell fprop functions are nested maps')
    return p

```
具体FProp函数逻辑如下图：
```
+--------------------------------------------------------------+
| FProp             _CalculateOutputShapes                     |
|                             +                                |
|                             |                                |
|                             |                                |
|                             v                                |
|                        state_shapes                          |
|                             +                                |
|                             |                                |
|                             |                                |
|                             |                                |
|                             v                                |
|                for cell_idx in range(num_cells):             |
|                             +                                |
|                             |                                |
|                             |                                |
|                             v                                |
|       devices.append(WorkerDeviceInModelSplit(cell_idx))     |
|                             +                                |
|                             |                                |
|                             |                                |
|                             v                                |
|                  with tf.device(devices[0])                  |
|                             +                                |
|                             |                                |
|                             |                                |
|                             v                                |
|             recurrent.StackedRecurrent(cell_outs)            |
|                             +                                |
|                             |                                |
|                             |                                |
|                             v                                |
|                 with tf.device(devices[-1])                  |
|                             +                                |
|                             |                                |
|                             |                                |
|                             v                                |
|                       output_tensors                         |
|                                                              |
+--------------------------------------------------------------+
Device流水线逻辑如下：
                   devices[0]
                       +
                       |
                       |
                       |
                       v
+----------------------+-------------------------+
|Pipeline                                        |
|                         devices[1]             |
|                             +                  |
|                             |                  |
|                             |                  |
|                             v                  |
|  cell_grads[1~n]        devices[2]             |
|                             +                  |
|  cell_outs[1~n]             |                  |
|                             |                  |
|  cell_out_grads[1~n]        v                  |
|                         devices[3]             |
|                             +                  |
|                             |                  |
|                             |                  |
|                             v                  |
|                         devices[4]             |
|                                                |
+----------------------+-------------------------+
                       |
                       |
                       |
                       v
                   devices[-1]

```
### 4.2.6 使用
源码中给出的例子是 GPipeBatchMajorTransformerStack，目前看来，继承PipeliningLayer即可。

```python
class GPipeBatchMajorTransformerStack(PipeliningLayer):
  """Stacked self- multi-head attention and fully connected layers.

  With optional layer normalization applied to the final output.

  See 'Attention Is All You Need' https://arxiv.org/abs/1706.03762for details. 

  Implements a gipe stack for the batch major transformer variant.
  """

```
GPipeBatchMajorTransformerStack 的 FProp 返回一个输出张量的列表，其中下面代码调用了PipeliningLayer的功能。

```python
logits = super().FProp(theta, source_input, source_paddings, target_input,
                       target_paddings, encoder_self_atten_segment_mask,
                       decoder_self_atten_segment_mask,
                       decoder_cross_atten_segment_mask, source_segment_pos,
                       target_segment_pos)
```

具体代码如下：

```python
  def FProp(self,
            theta,
            source_input,
            source_paddings,
            target_input=None,
            target_paddings=None,
            source_segment_id=None,
            target_segment_id=None,
            labels=None,
            label_weights=None,
            source_segment_pos=None,
            target_segment_pos=None):

    p = self.params
    if p.num_decoder_layers > 0:
      assert target_input is not Noneassert target_paddings is not None
      target_time = tf.shape(target_input)[1]
      batch = tf.shape(target_input)[0]
    encoder_self_atten_segment_mask = None
    decoder_self_atten_segment_mask = None
    decoder_cross_atten_segment_mask = None# Prepare segment masks from segment ids.if p.packed_input:
      dtype = py_utils.FPropDtype(p)
      assert source_segment_id is not None, (
          'Need to specify src_segment_id if packed input is supported.')
      assert source_segment_pos is not None, (
          'Need to specify src_segment_pos for packed input and embeddings.')
      encoder_self_atten_segment_mask = batch_major_attention.SegmentMask(
          source_segment_id, source_segment_id, dtype, False)
      if target_segment_id is not None:
        decoder_self_atten_segment_mask = batch_major_attention.SegmentMask(
            target_segment_id, target_segment_id, dtype, False)
        causal_padding = tf.expand_dims(
            tf.tile(
                tf.expand_dims(
                    batch_major_attention.CausalPadding(
                        target_time, dtype=dtype), 0), [batch, 1, 1]), 1)
        decoder_self_atten_segment_mask = tf.math.maximum(
            causal_padding, decoder_self_atten_segment_mask)
        decoder_cross_atten_segment_mask = batch_major_attention.SegmentMask(
            target_segment_id, source_segment_id, dtype, False)

    # FProp through the gpipe pipeline.# 这里调用了基类的PipeliningLayer，完成流水线操作。
    logits = super().FProp(theta, source_input, source_paddings, target_input,
                           target_paddings, encoder_self_atten_segment_mask,
                           decoder_self_atten_segment_mask,
                           decoder_cross_atten_segment_mask, source_segment_pos,
                           target_segment_pos)
            
    label_weights = tf.reshape(label_weights, [-1])
    target_probs = Noneif p.label_smoothing:
      target_probs = self.smoother.FProp(
          theta.smoother, target_paddings, labels, target_ids=None)
      target_probs = tf.reshape(target_probs, [-1, p.softmax_tpl.num_classes])
    reshaped_logits = tf.reshape(logits, [-1, p.softmax_tpl.num_classes])
    tgt_labels = tf.reshape(labels, [-1])
    num_splits = len(p.splits)
    softmax = self.children['cell_{}'.format(num_splits - 1)].softmax
    softmax_theta = theta['cell_{}'.format(num_splits - 1)].softmax
    per_example_xent, _ = softmax.XentLossFromLogits(
        softmax_theta,
        reshaped_logits,
        class_weights=tf.reshape(label_weights, [-1]),
        class_ids=tgt_labels,
        class_probabilities=target_probs)
    xent_shape = tf.shape(logits)[:2]
    per_example_xent = tf.reshape(per_example_xent, xent_shape)
    return per_example_xent, logits
```