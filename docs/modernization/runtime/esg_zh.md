# ESG 原生迁移与训练区间图特征

ESG 已接入统一 train/evaluate/resume，支持点序列与完整网格原子数据。保留演化图、多尺度时间卷积、混合传播、残差/skip 和多步预测，不使用预测标签或真实未来观测作为模型输入。

## 原问题与修复

旧 static_feat 来自展开后训练窗口的第一历史帧，选中的时间范围随窗口、切分样本数和 static_feat_len 改变，含义不等同于原始训练区间。该 Tensor 固定在构造时的设备，没有注册 buffer；单节点 BatchNorm 会失败。每个 extractor 都创建从未使用的 residual_conv，零 hop 时还会执行与输出无关的图网络。动态图把节点状态重复为 B×N×N×D 并拼接成双倍宽度的节点对，产生可避免的中间数组。

新数据适配从原始训练区间起点选取明确前缀，默认使用全部训练区间，可用 data.graph.history_steps 限制原始时间步数；history_channels 默认全部目标通道，也可显式选择其中一部分。先补原始缺失输入为 0，再用只在训练区间拟合的目标 scaler 转换；不把日历或外部协变量用作图训练序列。

保留 LibCity 的「时间、选中目标通道、节点」顺序，展平成 `[训练步数×选中通道数,N]`，再交给两层核宽 10 的 Conv1d/BatchNorm、全连接/BatchNorm 节点特征编码器。多目标时保留这种标量交错含义，没有隐含改为另一种多通道卷积。图历史的起止、列、顺序、归一化及缺失处理记录在只读元数据中，样本窗口/预测步/训练 padding 不参与图历史选取。

非零图深度要求至少 19 行标量历史；时间序列须连续，存在缺口时可用 history_steps 选择首段连续前缀。零 hop 模式不创建图编码器及动态图参数，不要求这 19 行。历史作为模型独立 buffer 保存，跟随 device/dtype，安全 best.pt 可独立评估。完整续训继续校验原子文件版本和归一化/输入定义。

点数据使用 .dyna，网格数据使用 .geo 坐标和 .grid 或坐标式 .dyna，矩形及实际 ID 对齐由公共读取器检查。网格的历史全局特征按节点广播合入 X，点数据沿用公共历史布局；模型均接收 `[B,T,N,F]`。已知未来特征不参与本网络计算，不能把实际未来天气或目标混入历史。原始 NaN/0 目标保持，掩码依据 y_raw；补零只作用于历史输入。

原始缓存复用公共点/网格缓存；改变 history_steps/history_channels、窗口或预测长度时只重新准备相应图特征，不重读完整原子数组。graph.name=dynamic 不读取无关 .rel；原子来源清单也能正确记录只有 .grid 的连续窗口数据。缓存/续训只检查路径、大小与修改时间，不哈希源文件内容。

## 保留的模型计算

每个时间 extractor 使用多核膨胀时间卷积和 tanh/sigmoid 门，再按 dy_interval 分段更新动态图状态。状态更新保留原 GRU 风格 reset/update/candidate；两组独立的节点对网络分别产生有符号支持值与 sigmoid 门，图等于二者乘积。图不额外 softmax、不加自环、不擅自转为非负图。

传播保留接收节点为矩阵行、发送节点为列的 `A @ X` 方向，逐层使用 `alpha*X+(1-alpha)*propagated`，拼接 0～gcn_depth 各阶后用 1×1 卷积投影。层内图状态沿历史片段推进，每层/每次 forward 从编码后的训练节点特征初始化；不把前一批状态写入下一批。

将节点对第一层线性投影按两半权重分解为接收端投影与发送端投影再广播相加，保留 bias、ReLU 和第二层线性，数学含义与拼接相同。避免构造双倍宽度拼接输入；仍保留 B×N²×D 的中间非线性与稠密图成本，不声称算法变成稀疏或线性复杂度。

各 block 重新从 dilation=1 开始，各层/各 block 连续推进实际剩余时间长度；短输入只在左侧复制最早观测补到感受野。kernel_set 要求严格递增，conv_channels 可按核数均分，dy_interval 要求与层数一致，避免旧整数截断、核顺序和无效设置造成隐蔽形状错误。所有 skip 顺序相加，避免先拼接再归约；始终输出 `[B,H,N,O]`。

删除所有未使用 residual_conv；零 hop 删除不参与输出的静态/动态编码器。静态编码的 BatchNorm 复用单值回退，仅在统计样本数不足 2 时使用运行统计，不伪造重复节点。dropout、BatchNorm 状态和层归一化均遵从标准 train/eval。节点对公式和死层也对照了[原作者图模块](https://github.com/LiuZH-19/ESG/blob/master/model/graph.py)与[网络](https://github.com/LiuZH-19/ESG/blob/master/model/esg.py)，本批不宣称论文精度复现。

## 配置与使用

默认保留一个 block、每块三层、kernel_set=[2,6]、dilation_exp=1，32 个 residual/conv 通道、64 个 skip 通道、128 个 end 通道，静态/动态宽度 40/20，图深度 2、propalpha=0.05、dropout=0.3、dy_interval=[1,1,1]。可显式选择多 block、指数膨胀、不同 residual/conv 宽度、按节点位置的 LayerNorm affine，以及零 hop。图初始特征的设置属于 data.graph，结构参数留在 model.params。

默认训练为原尺度 masked_mae、mask_value=0，Adam 0.001/eps=1e-8/weight_decay=0.0001、梯度裁剪 5、早停 30、最多 300 轮。模型预设不改共享切分或预测窗；本网络默认没有额外调度器。

```powershell
python tools/make_runtime_fixture.py --directory ../esg-demo --model ESG
python run_model.py train --config ../esg-demo/esg.json --set runtime.exp_id=train
python run_model.py evaluate --config ../esg-demo/esg.json --checkpoint ../esg-demo/runs/train/checkpoints/best.pt --set runtime.exp_id=evaluate
python run_model.py train --config ../esg-demo/esg.json --resume ../esg-demo/runs/train/checkpoints/last.pt --set training.epochs=3 --set runtime.exp_id=resumed
```

示例使用 60 个小时、2×3 网格、两个目标和一个额外观测，较小的两层网络；图特征来自前 36 个原始训练时刻，展平为 [72,6]。新增 38 项专项通过，相关网格/配置合计 93 项通过；完整回归 1263 项通过，无失败/错误/跳过，见 [汇总](remaining-validation.json)。实际仓库外生成、训练、独立评估、续训 4 条 CLI 均成功，预测 [11,2,6,2]，两次预测/指标完全一致、第三轮累计 12 次更新、缓存 miss/hit/hit；原尺度 masked MAE 独立复算一致，纯权重带输入说明、独立评估无训练产物，见 [记录](esg-cli-validation.json)。

专项覆盖四组旧网络完整前向及所有有效参数/输入梯度与 BatchNorm 状态（默认宽度、多个 block、指数膨胀、不同通道、affine、零 hop）；独立逐节点对状态/得分与 NumPy 有符号方向传播；原始训练前缀/通道/限制/缓存/未来隔离、点与网格全局历史特征、原始零/NaN、物理节点置换、batch 独立/跨样本零梯度、buffer 所有权/float64/无标签无 RNG；点/网格和两种 scaler 的 dropout/BatchNorm/优化器/metric 调度/最佳模型/RNG/加载器精确恢复；真实尾批/单节点最小历史 CPU AMP、Windows spawn、缺口拒绝/显式连续前缀，以及 15 类启动配置校验。

本模型尚无真实城市精度、CUDA/MPS 专项认证。默认全训练前缀会增加节点编码器全连接层和权重文件大小，history_steps 可显式控制成本；没有通过自动截短改变默认训练数据含义。
