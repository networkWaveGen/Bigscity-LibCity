# HIEST 原生迁移与逐样本层次图

HIEST 已接入统一 train/evaluate/resume。保留原节点、重叠区域与学习到的全局节点三级图、门控膨胀时间卷积、四步跨层增强及图重建/余弦辅助目标。模型为标准 nn.Module；forward 只接收历史输入并返回 [B,H,N,O]，归一化和损失由数据层与训练步负责。

## 原问题与修复

旧全局图把不同样本展平到同一个图中，且两次 permute/reshape 并不组成正确的节点特征 Gram；重建损失也存在相同的轴问题。新实现先将每个样本变成 [N,C×T]，独立计算 V@V.T。全局图仍按原算法执行 relu(Gram-0.5)、停止图生成梯度、行归一化与 softmax，随后供原共享 GCN 使用；保留正反两个方向，单支持类型只创建一个方向，修复原输入宽度不匹配。图计算留在 Tensor 设备上，不经过 NumPy 或复制到 CPU。

默认在无向连通关系上提取双连通分量：任一方向的邻接权重大于 region_threshold 即建立边，忽略自环。关节点可以同时属于多个区域，桥也形成区域，未覆盖的孤立节点各自形成单点区域。区域列按真实节点 ID 的稳定表示排序，改变 .geo 行顺序只改变节点轴，不改变区域身份。也可用 data.graph.region_groups 显式填写重叠的真实 ID 列表；要求组内无重复、组间不完全重复、覆盖全部节点且不含未知节点。

成员矩阵 M[N,R] 按列归一化，区域汇聚是 M.T@X，逆向广播保留成员权重，不改为逐行平均。M、原图支持和区域支持均注册独立 buffer；真实节点 ID、分区成员和规则写入只读元数据与输入说明。公共点数据的真实 ID/时间对齐修复已由独立的 commit56 交付，本批基于其正确读取结果建立分区。

图重建默认以二值 A>0 为目标；data.graph.reconstruction=weights 可显式使用 [0,1] 内权重，超出概率范围立即报错。区域目标为 M.T@A@M。重建从 sigmoid+BCE 改用数值稳定的 BCEWithLogits；低精度下 Gram、重建和余弦计算使用 float32。全局余弦仍为逐节点对的有符号平均，不取绝对值或平方，因此不宣称其数学最优点就是严格正交；一个全局节点时该项定义为连通梯度的零值。

训练主损失仍由原尺度预测和 y_raw 计算，辅助目标按每个样本的有效目标数量加权，再由公共循环按有效数量聚合；计数先在整数域完成，避免 BF16 将 257 等数量舍入。全缺测样本不贡献辅助项。固定特征下的微批目标/梯度与整批一致；训练态 BatchNorm/dropout 仍按正常批次执行，不声称换 batch_size 后训练轨迹相同。验证和独立评估只计算预测主损失，不用辅助目标挑选最佳权重。

旧输出把卷积剩余时间维当成 output_dim，新头明确输出 H×O 并恢复 [B,H,N,O]，全部预测步/目标通道受监督。自适应层数至少为 1，kernel_size=1 不执行除零公式，零 hop 只保留自身通道。删除从未使用的 residual_convs、非最终区域/全局 BN 及最后无效原节点 BN；普通 forward 跳过最后一层只为辅助损失服务的层次图，forward_with_aux 显式返回训练特征。保留原网络总层数为一时预测仅依赖时间 skip、最后图仅参与辅助目标的结构，没有悄悄新增图到预测头的连接。

普通/区域 BN 复用公共单值回退：每通道只有一个统计值时使用运行统计，保留真实尾批，不伪造重复样本。历史 NaN 在原尺度补 0 后再归一化，原始目标中的 NaN/0 保留给掩码；无穷观测拒绝。支持矩阵的六类变换与 HGCN 共用 normalized_supports，保留有向图转置规则和缩放拉普拉斯原 lambda_max=2 语义，HGCN 错误信息及数值回归保持。

## 配置与使用

默认 blocks=4、layers=2、apt_layer=true、kernel_size=2，residual/dilation 通道 32、skip=256、end=512、global_nodes=15，doubletransition、order=2、dropout=0.3，n1～n4=1。启用 apt_layer 时根据 input_window 和 blocks 推导实际层数；关闭后采用显式 layers。六种 adjtype 为 doubletransition、transition、symnadj、normlap、scalap、identity。

data.graph 默认使用任务的二值图，自带 region_threshold=0.8、region_groups=null、reconstruction=binary；切换图选择器后仍补齐 HIEST 自身默认参数。分区选项不污染原始缓存键，改变分区可复用原子数组。数据使用连续 window 与训练区间 standard scaler，模型预设不改共享切分/窗口/指标协议。默认 Adam 0.001、100 轮、裁剪 5、早停关闭，原尺度 masked_mae(0) 加 original_weight=0.01、regional_weight=0.01、orthogonal_weight=0.1。

```powershell
python tools/make_runtime_fixture.py --directory ../hiest-demo --model HIEST
python run_model.py train --config ../hiest-demo/hiest.json --set runtime.exp_id=train
python run_model.py evaluate --config ../hiest-demo/hiest.json --checkpoint ../hiest-demo/runs/train/checkpoints/best.pt --set runtime.exp_id=evaluate
python run_model.py train --config ../hiest-demo/hiest.json --resume ../hiest-demo/runs/train/checkpoints/last.pt --set training.epochs=3 --set runtime.exp_id=resumed
```

发布示例有 60 个小时、7 个非连续 ID 节点、2 个目标、两个重叠三角区域、桥和孤立节点，.geo 行故意打乱；自动得到 4 个区域。小网络使用一个 block、两层和 3 个全局节点。安全 best.pt 只有 Tensor 权重（含静态 buffer）及输入说明，last.pt 保存完整辅助训练、优化器、调度器、BN、RNG 和加载器状态；独立评估不创建训练产物。

## 验证与范围

新增 44 项专项测试，包含四组修正旧轴后的预测/全部有效参数和输入梯度（默认宽度、多目标、零 hop、单节点），独立逐样本重建/余弦公式及梯度，重叠分区/孤立节点/真实 ID 重排，六种支持矩阵与辅助训练，原始缓存/未来隔离/NaN/0，257/259 个有效目标的精确计数，跨样本推理隔离，两种 scaler 的 dropout/BN/Adam/metric 调度/早停/最佳模型/RNG/加载器精确续训，真实单样本尾批 CPU AMP、Windows spawn 和 14 类配置拒绝。与 HGCN、配置、公共节点身份相关测试合计 121 项通过，完整回归 **1325 passed，0 failed，0 errors，0 skipped**，见 [汇总](remaining-validation.json)。

仓库外四条实际命令全部成功，预测 [11,2,7,2]，独立预测/指标完全一致，第三轮累计 12 次更新；缓存 miss/hit/hit。原尺度 masked MAE 独立复算约 3.913424，与评估一致，分区/支持矩阵已随安全权重保存，见 [CLI 记录](hiest-cli-validation.json)。本批验证 CPU 功能和恢复语义，未宣称真实城市精度、CUDA/MPS 全模型认证；稠密原图/区域/全局图的平方空间成本仍保留。
