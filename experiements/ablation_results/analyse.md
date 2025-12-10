📊 消融实验结果汇总
实验配置	mAP@50:95	mAP@50	AP (small)	AP (medium)	AP (large)	AR@100
Baseline	0.697	0.966	0.348	0.698	0.740	0.783
Only Density Init	0.715	0.955	0.364	0.714	0.757	0.790
Only SOQB	0.686	0.961	0.346	0.710	0.732	0.786
Combined (Both)	0.702	0.966	0.347	0.698	0.747	0.783
📈 分析
你的判断是正确的！只添加 Density Init 确实是最好的配置。
1. 整体性能 (mAP@50:95):

Density Init: 0.715 ← 最高！
Combined: 0.702
Baseline: 0.697
SOQB: 0.686
2. 小目标检测 (AP small):

Density Init: 0.364 ← 最高提升！比 baseline 提升了 +1.6%
Combined: 0.347
SOQB: 0.346
Baseline: 0.348
3. 为什么 Combined 效果不如单独使用 Density Init:

组件冲突: Density Init 和 SOQB 可能在 Query 初始化方面存在一定的功能重叠或相互干扰
过度正则化: 两个模块同时作用可能导致过度约束，限制了模型的学习能力
SOQB 负作用: 从结果看 SOQB 单独使用时 (0.686) 甚至低于 baseline (0.697)，说明 SOQB 本身可能对该数据集不太适用
4. Density Init 的优势分析:

对小目标检测提升最为显著 (+1.6%)
中型目标提升 (+1.6%)
大型目标提升 (+1.7%)
整体 mAP 提升 +1.8%
✅ 结论
最佳选择: 只使用 Density Guided Query Init
SOQB 模块 在当前数据集上表现不佳，建议移除
不建议组合使用 两个模块，效果反而下降
这个结果说明 Density Guided Query Init 能够有效地根据目标密度分布来初始化 Query，使模型能更好地定位不同尺度的目标，尤其是对小目标的检测有明显提升。