# 实验二：搜索算法

课程：当代人工智能。完成四道题，并进行启发函数、搜索深度、访问顺序和场景变化的对照实验。

仓库地址：https://github.com/miracle7017/26fall-AI-homework

## 环境与快速运行

核心代码仅使用 Python 标准库，建议 Python 3.9 或以上。随包归档生成于 Python 3.12.14 / Linux x86_64，修订版在 Python 3.12.14 / Windows x86_64 独立复跑。

在终端进入本目录：

```bash
python main.py --test
python main.py --out new_results
```

第一条执行正确性检查。第二条运行四题与所有实验，将新结果写入 `new_results/`。省略 `--out` 会写入本目录 `results/` 并覆盖同名文件。运行无需联网。

```bash
python main.py --out new_results --repeats 7
```

`--repeats` 指计时批次数，至少4，默认7。首次先预热并选择批量大小，使一批约达到10毫秒，再计算各批单次平均耗时的中位数和IQR。实验总运行时间通常为数秒至数十秒，取决于设备。CSV用UTF-8 BOM编码，便于Excel打开。

## 目录结构

```text
experiment2/
├── README.md
├── main.py
├── build_report.py
├── requirements.txt
├── examples/
│   └── q1.json, q2.json, q3.json, q4_h2.json, q4_h4.json
└── results/
    ├── astar_metrics.csv, tree_metrics.csv, distance_matrix.csv
    ├── game_metrics.csv, grid_extension.csv, value_sensitivity.csv
    ├── results.json
    └── game_nodes.png, extensions.png
```

## 文件说明

| 文件 | 用途 |
|---|---|
| main.py | 手写A*、Minimax、Alpha-beta；输入、实验、验证和结果导出 |
| build_report.py | 从已生成的results.json绘制图表并生成五页报告 |
| requirements.txt | 仅重建报告时需要的可选第三方依赖 |
| examples/*.json | 四题可直接使用的输入示例 |
| results/results.json | 完整结果、最优路径、行动序列、精确根分支价值、性能和运行环境 |
| results/astar_metrics.csv | 题一、题二启发函数对比 |
| results/tree_metrics.csv | 题三顺序与剪枝对比 |
| results/distance_matrix.csv | 题四6×6最小代价矩阵 |
| results/game_metrics.csv | 两个深度、两种算法、四种顺序的16组实验 |
| results/grid_extension.csv | 30×30地图：3种密度×10个种子×2种启发函数 |
| results/value_sensitivity.csv | T3四种价值×两个深度的敏感性实验 |
| results/*.png | 报告采用的对比图 |

## 单题输入输出

函数接口满足题目要求，路径中的坐标为(row,col)，从0计数：

```python
from main import solve_maze, solve_weighted, alpha_beta, TreeNode
path = solve_maze(start, end, grid)       # List[Tuple[int,int]]，无解[]
cost = solve_weighted(start, end, grid)   # int，无解-1
value = alpha_beta(root)                 # 根节点为MAX的最佳值
```

命令行也可以只输出某一题的答案，不混入实验日志：

```bash
python main.py --task 1 --input examples/q1.json
python main.py --task 2 --input examples/q2.json
python main.py --task 3 --input examples/q3.json
python main.py --task 4 --input examples/q4_h2.json
python main.py --task 4 --input examples/q4_h4.json
```

题1/2 JSON为 `{"start":[0,0],"end":[3,4],"grid":[...]}`；题3为课件的tree_structure嵌套结构；题4输入为 `{"depth":2}` 或 `{"depth":4}`，地图和价值采用题定常量。JSON不支持元组，命令行路径以数组输出；Python函数接口返回元组坐标。

对应预期结果：题1任意合法7步路径；题2为7；题3为5；题4的H=2为T2/-1，H=4为T3/3。题4cost为浮点形式时数值仍相同，例如-1.0。

## 重要规则与实现选择

A* 使用最小堆，主键 f=g+h，同 f 按入队序号处理。邻居固定为上、右、下、左。更新更优 g 时重新入队，允许重新打开，跳过旧条目。非矩形地图报错，越界或障碍端点返回无解。有效起点兼终点时，成本为0，扩展数、生成数、堆峰值均为1。

课件第5–7页给出整数地形示例并要求返回 `int`。`solve_weighted` 及题二 JSON 接口据此采用有限非负整数权重，接受整数值及整数数字字符串（如 `3`、`3.0`、`"3"`、`"3.0"`）。非整数、负数及非有限权重抛出 `ValueError`。起点初始成本和终点进入成本均为0，支持原有地图标记，输入地图保持原样。

题一普通格和终点进入成本均为1，安全h等于曼哈顿距离。题二起点初始成本为0，进入E为0；安全h为 `m*max(D-1,0)+c(E)`（非终点），终点h=0。m为非终点格的最小进入权，支持零权。对于题定整数正权，m=1，题二h=max(D-1,0)。`heuristic='raw'`仅作诊断，免费终点下不保证可采纳，正式结果不使用。

题三按JSON中的完整树结构，右侧MIN也包含[6,9]子树；内容相同不自动缓存。默认从左到右；仅对照时反向。叶子value支持负值，非叶子按层交替MAX/MIN；不要求完全二叉树。

题四A/B/宝藏格成本为1；每次只收集选定宝藏，经过其他宝藏不收集，另一玩家不挡路；拿走宝藏不改变地图。状态是不可变State，记录双方位置、剩余宝藏、双方累计收益和当前玩家。深度一次行动减1；收益可为负，不擅自增加跳过行动选项。默认natural顺序T1至T4；gain按当前玩家即时净收益降序，bad_gain升序；相同收益按名称排序，最终同值保留先访问动作。若自设不连通地图导致所有宝藏均不可达，题目没有定义跳过或终止规则，博弈函数会明确报错，不虚构规则。距离矩阵计算本身支持无解并返回-1。

距离矩阵预计算所有36个有序位置对，没有假设一般加权图必然对称。博弈计时只测预计算后的搜索，含状态构造和排序，排除距离预计算、I/O和绘图。题三计时含小树构建。trace/行动序列仅用于解释；所有首步精确价值由独立无剪枝搜索得到，不把剪枝界当作完整根分支精确值。

## 指标定义与对照

- expanded：有效弹出并处理的A*节点数，包含目标；不含过时堆条目。
- generated：起点与每次成本改善后的入队次数，非唯一坐标数。
- peak_open：堆条目峰值，含尚未清理的旧条目；仅是空间开销代理。
- nodes / leaves：博弈递归调用数 / 截断或终局评估数。
- cutoffs：α≥β且确实存在未访问兄弟时的截断事件数，不等于剪掉节点总数。
- median_us / iqr_us：各批单次耗时的中位数 / 四分位距，单位微秒。批数记录在 `results.json` 的 `environment.repeats`，报告据此显示。

A*对比保持地图及平局规则一致；博弈对比保持深度和评价函数一致。耗时不用于跨机器排名。小样例的微秒差值不应过度解读。

## 创新、拓展与验证

1. 根据终点免费规则修正启发函数，并保留h=0基线；零权情况下自动退化为安全下界。
2. Alpha-beta增加即时净收益排序，和自然、反向、坏收益顺序比较；排序不调用完整子树搜索，计时包含排序成本。
3. 30×30地图，障碍概率0/15%/30%，种子2026至2035，固定起终点。同一种子在三个概率下复用逐格随机数，障碍集合随阈值升高而嵌套。逐图比较最优成本和扩展数，同时报告含无解图的全样本均值与仅可解样本均值。图中误差线为最小至最大范围。
4. 仅改变T3价值为3/7/11/15，比较H=2/H=4的首步与收益差。
5. 原归档586组包括样例、边界、100张随机图的 Bellman-Ford 对照（两种终点计费规则）、36项距离、50棵随机树、30组随机价值的独立行动排列对照。修订版 `--test` 和全量复跑各通过588组，新增单可通行格的重合端点回归，以及整数权重契约检查。每组可含多条断言。

无障碍图采用 FIFO 平局，h=0 与曼哈顿均扩展900节点。

H=4 时，完整 Minimax 给出的最优应对与原归档首步、收益差一致。T3 价值为3时，A:T2(+2) → B:T1(+1) → A:T4(+1) → B:T3(-3)，两人累计收益为3和-2，收益差为5。T3 价值为7时，A:T3(-1) → B:T1(+1) → A:T2(0) → B:T4(-5)，累计收益为-1和-4，收益差为3。T3的归属由B转到A，末步由B取T3改为取T4，A的累计收益比原来少4，B少2。

## 重建图表与报告（可选）

```bash
python -m pip install -r requirements.txt
python build_report.py
```

报告从随包 `results/results.json` 读取原归档数据，在内存中绘制图表。PDF 默认写到上一级 `experiment2_report.pdf`，也可通过 `--output` 指定路径。交付版通过 `--font` 嵌入微软雅黑 TrueType 字体子集。若省略该参数，使用未内嵌的 STSong-Light。

```bash
python build_report.py --name "姓名" --student-id "学号" --class-name "班级" --date "YYYY.MM.DD" --font "C:/Windows/Fonts/msyh.ttc"
```

省略身份参数时，首页显示相应占位符。
