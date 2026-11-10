"""从 results/results.json 构建五页报告和图表。先运行 main.py。"""
import argparse
import json
from io import BytesIO
from itertools import accumulate
from xml.sax.saxutils import escape
import statistics
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.pdfbase.ttfonts import TTFont

from main import BOARD, MAZE, POINTS, VALUES, WEIGHTED, game_search, sample_tree, validate

ROOT = Path(__file__).parent
D = json.loads((ROOT / 'results' / 'results.json').read_text(encoding='utf-8'))
batches = D['environment']['repeats']

parser = argparse.ArgumentParser(description=__doc__)
for field in ('name', 'student-id', 'class-name', 'date'):
    parser.add_argument('--' + field)
parser.add_argument('--font', type=Path, help='用于内嵌的中文 TrueType 字体（TTF 或 TTC）')
parser.add_argument('--output', type=Path, default=ROOT.parent / 'experiment2_report.pdf')
identity = parser.parse_args()

def saved_figure(fig):
    """Render the PDF figure without overwriting the archived PNG."""
    buffer = BytesIO()
    fig.tight_layout()
    fig.savefig(buffer, format='png')
    plt.close(fig)
    buffer.seek(0)
    return buffer

def reduction(baseline, candidate):
    return f'{(1 - candidate / baseline) * 100:.1f}%'

game = {(r['depth'], r['algorithm'], r['order']): r for r in D['game']}
settings = [('Minimax', 'natural'), ('AB', 'natural'), ('AB', 'reverse'),
            ('AB', 'gain'), ('AB', 'bad_gain')]
depths = sorted({r['depth'] for r in D['game']})
densities = sorted({r['density'] for r in D['extra']})
pairs = {}
for row in D['extra']:
    pairs.setdefault(row['density'], {}).setdefault(row['seed'], {})[row['heuristic']] = row

def summarize_maps(density, reachable_only=False):
    cases = list(pairs[density].values())
    reachable = [pair for pair in cases if pair['zero']['cost'] >= 0]
    sample = reachable if reachable_only else cases
    zero = [pair['zero']['expanded'] for pair in sample]
    safe = [pair['safe']['expanded'] for pair in sample]
    return len(reachable), len(cases), statistics.mean(zero), statistics.mean(safe)

def route(path):
    return ' → '.join(str(tuple(point)).replace(' ', '') for point in path)

# 图的轴标签使用英文，避免依赖本机中文字体；报告正文使用 PDF 内置中文字体。
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':160})
BLUE='#2c6385'; TEAL='#378c85'; GRAY='#8b98a0'; ORANGE='#be7936'
fig,axes=plt.subplots(1,2,figsize=(9.6,2.8))
for ax,h in zip(axes,(2,4)):
 rows=[game[h, algorithm, order] for algorithm, order in settings]
 labels=['Minimax','AB natural','AB reverse','AB gain','AB bad gain']
 vals=[r['nodes'] for r in rows]
 ax.bar(range(5),vals,color=[GRAY,BLUE,BLUE,TEAL,ORANGE],width=.65)
 for i,v in enumerate(vals):ax.text(i,v+.6,str(v),ha='center',fontsize=9)
 ax.set_xticks(range(5),labels,rotation=25,ha='right',fontsize=8)
 ax.set_ylim(0,max(vals)*1.18);ax.set_ylabel('Visited nodes');ax.set_title(f'H = {h}')
game_figure = saved_figure(fig)
fig,axes=plt.subplots(1,2,figsize=(9.6,2.8))
for h,col in [('zero',GRAY),('safe',TEAL)]:
 means=[];low=[];high=[]
 for density in (0,.15,.30):
  vals=[r['expanded'] for r in D['extra'] if r['density']==density and r['heuristic']==h]
  avg=statistics.mean(vals);means.append(avg);low.append(avg-min(vals));high.append(max(vals)-avg)
 x=[i+(-.17 if h=='zero' else .17) for i in range(3)]
 axes[0].bar(x,means,width=.3,color=col,label=h,yerr=[low,high],capsize=3)
axes[0].set_xticks([0,1,2],['0%','15%','30%']);axes[0].set_xlabel('Obstacle probability');axes[0].set_ylabel('Expanded nodes');axes[0].set_ylim(0,1020);axes[0].legend(fontsize=8)
for h,col in [(2,BLUE),(4,TEAL)]:
 rows=[r for r in D['sensitivity'] if r['depth']==h]
 axes[1].plot([r['T3_value'] for r in rows],[r['best_value'] for r in rows],marker='o',color=col,label=f'H={h}')
 for r in rows:axes[1].annotate(r['best_target'],(r['T3_value'],r['best_value']),xytext=(0,6 if h==4 else -13),textcoords='offset points',ha='center',fontsize=8)
axes[1].set_xticks([3,7,11,15]);axes[1].set_xlabel('Value of T3');axes[1].set_ylabel('Optimal score difference');axes[1].set_ylim(-3,13);axes[1].axhline(0,color='#cccccc',linewidth=.7);axes[1].legend(fontsize=8)
extension_figure = saved_figure(fig)

if identity.font:
    font_name = 'ReportCJK'
    pdfmetrics.registerFont(TTFont(font_name, str(identity.font)))
else:
    font_name = 'STSong-Light'
    pdfmetrics.registerFont(UnicodeCIDFont(font_name))
style=ParagraphStyle('body',fontName=font_name,fontSize=10.2,leading=15.5,spaceAfter=7,textColor=colors.HexColor('#26343c'),wordWrap='CJK')
heading=ParagraphStyle('heading',parent=style,fontSize=14,leading=20,spaceBefore=5,spaceAfter=10,textColor=colors.HexColor('#24526d'))
small=ParagraphStyle('small',parent=style,fontSize=8.6,leading=12.5,spaceAfter=6)
title=ParagraphStyle('title',parent=style,fontSize=19,leading=25,spaceAfter=10)
story=[]
def p(text,kind=style):story.append(Paragraph(text,kind))
def h(text):p(text,heading)
def table(rows,widths=None):
 rows=[[Paragraph(str(x),small) for x in row] for row in rows]
 t=Table(rows,colWidths=widths,hAlign='LEFT',repeatRows=1)
 t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e9f0f4')),('VALIGN',(0,0),(-1,-1),'TOP'),('LINEBELOW',(0,0),(-1,0),.6,colors.HexColor('#9bb1bf')),('LINEBELOW',(0,-1),(-1,-1),.5,colors.HexColor('#b8c6cf')),('LEFTPADDING',(0,0),(-1,-1),6),('RIGHTPADDING',(0,0),(-1,-1),6),('TOPPADDING',(0,0),(-1,-1),4),('BOTTOMPADDING',(0,0),(-1,-1),2)]))
 story.append(t);story.append(Spacer(1,8))
def page():story.append(PageBreak())
def num(x):return str(int(x)) if int(x)==x else f'{x:.2f}'

p('搜索算法实验报告',title)
p('GitHub：<link href="https://github.com/miracle7017/26fall-AI-homework" color="#24526d">https://github.com/miracle7017/26fall-AI-homework</link>',small)
p('　'.join(f'{label}：{escape(value or placeholder)}' for label, value, placeholder in [
    ('姓名', identity.name, '<姓名>'), ('学号', identity.student_id, '<学号>'),
    ('班级', identity.class_name, '<班级>'), ('日期', identity.date, '<YYYY-MM-DD>'),
]), small)
p('当代人工智能 | 实验二 | A* [1]、Minimax 与 Alpha-beta [2]')
h('1  实验目标、设计与评价口径')
p('四题分别考察迷宫路径、加权地形成本、博弈树回传和宝藏竞争。对照实验改变启发函数、访问顺序、搜索深度、障碍概率或单个宝藏价值。')
p('A* 以网格坐标为状态，维护最优已知成本 g、剩余成本下界 h 和 f=g+h。最小堆按 (f,入队序号) 排序，邻居固定按上、右、下、左访问。发现更小的 g 时更新父节点并重新入队；弹出旧成本条目时跳过。终点从堆中有效弹出后回溯路径。此实现不永久封闭节点，允许改进路径重新打开。')
p('博弈搜索在 MAX 层取最大值、MIN 层取最小值。Alpha-beta 维护 MAX 已有下界 α 与 MIN 已有上界 β；当 α≥β 时，当前分支剩余兄弟节点不可能改变祖先决策，停止搜索。基础实验遵循题目顺序；反向与收益排序只作为单独的对照设置。')
table([
    ['指标', '统计口径'],
    ['A* 扩展数 / 生成数', '扩展数含终点有效弹出，排除过时条目。生成数含起点及每次改进入队。'],
    ['博弈访问数 / 叶子数', '每次递归调用计一个节点。达到深度或无剩余宝藏时计一次叶子评估。'],
    ['剪枝 / 堆峰值', 'cutoffs 统计确实跳过剩余兄弟的事件。peak_open 为堆条目峰值，含旧条目。'],
    ['时间', f'预热后自适应批量运行，每批约10 ms。取{batches}批单次耗时的中位数及IQR。'],
], [105, 410])
p(f"计时采用原归档数据。Python {D['environment']['python']}，"
  f"{escape(D['environment']['platform'])}，单进程，{D['environment']['timer']}。", small)
def size(grid):
    return f'{len(grid)}×{len(grid[0])}'

table([['题目常量','输入与计费'],
       ['MAZE / WEIGHTED', f'{size(MAZE)} 迷宫，入格成本为1。{size(WEIGHTED)} 加权图，进入 E 免费。'],
       ['BOARD', f'{size(BOARD)} 竞争地图，进入 A、B 或宝藏格成本为1。'],
       ['POINTS', '，'.join(f'{name}={route([point])}' for name, point in POINTS.items())],
       ['VALUES', '，'.join(f'{name}={value}' for name, value in VALUES.items())]], [105,410])
h('2  题目一：迷宫最短路径')
p('每次移动成本为1，使用 h=|r-rE|+|c-cE|。每次最多缩短一个单位的行列差，因此 h 不超过真实剩余步数；障碍只会增加必要绕行。父节点回溯得到路径：')
q1 = D['q1']
p(f"{route(q1['path'])}。共{len(q1['path'])}个坐标、{num(q1['cost'])}次移动，符合样例。无解返回[]，起终点相同时返回单坐标列表。")
a=D['astar'];table([['设置','最优步数','扩展数','生成数','堆峰值']]+[[r['setting'],num(r['cost']),r['expanded'],r['generated'],r['peak_open']] for r in a[:2]],[115,100,100,100,100])
q1_zero, q1_safe, q2_zero, q2_safe = (r['expanded'] for r in a)
p(f'曼哈顿启发使扩展数由{q1_zero}降至{q1_safe}，减少{reduction(q1_zero, q1_safe)}。'
  '它优先处理更有希望的方向，仍需探索障碍附近的候选节点。', small)
page()
h('3  题目二：地形成本与启发函数修正')
q2 = D['q2']
p(f"按进入目标格的权值计费，起点初始 g=0。E 的进入成本为0。最优路线为 {route(q2['path'])}。")
entered = q2['path'][1:]
step_costs = [0 if [r,c] == q2['path'][-1] else int(WEIGHTED[r][c]) for r,c in entered]
table([['进入位置'] + [route([point]) for point in entered[:-1]] + ['E'],
       ['单步成本'] + step_costs, ['累计成本'] + list(accumulate(step_costs))],
      [83] + [432 / len(entered)] * len(entered))
p(f"路径共{len(entered)}步，总成本为{num(q2['cost'])}，无解返回-1。"
  '题二输出为 int，接口接受有限非负整数值及整数数字字符串，遇到非整数权重报错。')
p('与 E 相邻时，普通曼哈顿距离 h=1，真实剩余成本为0。raw 保留作高估诊断，正式对照使用 zero 与 safe。')
p('采用 h(n)=m×max(D(n,E)-1,0)+c(E)（n≠E），h(E)=0。其中 m 为非终点可通行格的最小进入成本；该下界至少计入 D-1 次普通进入和最后一次进入终点。本题 m=1、c(E)=0，故 h=max(D-1,0)。若允许零权，m=0 仍保持安全，但方向信息可能退化。该公式对普通相邻边及最后进入 E 的边均满足一致性。')
table([['设置','最优成本','扩展数','生成数','堆峰值']]+[[r['setting'],num(r['cost']),r['expanded'],r['generated'],r['peak_open']] for r in a[2:]],[115,100,100,100,100])
p(f"修正启发保持最优成本{num(q2['cost'])}，扩展数由{q2_zero}降至{q2_safe}，减少{reduction(q2_zero,q2_safe)}。"
  '昂贵地形会提高 g，因此步数较多的路线也可能总成本更低。')
h('4  题目三：数值回传与剪枝')
tree = sample_tree()
visited = {tuple(row['path']) for row in D['q3']['trace']}
groups = [[max(leaf.value for leaf in branch.children) for branch in node.children]
          for node in tree.children]
tree_rows = [['层 / 分支','左 MIN 分支','右 MIN 分支'],
             ['根 MAX', f"回传 {D['q3']['value']}", '取两个 MIN 回传值的较大者'],
             ['MIN'] + [min(group) for group in groups],
             ['MAX（完整回传）'] + ['，'.join(map(str, group)) for group in groups]]
leaf_rows = []
for i, node in enumerate(tree.children):
    parts = []
    for j, branch in enumerate(node.children):
        labels = [str(leaf.value) + ('（剪）' if (i,j,k) not in visited else '')
                  for k, leaf in enumerate(branch.children)]
        parts.append('[' + '，'.join(labels) + ']')
    leaf_rows.append(' → '.join(parts))
table(tree_rows + [['叶子（从左至右）'] + leaf_rows], [120,197.5,197.5])
p('左右 MIN 各含一棵叶值相同的子树，按各自分支独立计算。上表 MAX 值为完整 Minimax 回传值，“剪”标出默认 Alpha-beta 未访问的叶子。',small)
p('左到右 Alpha-beta 中，左侧 MIN 已得到5；进入其第二个 MAX 时 β=5，先读到6便有 α≥β，因此跳过9。右侧的 [6,9] 仍需访问，不能因为其内容相同就算作已搜索。默认共评估7个叶子。')
table([['算法','顺序','最佳值','访问节点','叶子评估','有效剪枝']]+[[r['algorithm'],r['order'],r['value'],r['nodes'],r['leaves'],r['cutoffs']] for r in D['tree']],[85,85,75,90,90,90])
p('反向访问剪掉另一处叶子，仍访问14个节点、评估7个叶子。')
page()
h('5  题目四：距离预计算与博弈建模')
p('用 A* 对 A、B、T1、T2、T3、T4 的全部有序位置对独立寻路。A、B、T 所在格进入成本均为1，区别于题二免费终点。得到距离矩阵如下（行是出发点，列是目的地）：')
names=list(D['distances']);table([['起点 / 终点']+names]+[[a]+[num(D['distances'][a][b]) for b in names] for a in names],[95]+[70]*6)
p('本矩阵对称不是任意进入权图的普遍性质：这里所有端点成本相同，反向走同一路径的成本相同。预计算36项后，博弈只查询距离；矩阵成本不计入后续博弈搜索的计时。')
p('状态为 (posA,posB,remaining,scoreA,scoreB,player)，以不可变数据类表示。选择 Ti 后，仅移动当前玩家、累计 Vi-d(pos,Ti)、移除该宝藏并切换玩家。经过其他宝藏不收集，已取位置保持通行，对手不挡路。每回合必须选一个宝藏，即使收益为负。搜索深度按单次行动计，截断或取完时返回 U=scoreA-scoreB。')
p('对各首步分别进行不剪枝搜索，可得到精确根分支价值，避免把剪枝产生的界误写成所有分支的精确分数：')
table([['深度','首选T1','首选T2','首选T3','首选T4','最优首步']]+[[h]+[num(D['root_values'][str(h)][t]) for t in ('T1','T2','T3','T4')]+[D['lines'][str(h)]['best_target']] for h in (2,4)],[65,90,90,90,90,90])
shallow, full = D['lines']['2'], D['lines']['4']
first, reply = shallow['line']
def gain_formula(action):
    target = action['target']
    cost = D['distances'][action['player']][target]
    return f"{VALUES[target]}-{num(cost)}={num(action['gain'])}"

alternative = full['best_target']
alternative_gain = VALUES[alternative] - D['distances']['A'][alternative]
alternative_value = D['root_values']['2'][alternative]
reply_gain = alternative_gain - alternative_value
reply_targets = [t for t in VALUES if t != alternative
                 and VALUES[t] - D['distances']['B'][t] == reply_gain]
p(f"H=2 时，A取{first['target']}的收益为{gain_formula(first)}。B取{reply['target']}的收益为{gain_formula(reply)}，"
  f"U={num(shallow['best_value'])}。若A先取{alternative}，收益为{num(alternative_gain)}，"
  f"B可取{reply_targets[0]}获{num(reply_gain)}，U={num(alternative_value)}，故浅层选择{shallow['best_target']}。")
sequence = '、'.join(f"{r['player']}:{r['target']}({r['gain']:+g})".replace('(+0)', '(0)') for r in full['line'])
score_a, score_b = (sum(r['gain'] for r in full['line'] if r['player'] == player) for player in ('A','B'))
p(f"H=4 的一条最优应对序列是 {sequence}。A累计{num(score_a)}，B累计{num(score_b)}，U={num(full['best_value'])}。"
  f"首步虽亏{num(-full['line'][0]['gain'])}，却占据B附近收益较高的{alternative}，也改变了后续分配。")
outcomes = '、'.join(f"H={depth} 时 {{{D['lines'][str(depth)]['best_target']},{num(D['lines'][str(depth)]['best_value'])}}}" for depth in depths)
p(f'两种算法均得到 {outcomes}。H=2 的评价停在中途，遗漏后两步的移动与强制选取成本。H=4 搜索完整博弈。')
h('6  排序优化的实现')
p('自然顺序为T1至T4；反向顺序相反；gain 按当前行动者 Vi-d 降序；bad_gain 按其升序。B也按自己的即时收益降序，因为B收益增加会降低U。排序仅使用当前状态，不预先搜索整棵树；排序开销包含在计时内。相同价值按访问先后保留首个最优动作，故一般情况下动作可能并列，价值应保持不变。')
page()
h('7  搜索效率：结构开销与运行时间')
story.append(Image(game_figure,width=515,height=150))
p('图1  两个深度分别使用从0开始的纵轴，柱顶标实际节点数；各面板纵轴上限不同，跨深度比较以数字为准。AB=Alpha-beta；gain为即时收益降序。',small)
rows=[['深度 / 设置','访问 / 叶子','较Minimax减少','时间中位数μs','IQR μs']]
for depth in (2,4):
 for algorithm,order in settings:
  r=game[depth,algorithm,order]
  base=game[depth,'Minimax','natural']['nodes']
  rows.append([f'{depth} / {algorithm} {order}',f"{r['nodes']} / {r['leaves']}",f"{(1-r['nodes']/base)*100:.1f}%",f"{r['median_us']:.2f}",f"{r['iqr_us']:.2f}"])
table(rows,[155,90,90,95,85])
m4, natural4, reverse4, gain4, bad4 = (game[(4,) + key] for key in settings)
p(f"完整四回合 Minimax 访问{m4['nodes']}个节点、{m4['leaves']}个叶子。自然顺序 AB 访问{natural4['nodes']}个节点，"
  f"较基线减少{reduction(m4['nodes'],natural4['nodes'])}。收益排序 AB 访问{gain4['nodes']}个，较基线减少"
  f"{reduction(m4['nodes'],gain4['nodes'])}，较自然 AB 再减少{reduction(natural4['nodes'],gain4['nodes'])}。"
  f"叶子评估从{m4['leaves']}减至{gain4['leaves']}。较早收紧界可跳过更多分支。")
p(f"坏收益顺序访问{bad4['nodes']}个节点，多于自然顺序的{natural4['nodes']}个。反向访问{reverse4['nodes']}个，与自然顺序接近。"
  f"四种顺序下 Minimax 均访问{m4['nodes']}个节点，排序增加了额外操作。")
p(f"H=2 时 Minimax、自然 AB 和收益排序 AB 分别访问{game[2,'Minimax','natural']['nodes']}、"
  f"{game[2,'AB','natural']['nodes']}、{game[2,'AB','gain']['nodes']}个节点。H=4 在本例改变了首步，也增加搜索开销。"
  "同一深度下，排序只影响搜索开销，保留最优收益差。")
rows=[['A*任务 / 设置','中位数 μs','IQR μs']]+[[r['task']+' / '+r['setting'],f"{r['median_us']:.2f}",f"{r['iqr_us']:.2f}"] for r in D['astar']]
table(rows,[255,130,130])
p('计时包含函数内状态构造、堆操作或排序，排除打印与磁盘输出；题三计时含小树构建。样例耗时仅为微秒量级，不把微小差值解释为稳定倍数优势。IQR记录批次间波动，原始CSV保留批量大小。节点减少也不保证耗时按同一比例下降，因为启发计算和排序本身有成本。',small)
page()
h('8  自设场景：边界、泛化与局限')
story.append(Image(extension_figure,width=470,height=137))
p('图2  左：30×30地图，各密度10张，柱为扩展数均值，误差线为最小至最大范围（非置信区间），纵轴从0开始；右：只改变T3价值，标注每个深度的最优首步。',small)
map_rows = [['障碍概率','可达数 / 总数','h=0均值','安全h均值','均值减少']]
for density in densities:
    reachable, total, zero, safe = summarize_maps(density)
    map_rows.append([f'{density:.0%}', f'{reachable} / {total}', f'{zero:.1f}'.removesuffix('.0'),
                     f'{safe:.1f}'.removesuffix('.0'), reduction(zero, safe)])
table(map_rows,[100,110,100,105,100])
p('种子为2026至2035。对同一种子，各概率复用逐格随机数，障碍集合随阈值升高而嵌套。起终点保持可通行，入格成本为1。两法逐图核对可达性和最优成本。表中全样本均值含无解图。')
empty = summarize_maps(0)
empty_cost = next(iter(pairs[0].values()))['zero']['cost']
p(f'空地图上各坐标的 g+h={num(empty_cost)}。FIFO 平局使目标最后弹出，两法均扩展{num(empty[2])}个节点。'
  '含障碍时，曼哈顿距离可以减少无关探索。终点不可达时，两法都会遍历起点所在的可达区域。')
for density in densities:
    failed = [f"{seed}（{pair['zero']['expanded']}/{pair['safe']['expanded']}）"
              for seed,pair in pairs[density].items() if pair['zero']['cost'] < 0]
    if not failed:
        continue
    reachable, _, zero, safe = summarize_maps(density, reachable_only=True)
    solved = [pair for pair in pairs[density].values() if pair['zero']['cost'] >= 0]
    ratios = [pair['zero']['expanded']/pair['safe']['expanded'] for pair in solved]
    worse = sum(ratio < 1 for ratio in ratios)
    smallest_failure = min(pair['zero']['expanded'] for pair in pairs[density].values()
                           if pair['zero']['cost'] < 0)
    context = f"最小的起点可达区域只有{smallest_failure}格。" if len(failed) > 1 else ''
    p(f"{density:.0%}组无解种子为{'、'.join(failed)}，括号内为 h=0/安全 h 的扩展数。{context}"
      f"仅可解的{reachable}张图中，扩展数均值从{zero:.1f}降至{safe:.1f}，降幅{reduction(zero,safe)}。"
      f"两者比值中位数{statistics.median(ratios):.3f}，范围{min(ratios):.3f}–{max(ratios):.3f}，"
      f"安全 h 扩展更多的图为{worse}张。", small)
sensitivity = {depth: [r for r in D['sensitivity'] if r['depth'] == depth] for depth in depths}
value_levels = '、'.join(str(r['T3_value']) for r in sensitivity[2])
sensitivity_outcomes = '。'.join(
    f"H={depth} 的收益差依次为" + '、'.join(num(r['best_value']) for r in rows)
    + '，对应首步为' + '、'.join(r['best_target'] for r in rows)
    for depth, rows in sensitivity.items())
p(f'固定地图，T3价值取{value_levels}。{sensitivity_outcomes}。', small)
for archived in sensitivity[4][:2]:
    value = archived['T3_value']
    exact = game_search(D['distances'], 4, pruning=False, values={**VALUES, 'T3': value})
    for key in ('best_target', 'best_value'):
        if exact[key] != archived[key]:
            raise ValueError(f'T3={value} 的最优应对与归档结果不符')
    sequence = ' → '.join(
        f"{r['player']}:{r['target']}({r['gain']:+g})" for r in exact['line']
    )
    score_a, score_b = (
        sum(r['gain'] for r in exact['line'] if r['player'] == player)
        for player in ('A', 'B')
    )
    p(f'T3={value}：{sequence}。A累计{num(score_a)}，B累计{num(score_b)}，'
      f"U={num(exact['best_value'])}。", small)
p('T3增值后归属从B转到A，末步由B取T3改为取T4。A累计收益下降4，B下降2，因此收益差从5降到3。', small)
h('9  正确性验证与文献')
archived_checks = D['validation']['assertion_groups_passed']
current_checks = validate(D['distances'])['assertion_groups_passed']
p(f'原归档通过{archived_checks}组，包括样例和边界、36项距离、100张随机图、50棵随机树及30组随机宝藏价值。'
  f'修订版通过{current_checks}组，增加单可通行格的重合端点回归和整数权重契约检查。', small)
p('[1] P. E. Hart, N. J. Nilsson &amp; B. Raphael (1968). A Formal Basis for the Heuristic Determination of Minimum Cost Paths. '
  'IEEE Transactions on Systems Science and Cybernetics, 4(2), 100-107. '
  '<link href="https://doi.org/10.1109/TSSC.1968.300136" color="#24526d">10.1109/TSSC.1968.300136</link>', small)
p('[2] D. E. Knuth &amp; R. W. Moore (1975). An Analysis of Alpha-Beta Pruning. Artificial Intelligence, 6(4), 293-326. '
  '<link href="https://doi.org/10.1016/0004-3702(75)90019-3" color="#24526d">10.1016/0004-3702(75)90019-3</link>', small)
p('课程课件：《当代人工智能实验课：实验二 - 搜索算法》，2026年10月15日，第3-15页题面、第16-17页报告要求与评分细则。', small)

def footer(c,doc):
    c.setFont(font_name, 8)
    c.setFillColor(colors.HexColor('#60737d'))
    c.drawString(40, 25, '当代人工智能 | 实验二：搜索算法')
    c.drawRightString(555, 25, f'{doc.page} / 5')
pdf = identity.output
doc=SimpleDocTemplate(str(pdf),pagesize=(595.28,841.89),rightMargin=40,leftMargin=40,topMargin=35,bottomMargin=42)
doc.build(story,onFirstPage=footer,onLaterPages=footer)
print(pdf)
