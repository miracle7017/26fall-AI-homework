"""实验二：A*、Minimax、Alpha-beta。核心求解仅依赖 Python 标准库。"""
import argparse
import csv
import heapq
import itertools
import json
import math
import platform
import random
import statistics
import time
from dataclasses import dataclass, asdict
from decimal import Decimal, InvalidOperation
from pathlib import Path

DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))
MAZE = [list(s) for s in ['S..X.', '.X...', '.X.X.', '...XE', 'XX...']]
WEIGHTED = [['S',1,1,8,1],[1,8,3,1,1],[1,8,1,8,1],[1,1,1,8,1],[8,8,1,1,'E']]
BOARD = [['A',1,1,'X',2,1,'T1'],[1,'X',6,'X',1,'X',1],[1,1,'T2',1,1,1,1],['X',1,'X',8,'X',1,'X'],[1,1,1,1,'T3',1,1],[1,'X',9,'X',1,'X',1],['T4',1,1,'X',1,1,'B']]
POINTS = {'A':(0,0),'B':(6,6),'T1':(0,6),'T2':(2,2),'T3':(4,4),'T4':(6,0)}
VALUES = {'T1':9,'T2':6,'T3':7,'T4':7}

def manhattan(a, b):
    return abs(a[0]-b[0])+abs(a[1]-b[1])

def astar(grid, start, goal, weighted=False, free_goal=False, heuristic='safe',
          *, integer_weights=False):
    """返回路径、代价、统计。入格计费；起点免收初始成本；支持零权与重新打开。"""
    if not grid or not grid[0] or any(len(row)!=len(grid[0]) for row in grid):
        raise ValueError('地图必须为非空矩形')
    rows, cols = len(grid), len(grid[0])
    def valid(p):
        return 0<=p[0]<rows and 0<=p[1]<cols and grid[p[0]][p[1]]!='X'
    if not valid(start) or not valid(goal):
        return {'path':[], 'cost':-1, 'expanded':0, 'generated':0, 'peak_open':0}
    if integer_weights:
        grid = [
            [x if x == 'X' else _integer_weight(x) for x in row]
            for row in grid
        ]
    if start == goal:
        return {'path': [start], 'cost': 0, 'expanded': 1,
                'generated': 1, 'peak_open': 1}

    def cost(p):
        if free_goal and p == goal:
            return 0
        x = grid[p[0]][p[1]]
        if integer_weights:
            return x
        try:
            c=float(x) if weighted else 1
        except (ValueError, TypeError):
            if x not in ('S','E','A','B','T1','T2','T3','T4','.'):
                raise ValueError('未知地形: '+str(x))
            c=1
        if c<0 or not math.isfinite(c): raise ValueError('仅支持有限非负代价')
        return c
    floor = min(cost((r, c)) for r in range(rows) for c in range(cols)
                if valid((r, c)) and (r, c) != goal)
    def h(p):
        d=manhattan(p,goal)
        if heuristic=='zero': return 0
        if heuristic=='raw': return d  # 仅供可采纳性诊断
        return floor*max(d-1,0)+cost(goal) if d else 0
    counter=itertools.count()
    # f 相同时按入队先后处理，固定邻居次序，保证节点数可复现。
    queue=[(h(start),next(counter),0,start)]
    best={start:0}; parent={}; expanded=0; generated=1; peak=1
    while queue:
        _,_,g,p=heapq.heappop(queue)
        if g!=best[p]: continue  # 跳过过时条目，不计为有效扩展
        expanded+=1
        if p==goal:
            path=[p]
            while p in parent: p=parent[p]; path.append(p)
            return {'path':path[::-1],'cost':g,'expanded':expanded,'generated':generated,'peak_open':peak}
        for dr,dc in DIRS:
            q=(p[0]+dr,p[1]+dc)
            if not valid(q): continue
            ng=g+cost(q)
            if ng<best.get(q,math.inf):
                best[q]=ng; parent[q]=p
                heapq.heappush(queue,(ng+h(q),next(counter),ng,q)); generated+=1
        peak=max(peak,len(queue))
    return {'path':[],'cost':-1,'expanded':expanded,'generated':generated,'peak_open':peak}

def solve_maze(start, end, grid):
    return astar(grid,start,end)['path']

def _integer_weight(value):
    if isinstance(value, str):
        if value in ('S', 'E', 'A', 'B', 'T1', 'T2', 'T3', 'T4', '.'):
            return 1
        try:
            value = Decimal(value)
        except InvalidOperation:
            raise ValueError(f'未知地形: {value}') from None
    if isinstance(value, int) and value >= 0:
        return int(value)
    if isinstance(value, float) and math.isfinite(value) and value >= 0 and value.is_integer():
        return int(value)
    if isinstance(value, Decimal) and value.is_finite() and value >= 0:
        if value == value.to_integral_value():
            return int(value)
    raise ValueError('地形权重必须为有限非负整数')

def solve_weighted(start, end, grid):
    """题二：支持有限非负整数值及整数数字字符串，进入终点免费。"""
    return astar(grid, start, end, weighted=True, free_goal=True,
                 integer_weights=True)['cost']

class TreeNode:
    def __init__(self,value=None,children=None):
        self.value=value
        self.children=children if children is not None else []

def sample_tree():
    def pair(a,b): return TreeNode(children=[TreeNode(a),TreeNode(b)])
    return TreeNode(children=[TreeNode(children=[pair(3,5),pair(6,9)]),TreeNode(children=[pair(6,9),pair(1,2)])])

def tree_from_dict(data):
    data=data.get('tree_structure',data)
    return TreeNode(data.get('value'),[tree_from_dict(x) for x in data.get('children',[])])

def tree_search(root, pruning=True, order='left'):
    stats={'nodes':0,'leaves':0,'cutoffs':0}; trace=[]
    def rec(node,maximizing,alpha,beta,path):
        stats['nodes']+=1
        if not node.children:
            if node.value is None: raise ValueError('叶子必须有 value')
            stats['leaves']+=1; trace.append({'path':path,'value':node.value})
            return node.value
        children=list(enumerate(node.children))
        if order=='reverse': children.reverse()
        best=-math.inf if maximizing else math.inf
        for i,(idx,child) in enumerate(children):
            v=rec(child,not maximizing,alpha,beta,path+[idx])
            best=max(best,v) if maximizing else min(best,v)
            if maximizing: alpha=max(alpha,best)
            else: beta=min(beta,best)
            if pruning and alpha>=beta and i<len(children)-1:
                stats['cutoffs']+=1; break
        return best
    value=rec(root,True,-math.inf,math.inf,[])
    return {'value':value,**stats,'trace':trace}

def alpha_beta(root):
    return tree_search(root)['value']

def distance_matrix():
    return {a:{b:astar(BOARD,p,q,True)['cost'] for b,q in POINTS.items()} for a,p in POINTS.items()}

@dataclass(frozen=True)
class State:
    a: str='A'
    b: str='B'
    remaining: tuple=('T1','T2','T3','T4')
    score_a: float=0
    score_b: float=0
    player: str='A'

def game_search(dist, depth, pruning=True, order='natural', values=None, initial=None):
    """深度按单次行动计。排序仅使用当前净收益，不预知完整子树值。"""
    values=VALUES if values is None else values
    stats={'nodes':0,'leaves':0,'cutoffs':0}
    def rec(s,h,alpha,beta):
        stats['nodes']+=1
        if h==0 or not s.remaining:
            stats['leaves']+=1
            return s.score_a-s.score_b,None,[]
        maximizing=s.player=='A'; pos=s.a if maximizing else s.b
        targets=[t for t in s.remaining if dist[pos][t]>=0]
        if order=='reverse': targets.reverse()
        elif order in ('gain','bad_gain'):
            targets.sort(key=lambda t:(-(values[t]-dist[pos][t]) if order=='gain' else values[t]-dist[pos][t],t))
        if not targets:
            raise ValueError('当前玩家无可达宝藏：题目未定义跳过回合规则')
        best=-math.inf if maximizing else math.inf; target=None; line=[]
        for i,t in enumerate(targets):
            gain=values[t]-dist[pos][t]
            nxt=State(t if maximizing else s.a,s.b if maximizing else t,
                      tuple(x for x in s.remaining if x!=t),s.score_a+(gain if maximizing else 0),
                      s.score_b+(0 if maximizing else gain),'B' if maximizing else 'A')
            v,_,pv=rec(nxt,h-1,alpha,beta)
            if target is None or (v>best if maximizing else v<best):
                best,target,line=v,t,[{'player':s.player,'target':t,'gain':gain}]+pv
            if maximizing: alpha=max(alpha,best)
            else: beta=min(beta,best)
            if pruning and alpha>=beta and i<len(targets)-1:
                stats['cutoffs']+=1; break
        return best,target,line
    value,target,line=rec(initial or State(),depth,-math.inf,math.inf)
    return {'best_target':target,'best_value':value,'line':line,**stats}

def root_values(dist,depth):
    result={}
    for t in VALUES:
        s=State(t,'B',tuple(x for x in VALUES if x!=t),VALUES[t]-dist['A'][t],0,'B')
        result[t]=game_search(dist,depth-1,False,initial=s)['best_value']
    return result


def timed(fn,repeats):
    # 预热后每批至少约 10 ms；取各批中位数和四分位距。
    fn(); n=1
    while True:
        begin=time.perf_counter_ns()
        for _ in range(n): fn()
        if time.perf_counter_ns()-begin>=10_000_000: break
        n*=2
    samples=[]
    for _ in range(repeats):
        begin=time.perf_counter_ns()
        for _ in range(n): fn()
        samples.append((time.perf_counter_ns()-begin)/n/1000)
    quart=statistics.quantiles(samples,n=4)
    return {'median_us':statistics.median(samples),'iqr_us':quart[2]-quart[0],'batch_size':n,'batches':repeats}

def write_csv(path, rows):
    with open(path,'w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def run(out,repeats=7):
    out.mkdir(parents=True,exist_ok=True)
    dist=distance_matrix(); records=[]
    for q,grid,start,end,weighted,free in [('Q1',MAZE,(0,0),(3,4),False,False),('Q2',WEIGHTED,(0,0),(4,4),True,True)]:
        for h in ('zero','safe'):
            fn=lambda g=grid,s=start,e=end,w=weighted,f=free,h=h: astar(g,s,e,w,f,h)
            r=fn(); records.append({'task':q,'setting':h,**{k:v for k,v in r.items() if k!='path'},**timed(fn,repeats)})
    trees=[]
    for pruning in (False,True):
        for order in ('left','reverse'):
            fn=lambda p=pruning,o=order:tree_search(sample_tree(),p,o)
            r=fn(); trees.append({'algorithm':'AB' if pruning else 'Minimax','order':order,**{k:v for k,v in r.items() if k!='trace'},**timed(fn,repeats)})
    games=[]
    for depth in (2,4):
        for pruning in (False,True):
            for order in ('natural','reverse','gain','bad_gain'):
                fn=lambda d=depth,p=pruning,o=order:game_search(dist,d,p,o)
                r=fn();games.append({'depth':depth,'algorithm':'AB' if pruning else 'Minimax','order':order,**{k:v for k,v in r.items() if k!='line'},**timed(fn,repeats)})
    # 拓展：多种障碍密度、固定种子、多地图；无解情况也计入，不筛选有利样例。
    extra=[]
    for density in (0,.15,.30):
        for seed in range(10):
            rng=random.Random(2026+seed)
            grid=[['X' if rng.random()<density else 1 for _ in range(30)] for _ in range(30)]
            grid[0][0]=grid[-1][-1]=1
            for h in ('zero','safe'):
                r=astar(grid,(0,0),(29,29),True,False,h)
                extra.append({'density':density,'seed':2026+seed,'heuristic':h,**{k:v for k,v in r.items() if k!='path'}})
    sensitivity=[]
    for val in (3,7,11,15):
        values={**VALUES,'T3':val}
        for depth in (2,4):
            r=game_search(dist,depth,True,'gain',values)
            sensitivity.append({'T3_value':val,'depth':depth,**{k:v for k,v in r.items() if k!='line'}})
    # 正确性：独立参考算法，而非只匹配给定输出。
    for i in range(0,len(extra),2):
        assert extra[i]['cost']==extra[i+1]['cost'], '拓展图最优成本不一致'
    checks=validate(dist)
    data={'environment':{'python':platform.python_version(),'platform':platform.platform(),'timer':'perf_counter_ns','repeats':repeats},
          'q1':astar(MAZE,(0,0),(3,4)),'q2':astar(WEIGHTED,(0,0),(4,4),True,True),
          'q3':tree_search(sample_tree()),'distances':dist,'astar':records,'tree':trees,'game':games,
          'lines':{str(h):game_search(dist,h,False) for h in (2,4)},'root_values':{str(h):root_values(dist,h) for h in (2,4)},'extra':extra,'sensitivity':sensitivity,'validation':checks}
    for name,rows in [('astar_metrics',records),('tree_metrics',trees),('game_metrics',games),('grid_extension',extra),('value_sensitivity',sensitivity)]: write_csv(out/(name+'.csv'),rows)
    write_csv(out/'distance_matrix.csv',[{'source':a,**row} for a,row in dist.items()])
    (out/'results.json').write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:data[k] for k in ('q1','q2','distances','lines','validation')},ensure_ascii=False,indent=2))
    return data

def reference_cost(grid,start,goal,free=False):
    """独立 Bellman-Ford 松弛，验证随机小图；不调用 A*。"""
    rows,cols=len(grid),len(grid[0]);d={start:0}
    for _ in range(rows*cols-1):
        changed=False
        for r in range(rows):
            for c in range(cols):
                p=(r,c)
                if p not in d:continue
                for dr,dc in DIRS:
                    q=(r+dr,c+dc)
                    if not(0<=q[0]<rows and 0<=q[1]<cols) or grid[q[0]][q[1]]=='X':continue
                    cost=0 if free and q==goal else grid[q[0]][q[1]]
                    v=d[p]+cost
                    if v<d.get(q,math.inf):d[q]=v;changed=True
        if not changed:break
    return d.get(goal,-1)

def validate(dist=None):
    dist=dist or distance_matrix(); checks=0
    numeric=[[1 if isinstance(x,str) and x!='X' else x for x in row] for row in BOARD]
    for a,p in POINTS.items():
        for b,q in POINTS.items():
            assert dist[a][b]==reference_cost(numeric,p,q);checks+=1
    assert solve_weighted((0,0),(0,2),[['S','3','E']])==3;checks+=1
    assert len(solve_maze((0,0),(3,4),MAZE))-1==7;checks+=1
    assert solve_weighted((0,0),(4,4),WEIGHTED)==7;checks+=1
    assert solve_maze((0,0),(0,2),[['S','X','E']])==[];checks+=1
    assert solve_weighted((0,0),(0,2),[['S','X','E']])==-1;checks+=1
    assert solve_maze((0,0),(0,0),[['S']])==[(0,0)];checks+=1
    single_cell = [['S', 'X']]
    assert solve_maze((0, 0), (0, 0), single_cell) == [(0, 0)]
    assert astar(single_cell, (0, 0), (0, 0)) == {
        'path': [(0, 0)], 'cost': 0, 'expanded': 1, 'generated': 1, 'peak_open': 1,
    }
    checks += 1
    large = 2**53 + 1
    for weight, expected in ((3, 3), (3.0, 3), ('3', 3), ('3.0', 3),
                             (large, large), (str(large), large)):
        grid = [['S', weight, 'E']]
        assert solve_weighted((0, 0), (0, 2), grid) == expected
    for weight in (1.5, '1.5'):
        try:
            solve_weighted((0, 0), (0, 2), [['S', weight, 'E']])
        except ValueError:
            pass
        else:
            raise AssertionError(f'应拒绝非整数地形权重: {weight!r}')
    checks += 1
    assert solve_weighted((0,0),(0,1),[['S','E']])==0;checks+=1
    assert alpha_beta(sample_tree())==5;checks+=1
    rng=random.Random(42)
    for _ in range(100):
        grid=[['X' if rng.random()<.2 else rng.randint(0,9) for _ in range(5)] for _ in range(5)]
        grid[0][0]=grid[4][4]=1
        for free in (False,True):
            expected=reference_cost(grid,(0,0),(4,4),free)
            r=astar(grid,(0,0),(4,4),True,free)
            assert r['cost']==expected
            if r['path']:
                assert r['path'][0]==(0,0) and r['path'][-1]==(4,4)
                assert all(manhattan(a,b)==1 for a,b in zip(r['path'],r['path'][1:]))
            checks+=1
    for _ in range(50):
        def build(depth):return TreeNode(rng.randint(-20,20)) if depth==0 else TreeNode(children=[build(depth-1) for _ in range(3)])
        root=build(4);base=tree_search(root,False)['value']
        for order in ('left','reverse'):
            assert tree_search(root,True,order)['value']==base;checks+=1
    # 独立枚举完整行动排列作为博弈参考。
    def exhaustive(prefix,h,values):
        remaining=[t for t in VALUES if t not in prefix]
        if len(prefix)==h:
            a,b='A','B';score=0
            for i,t in enumerate(prefix):
                if i%2==0:score+=values[t]-dist[a][t];a=t
                else:score-=values[t]-dist[b][t];b=t
            return score
        scores=[exhaustive(prefix+[t],h,values) for t in remaining]
        return max(scores) if len(prefix)%2==0 else min(scores)
    for _ in range(30):
        values={t:rng.randint(0,20) for t in VALUES}
        for h in (2,4):
            ref=exhaustive([],h,values)
            for order in ('natural','reverse','gain','bad_gain'):
                assert game_search(dist,h,True,order,values)['best_value']==ref;checks+=1
    for h,t,v in [(2,'T2',-1),(4,'T3',3)]:
        r=game_search(dist,h)
        assert (r['best_target'],r['best_value'])==(t,v);checks+=1
    return {'assertion_groups_passed':checks,'random_grids':100,'random_trees':50,'random_value_sets':30}

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=Path(__file__).parent/'results')
    parser.add_argument('--test',action='store_true')
    parser.add_argument('--task',choices=['1','2','3','4'],help='单题 JSON 输入输出')
    parser.add_argument('--input',type=Path,help='JSON 文件，见 README')
    parser.add_argument('--repeats',type=int,default=7)
    args=parser.parse_args()
    if args.repeats<4:parser.error('--repeats 至少为4')
    if args.task:
        if not args.input:parser.error('--task 需要 --input')
        parse_float = Decimal if args.task == '2' else float
        data = json.loads(args.input.read_text(encoding='utf-8'), parse_float=parse_float)
        if args.task=='1':result=solve_maze(tuple(data['start']),tuple(data['end']),data['grid'])
        elif args.task == '2':
            result = solve_weighted(tuple(data['start']), tuple(data['end']), data['grid'])
        elif args.task=='3':result=alpha_beta(tree_from_dict(data))
        else:
            r=game_search(distance_matrix(),data.get('depth',4))
            result={'best_target':r['best_target'],'best_value':int(r['best_value'])}
        print(json.dumps(result,ensure_ascii=False))
    elif args.test:print(validate())
    else:run(args.out,args.repeats)
