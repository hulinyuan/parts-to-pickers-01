# -*- coding: utf-8 -*-
"""
Created on Wed May 19 16:51:20 2021

@author: 胡琳苑
"""

import sys
import math
import random
import gurobipy as gb
from gurobipy import GRB
from gurobipy import *

def printVarnum():
    print(f'alpha_pt的个数：{len(alpha_pt)}个')
    print(f'x_po的个数：{len(x_po)}个')
    print(f'k_pot的个数：{len(k_pot)}个')
    print(f'y_pr的个数：{len(y_pr)}个')
    print(f'l_prt个数：{len(l_prt)}个')
    print(f'pi_poit的个数：{len(pi_poit)}个')

def printsolution():
    vname = [alpha_pt,x_po,k_pot,y_pr,l_prt,pi_poit]
    alpha_pt_dict = Model.getAttr('x',alpha_pt)
    x_po_dict  = Model.getAttr('x',x_po)
    k_pot_dict = Model.getAttr('x',k_pot)
    y_pr_dict  = Model.getAttr('x',y_pr)
    l_prt_dict = Model.getAttr('x',l_prt)
    pi_dict    = Model.getAttr('x',pi_poit)
    Model_Objective = 0
    for p in range(m):
        for t in range(T):
            if alpha_pt_dict[p,t]!=0:
                Model_Objective = Model_Objective + 1
    print(f'Model_Objective:{Model_Objective+m}')

    # 按工作台来输出，工作台m号 时间 t 用货架 r 处理订单 n 的第s个sku
    for p in range(m):
        print(f'Workshop：{p+1}:')
        for t in range(T):
            for r in range(R):
                if (y_pr == 0) & (l_prt_dict[p,r,t]!=0):
                    print('error')
                if (y_pr != 0) & (l_prt_dict[p,r,t]!=0):
                    print(f' t{t+1} 使用 Rank({r+1}):{Rank[r]}',end='')
                    for o in range(n):
                        if (x_po ==0) & (k_pot_dict[p,o,t]!=0):
                            print('error')
                        if (x_po!=0) & (k_pot_dict[p,o,t]!=0):
                            print(f'处理 Order({o+1}):',end='')
                            for s in Order[o]:
                                  if pi_dict[p,o,s,t]!=0:
                                        Order_dict[o].append(s)
                            print(f'已处理:{[SKU[ss] for ss in Order_dict[o]]}; ',end='')
            print()
            
            
'''       
# 算例生成
def instanceGEN(w,s,R,r,O,o):
    
    w:the number of worktables
    s:the number of total SKU
    R:the number of ranks
    r:the number of skus in one rank,please enter a list,r >=  SKUs in ORDER/R
    first element is lb and the second element is ub,and [lb,ub]
    O:the number of orders
    o:the number of skus in one Order,same as 'r'
    
    P   = range(w)
    SKU = range(s)
    used_sku =[]
    Order= dict()
    Order_dict = dict()
    for i in range(O):
            J = random.randint(o[0],o[1])
            order_list =[]
            while len(order_list)<J:
                rsku = random.choice(SKU)
                used_sku.append(rsku)
                order_list.append(rsku)
                order_list =list(set(order_list))
            Order.update({i:order_list})
            Order_dict.update({i:[]})
            
    avail_sku = list(set(used_sku)) 
    
    # 从订单中使用的sku生成货架
    Rank =dict()
    ri = 0
    
    for i in range(R):
        J = random.randint(r[0],r[1])
        rank_list =[]
      
        while len(rank_list)<J:
            if ri < len(avail_sku):
                rank_list.append(avail_sku[ri])
                ri = ri + 1
            rank_list.append(random.choice(SKU))
            rank_list =list(set(rank_list))
            
        Rank.update({i:rank_list})
        
    m = len(P)# 工作台数量
    n = len(Order) # 待处理订单数量
    R = len(Rank) # 货架的数量
    S = len(SKU) # SKU的数量
    Ri= dict()
    for s in range(S):
        ranksetfors = []
        for r in range(R):
            if SKU[s] in Rank[r]:
                ranksetfors.append(r)           
        Ri.update({s: set(ranksetfors)})
        
    return P,SKU,Order,Order_dict,Rank,Ri
'''

def printVarnum():
    print(f'alpha_pt的个数：{len(alpha_pt)}个')
    print(f'x_po的个数：{len(x_po)}个')
    print(f'k_pot的个数：{len(k_pot)}个')
    print(f'y_pr的个数：{len(y_pr)}个')
    print(f'l_prt个数：{len(l_prt)}个')
    print(f'pi_poit的个数：{len(pi_poit)}个')
    
def printsolution():
    vname = [alpha_pt,x_po,k_pot,y_pr,l_prt,pi_poit]
    alpha_pt_dict = Model.getAttr('x',alpha_pt)
    x_po_dict  = Model.getAttr('x',x_po)
    k_pot_dict = Model.getAttr('x',k_pot)
    y_pr_dict  = Model.getAttr('x',y_pr)
    l_prt_dict = Model.getAttr('x',l_prt)
    pi_dict    = Model.getAttr('x',pi_poit)
    Model_Objective = 0
    for p in range(m):
        for t in range(T):
            if alpha_pt_dict[p,t]!=0:
                Model_Objective = Model_Objective + 1
    print(f'Model_Objective:{Model_Objective+m}')

    # 按工作台来输出，工作台m号 时间 t 用货架 r 处理订单 n 的第s个sku
    for p in range(m):
        print(f'Workshop：{p+1}:')
        for t in range(T):
            for r in range(R):
                if (y_pr == 0) & (l_prt_dict[p,r,t]!=0):
                    print('error')
                if (y_pr != 0) & (l_prt_dict[p,r,t]!=0):
                    print(f' t{t+1} 使用 Rank({r+1}):{Rank[r]}',end='')
                    for o in range(n):
                        if (x_po ==0) & (k_pot_dict[p,o,t]!=0):
                            print('error')
                        if (x_po!=0) & (k_pot_dict[p,o,t]!=0):
                            print(f'处理 Order({o+1}):',end='')
                            for s in Order[o]:
                                  if pi_dict[p,o,s,t]!=0:
                                        Order_dict[o].append(s)
                            print(f'已处理:{[SKU[ss] for ss in Order_dict[o]]}; ',end='')
            print()
            
            
 
# 算例生成
def instanceGEN(w,s,R,r,O,o):
    
    P = [i*1 for i in range(w)]
    SKU = range(s)
    used_sku =[]
    Order= dict()
    Order_dict = dict()
    for i in range(O):
            J = random.randint(o[0],o[1])
            order_list =[]
            while len(order_list)<J:
                rsku = random.choice(SKU)
                used_sku.append(rsku)
                order_list.append(int(rsku))
                order_list =list(set(order_list))
            Order.update({i:order_list})
            Order_dict.update({i:[]})
            
    avail_sku = list(set(used_sku))   
    # 从订单中使用的sku生成货架
    Rank =dict()
    ri = 0
    for i in range(R):
        J = random.randint(r[0],r[1])
        rank_list =[]
      
        while len(rank_list)<J:
            if ri < len(avail_sku):
                rank_list.append(avail_sku[ri])
                ri = ri + 1
            rank_list.append(random.choice(SKU))
            rank_list =list(set(rank_list))
        Rank.update({i:rank_list})
    m = len(P)# 工作台数量
    n = len(Order) # 待处理订单数量
    R = len(Rank) # 货架的数量
    S = len(SKU) # SKU的数量
    Ri= dict()
    for s in range(S):
        ranksetfors = []
        for r in range(R):
            if SKU[s] in Rank[r]:
                ranksetfors.append(r)           
        Ri.update({s: set(ranksetfors)})
        
    return P,SKU,Order,Order_dict,Rank,Ri


if __name__ == "__main__":
    # 集合生成 w,s,R,r,O,o
    P,SKU,Order,Order_dict,Rank,Ri = instanceGEN(w=5,s=100,R=10,r=[3,5],O=25,o=[1,3]) 
    
    print('算例生成完成')
    # 下标
    m = len(P)     # 工作台数量
    n = len(Order) # 待处理订单数量
    R = len(Rank)  # 货架的数量
    S = len(SKU)   # SKU的数量
    C = 5          # 工作台容量
    T = 10         # 处理期间
    Q = []         # 订单分配
    
    for p in range(m):
        trans_p = p+1
        if (trans_p>=1) & (trans_p<=n%m):
                Q.append(math.ceil(n/m))
        else:
            Q.append(math.floor(n/m))

    # 创建模型
    print('开始构建模型')
    Model = gb.Model('ORSJ')
    #Model.setParam('TimeLimit', 1*60)
    #odel.setParam('OutputFlag', 0)
    # alpha_t^p 当工作台p在t时更换货架
    alpha_pt = Model.addVars(m,T,lb = 0, ub = 1,vtype = GRB.CONTINUOUS,name='alpha_pt')
    # x_o^p  工作台p处理订单o
    x_po = Model.addVars(m,n,vtype=GRB.BINARY,name='x_po')
    #k_pot 工作台p在t时处理订单o
    k_pot = Model.addVars(m,n,T,lb = 0, ub = 1, vtype=GRB.CONTINUOUS,name='k_pot')
    # y_pr 工作台p使用货架r
    y_pr = Model.addVars(m,R,vtype=GRB.BINARY,name='y_pr')
    # l_prt 工作台p在t时使用货架r
    l_prt = Model.addVars(m,R,T,vtype=GRB.BINARY,name='l_prt')
    # pi_poit 工作台p在t时处理订单o的SKUi
    pi_poit =Model.addVars(m,n,S,T,vtype=GRB.BINARY,name='pi_poit')

    k_coffi =dict()
    for k in k_pot:
        k_coffi.update({k: k[2]*0.0001})

    # The objective is to minimize the number of ranks 
    objective1 = Model.setObjective(gb.quicksum(alpha_pt[p,t]for p in range(m) for t in range(T)) 
                                    + k_pot.prod(k_coffi,'*','*','*')                               
                                    ,GRB.MINIMIZE)

    # 保证工作台p的订单分配数量均衡
    C2 = Model.addConstrs((x_po.sum(p,'*') == Q[p] for p in range(m)), name='C2')

    # 保证一个订单o有且只有一个工作台p处理
    C3 = Model.addConstrs((x_po.sum('*',o) <= 1 for o in range(n)),name='C3')

    # 保证一个订单o在工作台p的t时刻内处理
    C4 = Model.addConstrs((k_pot.sum(p,o,'*')<= T * x_po[p,o]
                          for p in range(m) for o in range(n)),name='C4')


    C5 = Model.addConstrs((k_pot.sum(p,o,'*')>= x_po[p,o]
                                 for p in range(m) for o in range(n)),name='C5')

    # 保证每时每个工作台p在时刻t处理所有订单总数在容量内
    C6 = Model.addConstrs((k_pot.sum(p,'*',t) <= C 
                          for p in range(m) for t in range(T)),name='C6')

    # 保证每时每个工作台p在时间t最多有一个货架
    C7 = Model.addConstrs((l_prt.sum(p,'*',t) <= 1
                          for p in range(m) for t in range(T)),name='C7')

    #保证每个工作台p的货架r在处理时间内被用到
    C8 = Model.addConstrs((l_prt.sum(p,r,'*')<= T*y_pr[p,r] 
                          for p in range(m) for r in range(R)),name='C8')

    # 保证所有时间段中有使用货架r的工作台是属于工作台p的
    C9 = Model.addConstrs((l_prt.sum(p,r,'*') >= y_pr[p,r]
                          for p in range(m) for r in range(R)),name='C9')

    # 对于工作台p至少有一个时间段t处理某一订单o的一个SKU s
    vname = 0
    for p in range(m):
        for o in range(n):
            for i in Order[o]:
                        vname= vname + 1
                        C10 = Model.addConstr((pi_poit.sum(p,o,i,'*') >= x_po[p,o]),name=f'C10_{vname}') # 1

    # 对于工作台p在t时段时使用的货架r应该是从包含订单o中的i的货架集中得到的    
    vname = 0
    for p in range(m):
        for o in range(n):
            for i in Order[o]:
                for t in range(T):
                    vname= vname + 1
                    C11 = Model.addConstr( 
                            (gb.quicksum(l_prt[(p,r,t)] for r in Ri[i]) 
                            + k_pot[(p,o,t)] >= 2*pi_poit[(p,o,i,t)]),name=f'C11_{vname}')

    vname = 0                 
    for p in range(m):
        for t in range(T-2):
            for thh in range(t+1,T-1):
                for th in range(thh+1,T):
                    for o in range(n):
                        vname= vname +1
                        C12 = Model.addConstr( 
                            (k_pot[(p,o,t)] + k_pot[(p,o,th)] <= k_pot[(p,o,thh)] + 1),name=f'C12_{vname}')


    C13 = Model.addConstrs((l_prt[(p,r,t)] - l_prt[(p,r,t-1)]<= alpha_pt[(p,t)] 
                for p in range(m) for r in range(R) for t in range(1,T)),name=f'C13')

    print('约束输入完成')
    Model.optimize()
    print('-----------')
    print('Model Input')
    print(f'Workshop Set：{P}\n\nSKU Set:{SKU}\n\nRank Set:{Rank}\n')
    op = 'Order Set:\n'
    for i in Order:
        op = op + f'{i}:{Order[i]}  '
        if (i+1)%4 == 0:
            op = op+'\n'
    print(op)
    printVarnum()
    printsolution()
    opd ='Order_finish\n'
# 订单完成情况
    for i in Order_dict:
        if set(Order_dict[i])==set(Order[i]):
            opd = opd + f'{i}:已完成  '
        else:
            opd = opd + f'{i}:未完成  '

        if (i+1)%4 == 0:
            opd = opd+'\n'
    print(opd)
    # 模型错误检查
    if Model.status == GRB.Status.INFEASIBLE:
        print('Optimization was stopped with status %d' % Model.status)
        # do IIS, find infeasible constraints
        Model.computeIIS()
        for c in Model.getConstrs():
            if c.IISConstr:
                print('%s' % c.constrName)