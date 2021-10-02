# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 16:28:06 2020

@author: hulinyuan
"""
import random
import numpy as np
import time
from docplex.mp.model import Model

def skuall(skutype): 
    skulist= dict()
    skuindex = np.random.exponential(0.5,skutype)
    for i in range(1,skutype+1):
        skulist.update({f'{i}':skuindex[i-1]})
    return skulist

def gensku():
    index = np.random.exponential(0.5)
    sim = 10
    for i in skulist:
        sim1 = abs(index-skulist[i])
        if sim>sim1:
            recordsku=i
            sim  = sim1   
    return recordsku

def trigger_order(ordernum, skuperorder):
    Order = dict()
    for i in range(0, ordernum):
        perorder=[]
        skupo = random.randint(1,skuperorder+1)
        for ii in range(0,skupo):
            perorder.append(gensku())
        Order.update({f'{i+1}':set(perorder)})
    
    order_left=['1']
    for i in range(2,ordernum+1):
        order_left.append(f'{i}')
    return Order, order_left

def trigger_rack(Order, perrack):
    Rack = dict()
    orderpool = set()
    for i in Order:
        orderpool = orderpool | Order[i]
    rackpool = set()   
    j=0
    while (orderpool-(rackpool & orderpool)) != set():
        j = j+1
        per=[]
        while len(set(per)) < perrack:
            tr=[]
            tr.append(gensku())
            a = orderpool & set(tr)
            
            if len(a)!=0:
                per.append(tr[0])
        
        Rack.update({f'{j}':set(per)})
        
        for i in Rack:
            rackpool = rackpool|Rack[i]
      
    return Rack 

def showorderrack(Order,Rack):
    for i in Order:
        print(f'{i}:{Order[i]}')
    for i in Rack:
        print(f'{i}:{Rack[i]}')
        
def chooseorder(basketnow, order, capacitynow):
    kk = Orderleft[0]
    choseorder = [0,]
    while capacitynow > 0:
        sim_front = 0
        for k in Orderleft:
            result = basketnow & order[f'{k}']
            if len(basketnow) > 0 and len(order[f'{k}']) > 0:
                sim_now = (2 * len(result)) / (len(basketnow) + len(order[f'{k}']))
                if sim_front <= sim_now:
                    sim_front = sim_now
                    kk = k
        basketnow = basketnow | order[f'{kk}']
        choseorder.append(f'{kk}')
        Orderleft.remove(f'{kk}')
        capacitynow = capacitynow - 1
    return basketnow, choseorder, capacitynow

def chooserack(order_wait_sequence, rack):
    kk = 0
    sim_front = 0
    for k in rack:
        lengthr = 0
        lengtho = 0  
        for o in order_wait_sequence:
            result = order_wait_sequence[o] & rack[f'{k}']
            lengthr = lengthr + len(result)
            lengtho = lengtho +len(order_wait_sequence)
        sim_now = lengthr / lengtho
        if sim_front <= sim_now:
            sim_front = sim_now
            kk = k
    return kk

def starwork(O,R,Cap,Orderleft):
    # 开始排单
    order_choose_sequence = [0,]
    rack_sequence = [0,]
    order_wait_sequence = dict()
    order_finish_sequence = [0,]
    Capacity = Cap
    # 随机选择初始订单
    i = random.randint(0, len(O) - 1)
    Orderleft.remove(f'{i}')
    basket_now = O[f'{i}']
    order_choose_sequence.append(f'{i}')
    #print(f'选择的订单{i}')
    order_wait_sequence.update({f'{i}':O[f'{i}']})               
    capacity_now = Capacity-1
    basket_now, chose_order, capacity_now = chooseorder(basket_now, O, capacity_now)  # 计算相似性，选取相似的订单直到工作台满，返回最大的订单
    #print(f'当前选择的订单{chose_order}') 
    # print(f'----------{capacity_now}---------------')
    #print(f'排货前工作台内SUK：{basket_now}')
    for j in chose_order:
        if j != 0:
            #print(f'当前选择的订单{chose_order}') 
            order_choose_sequence.append(j)
            order_wait_sequence.update({f'{j}':O[f'{j}']})            
    chose_rack = chooserack(order_wait_sequence, R)  # 计算货架与工作台相似性，选取一个货架
    #b = R[f'{chose_rack}']
    # print(f'当前选择的货架{chose_rack}号{b}')                           
    rack_sequence.append(chose_rack)  # 记录货架的顺序 
    while len(basket_now) > 0:
        for ii in order_wait_sequence:
            if order_wait_sequence[ii] != set():
                #print(f'--------开始{ii}订单排货过程-----------')
                a = order_wait_sequence[ii] & R[f'{chose_rack}']
                # print(f'排货前工作台内SUK：{basket_now}')
                # print(f'排货前订单内SUK：{order_wait_sequence[ii]}') 
                #print(f'订单被完成的SUK：{a}')
                basket_now = basket_now - a  # 更新工作台上的品种
                # print(f'出货后工作台内SUK：{basket_now}')
                order_status = order_wait_sequence[ii] - a # 具体订单被完成的情况更新
                #print(f'订单处理后剩余SUK：{order_status}')
                if order_status == set():  # 查看具体是否有订单被完成
                    #print(f'-------{ii}订单完成------')
                    order_wait_sequence[ii] = order_status                
                    capacity_now = capacity_now + 1  # 订单被完成，工作台能力释放
                    order_finish_sequence.append(ii)
                else:
                    #print(f'------{ii}订单未完成------')
                    order_wait_sequence[ii] = order_status
                    #print(f'订单未完成SUK：{ii};{order_wait_sequence[ii]}')               
        if len( Orderleft) > 0:
            basket_now, chose_order, capacity_now = chooseorder(basket_now, O, capacity_now)  # 选择一个新的订单
            for j in chose_order:
                if j != 0:
                    # print(f'当前选择的订单{chose_order}') 
                    order_choose_sequence.append(j)
                    order_wait_sequence.update({f'{j}':O[f'{j}']})
                    #print(f'当前剩余未上工作台订单{Order_left}')
                    
        if basket_now == set():
            break
        else:
            chose_rack = chooserack(order_wait_sequence, R)
            #b = R[f'{chose_rack}']
            #print(f'当前选择的货架{chose_rack}号{b}') 
            rack_sequence.append(chose_rack)  # 记录货架的顺序  
            #print(f'当前剩余未上工作台订单{Order_left}')
    return order_choose_sequence,order_finish_sequence,rack_sequence

def optimization(O, rack):
    L = dict()
    for i in O:
        L.update({f'{i}':list()})
    
        for j in rack:
            l = list()
            order = O[f'{i}']
            r = rack[j]
            result = order & R[f'{r}']
            if result != set():
                l.append(j)
                order = order - result

                if order == set():
                    L[i].append(l)
                    continue
                else:                    
                    for jj in rack:
                        if int(jj)>int(j):
                            r = rack[jj]
                            result = order & R[f'{r}']
                            order = order - result
                            l.append(jj)
                            if order == set():
                                L[i].append(l)
                                break
    return L, task

def rack_cand(Order, Rack):
    orderpool = set()
    for i in Order:
        orderpool = orderpool | Order[f'{i}']
        
    rackcand = dict()   
    for j in orderpool:
        cand= []
        for k in Rack:
            if f'{j}' in Rack[f'{k}']:
                cand.append(f'{k}')
        rackcand.update({f'{j}':cand})
    return rackcand


def norm_rackchoose(sku):# 选择货架
    rack = random.choice(rackcand[sku])
    return rack

def norm_orderchoose(basketnow, Orderleft, capacitynow):
    choseorder = [0,]
    while capacitynow > 0:
        if len(Orderleft)>0:
            order = Orderleft[0]
            basketnow = basketnow | O[order]
            choseorder.append(order)
            order_choose_sequence.append(order)
            order_wait_sequence.update({order:O[order]})
            capacitynow = capacitynow-1
            Orderleft.pop(0)
        else:
            break
    return basketnow, choseorder, capacitynow

def norm_findsku(basketnow, order_wait_sequence):  
    maxactive = 0
    activesku=set()
    for i in basketnow:
        active = 0
        for j in order_wait_sequence:
            if i in order_wait_sequence[j]:
                active = active + 1
        if active > maxactive:
            maxactive  = active
            activesku = i          
    return activesku

# 算例生成
skulist = skuall(20)
ordersku = 5
C = 5
racksku = 4
O, Order_left = trigger_order(10, ordersku)  # 生成订单，（订单数，每个订单最大品种数）
orderset = O
R = trigger_rack(O, racksku)  # 生成货架，（订单集合，货架内品种数）
Orderleft = list(Order_left)

# benchmark 部分
skuPool = set()
for i in O:
     for j in O[i]:
        ll = []
        ll.append(j)
        skuPool = skuPool | set(ll)
    
skuPool = list(skuPool)
rc = dict()
for s in skuPool:
    r1 = []
    for r2 in R:
        if s in R[r2]:
            r1.append(int(r2))
    rc.update({s: r1})
            
lent = 0
for i in O:  # 计算总作业量
    lent = lent + len(O[i])
    
T = range(1, lent+1)
order = range(1, len(O)+1)   # 订单数
rack = range(1, len(R)+1)    
print(f'订单数：{len(O)}; 订单内sku数：{ordersku}；货架数：{len(R)};货架内sku数：{racksku}；工作台处理能力：{C}')
model = Model()  # 创建模型
model.parameters.timelimit(300)
X = model.binary_var_dict([(j, t) for j in rack for t in T], lb=0, name='Xrt')  # 决策变量列表
Y = model.binary_var_dict([(i, t) for i in order for t in T], lb=0, name='Yot')
Z = model.binary_var_dict([(s, i, t) for i in order for t in T for s in O[f'{i}']], lb=0, name='Zsot')
# 设定目标函
min = model.minimize(model.sum(model.sum(X[j, t]*t for j in rack for t in T) for t in T))
# 添加约束条件
model.add_constraints(model.sum(X[j, t] for j in rack) <= 1 for t in T)
model.add_constraints(model.sum(Y[i, t] for i in order) <= C for t in T)

for t in range(1, len(T)-1):
    for tt in range(t+1, len(T)):
        for ttt in range(tt+1, len(T)+1):
            model.add_constraints(Y[i, t] + Y[i, ttt] <= Y[i, tt] + 1 for i in order)
    
model.add_constraints(1 <= model.sum(Z[s, i, t] for t in T) for i in order for s in O[f'{i}'])

for i in order:
    for s in O[f'{i}']:
        model.add_constraints(2*Z[s, i, t] <= Y[i, t]  + model.sum(X[j, t] for j in rc[s])for t in T)
    
model.print_information()
solution = model.solve(agent='local',log_output=False)  # 求解模型  
XRT = {}
for t in T:
    for j in rack:
        name = solution.get_value(f'Xrt_({j}, {t})')
        if name > 0:
            XRT[j, t] = name
print(f'Cplextasks所需操作数:{len(XRT)}')
solutime = solution.solve_details.time
print(f'第一阶段求解时间{solutime}')

# 两阶段算法部分
Orderleft = Order_left[:]
start1 = time.time()
order_wait_sequence, order_finish_sequence, rack_sequence = starwork(O, R, C,Orderleft) 
#开始派单（订单，货架，工作台能力）

task = dict()
for i in range(1,len(rack_sequence)):
    task.update({f'{i}':rack_sequence[i]})
    
L, task = optimization(O, task)
interval = 0
for i in L:
    b = L[i]
    interval = max(len(b),interval)
arfa = [[[0]*(len(task)+1) for _ in range(interval+1)] for _ in range(len(O)+1)]
itvl = [0 for _ in range(len(O)+1)]
for o in O:
    a = int(o)
    b = L[o]
    for j in b:
        itvl[a] = len(b)+1
        v = b.index(j)+1
        for i in j:
            t = int(i)
            arfa[a][v][t] = 1
                
#print(f'第一阶段{len(task)}') 

#print(f'第一阶段求解时间{end1-start1}')
modeltwo = Model()  # 创建模型
modeltwo.parameters.preprocessing.presolve(1)
modeltwo.parameters.timelimit(600)
# modeltwo.parameters.mip.limits.repairtries(60)
X = modeltwo.binary_var_dict([(i,v) for i in range(1,len(O) + 1) for v in range(1,itvl[i])], lb=0, name='Xiv')  # 决策变量列表
G = modeltwo.binary_var_dict([(t) for t in range(1,len(task) + 1)], lb=0, ub=1, name='Gt')
# 设定目标函
min = modeltwo.minimize(modeltwo.sum(G[t] for t in range(1,len(task)+1)))
# 添加约束条件(15)
dd = modeltwo.add_constraints(modeltwo.sum(modeltwo.sum(arfa[i][v][t] * X[i, v] for v in range(1, itvl[i])) for i in range(1, len(O) + 1)) <= C * G[t] for t in range(1, len(task) + 1))
cc = modeltwo.add_constraints(modeltwo.sum(X[i,v] for v in range(1,itvl[i])) == 1 for i in range(1,len(O)+1))
#modeltwo.print_information()
# 添加约束条件(16)
solutiontwo = modeltwo.solve(agent='local',log_output=False,clean_before_solve=True)
#print(f'订单数：{len(O)}; 订单内sku数：{ordersku}；货架数：{len(R)};货架内sku数：{racksku}；工作台处理能力：{C}')
#print(f'第一阶段: {len(task)}') 
obj = solutiontwo.get_objective_value()
#print(f'第二阶段: {obj}')
soltime = solutiontwo.solve_details.time
#print(f'第二阶段求解时间{soltime}')
status = solutiontwo.solve_details.status
#print(f'第二阶段解状态{status}')
end1 = time.time()
# RWP
start = time.time()
basket_now=set()
rackcand = rack_cand(O, R)
order_choose_sequence = [0,]
rack_sequence = [0,]
order_wait_sequence = dict()
order_finish_sequence = [0,]
order_last_record =dict()
Capacity = C
capacity_now = Capacity
basket_now, choseorder, capacity_now = norm_orderchoose(basket_now, Order_left, capacity_now)
sku = norm_findsku(basket_now, order_wait_sequence)
chose_rack = norm_rackchoose(sku)
chose_rack
b = R[f'{chose_rack}']
#print(f'当前选择的货架{chose_rack}号{b}')                           
rack_sequence.append(chose_rack)  # 记录货架的顺序 
while len(basket_now) > 0:
    for ii in order_wait_sequence:
            if order_wait_sequence[ii] != set():
                #print(f'--------开始新的订单排货过程-----------')
                a = order_wait_sequence[ii] & R[f'{chose_rack}']
                #print(f'排货前工作台内SUK：{basket_now}')
                #print(f'排货前订单内SUK：{order_wait_sequence[ii]}') 
                #print(f'订单被完成的SUK：{a}')
                basket_now = basket_now - a  # 更新工作台上的品种
                #print(f'出货后工作台内SUK：{basket_now}')
                order_status = order_wait_sequence[ii] - a # 具体订单被完成的情况更新
                #print(f'订单处理后剩余SUK：{order_status}')
                if order_status == set():  # 查看具体是否有订单被完成
                    #print("-------订单完成------")
                    order_wait_sequence[ii] = order_status                
                    capacity_now = capacity_now + 1  # 订单被完成，工作台能力释放
                    order_finish_sequence.append(ii)
                else:
                    #print("------订单未完成------")
                    order_wait_sequence[ii] = order_status
                    order_last_record.update
                    #print(f'订单未完成SUK：{order_wait_sequence[ii]}')
    if len( Order_left) > 0:
        basket_now, choseorder, capacity_now = norm_orderchoose(basket_now,Order_left,capacity_now)  # 选择一个新的订单

    if  basket_now ==set():
        break
    else:
        sku = norm_findsku(basket_now, order_wait_sequence)
        chose_rack = norm_rackchoose(sku)
        b = R[f'{chose_rack}']
        #print(f'当前选择的货架{chose_rack}号{b}') 
        rack_sequence.append(chose_rack)  # 记录货架的顺序 
end = time.time()
        #print(f'当前剩余未上工作台订单{Order_left}')  
#print(f'订单数：{len(O)};订单内sku数：{ordersku}；货架数：{len(R)};货架内sku数：{racksku}；工作台处理能力：{C}')
#print (f'常规操作数 :{len(rack_sequence)}')
print('\n')
# 以下代码是为了显示结果
print('结果汇总----------------------------')
print(f'订单数：{len(O)}; 订单内sku数：{ordersku}；货架数：{len(R)};货架内sku数：{racksku}；工作台处理能力：{C}')
print(f'Cplextasks所需操作数:{len(XRT)}')
solutime = solution.solve_details.time
print(f'Cplex求解时间：{solutime}')
statu = solution.solve_details.status
print(f'Cplex解状态：{statu}')
gap1 = solution.solve_details.mip_relative_gap
print(f'Cplex解GAP：{gap1}')
print(f'第一阶段操作数: {len(task)}') 
print(f'两阶段求解时间：{end1-start1}')
print(f'第一阶段解GAP：{(len(task)-obj)/obj}')
obj = solutiontwo.get_objective_value()
print(f'第二阶段操作数: {obj}')
soltime = solutiontwo.solve_details.time
print(f'第二阶段求解时间：{soltime}')
status = solutiontwo.solve_details.status
print(f'第二阶段解状态：{status}')
gapp = solutiontwo.solve_details.mip_relative_gap
print(f'第二阶段GAP：{gapp}')
print (f'常规操作数 :{len(rack_sequence)}')
print(f'常规求解时间：{end-start}')
print('\n')