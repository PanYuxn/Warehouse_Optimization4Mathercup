
import pandas as pd
from coptpy import *

def get_data(path):
    df = pd.read_excel(path)
    T = 15
    TL = 3
    C_sh = [0.2, 0.2, 0.2, 0.2]  # 4个不同种类的持有成本比率
    C_sp = [0.8, 0.8, 0.8, 0.8]  # 4个不同种类的缺货成本比率
    center_wh_capacity = 20000 * 0.8
    district_wh_capacity = 12000 * 0.8
    k_price = list(df['price'])
    id_ls = ["seller_no", "product_no", "warehouse_no"]
    D_kt_col = [i for i in range(1, 19)]
    ss_k_col = ["Mean_for_ss"]
    D_kt = df.loc[:, D_kt_col].values
    ss_k = df.loc[:, ss_k_col].values
    category = list(df["Category_for_Price"].values)
    wh = list(df["warehouse_no"].values)
    return df, T, TL, C_sh, C_sp, k_price, D_kt, ss_k, category, wh, center_wh_capacity, district_wh_capacity


def get_category(original_list):
    # 初始化四个种类的位置列表
    category1_indices = []
    category2_indices = []
    category3_indices = []
    category4_indices = []

    # 遍历原始列表并记录每个种类的索引
    for index, item in enumerate(original_list):
        if item == 'low':
            category1_indices.append(index)
        elif item == 'medium':
            category2_indices.append(index)
        elif item == 'high':
            category3_indices.append(index)
        elif item == 'veryhigh':
            category4_indices.append(index)

    return category1_indices, category2_indices, category3_indices, category4_indices


def get_warehouse(my_list):
    indices = {}
    # 遍历列表，记录每个元素的索引
    for index, element in enumerate(my_list):
        if element not in indices:
            indices[element] = []
        indices[element].append(index + 1)
    return indices


path = 'src/Inputdata.xlsx'
df_data, Time, TL, C_sh, C_sp, k_price, D_kt, ss_k, category, wh, center_wh_capacity, district_wh_capacity = get_data(
    path)
low, medium, high, veryhigh = get_category(category)
wh_idx = get_warehouse(wh)
# K = [i for i in range(1, len(k_price) + 1)]
K = [i for i in range(1, 7)]
S = [1, 2, 3, 4]
T = [i for i in range(1, 16)]
big_M = 10000
env = Envr()
model = env.createModel("MIP")
# 决策变量
IB = model.addVars(K, T, lb=0, ub=COPT.INFINITY, vtype=COPT.INTEGER, nameprefix="IB")
IE = model.addVars(K, T, lb=0, ub=COPT.INFINITY, vtype=COPT.INTEGER, nameprefix="IE")
x = model.addVars(K, T, lb=0, ub=COPT.INFINITY, vtype=COPT.INTEGER, nameprefix="x")
v_s = model.addVars(K, lb=0, ub=COPT.INFINITY, vtype=COPT.INTEGER, nameprefix="v_s")
v_S = model.addVars(K, lb=0, ub=COPT.INFINITY, vtype=COPT.INTEGER, nameprefix='v_S')
O = model.addVars(K, T, lb=0, ub=COPT.INFINITY, vtype=COPT.INTEGER, nameprefix='O')
# additional varbial
buhuo = model.addVars(K, T, vtype=COPT.BINARY, nameprefix='buhuo')
quehuo = model.addVars(K, T, vtype=COPT.BINARY, nameprefix='quehuo')
kucun = model.addVars(K, T, vtype=COPT.BINARY, nameprefix='kucun')
# 目标函数
expr = LinExpr()
for t in T:
    for k in K:
        if k - 1 in low:
            expr.addTerm(IE[k, t], k_price[k - 1] * C_sh[0])  # 分组A持有成本
            expr.addTerm(O[k, t], k_price[k - 1] * C_sp[0])  # 分组A缺货成本
        if k - 1 in medium:
            expr.addTerm(IE[k, t], k_price[k - 1] * C_sh[1])  # 分组B持有成本
            expr.addTerm(O[k, t], k_price[k - 1] * C_sp[1])  # 分组B缺货成本
        if k - 1 in high:
            expr.addTerm(IE[k, t], k_price[k - 1] * C_sh[2])  # 分组C持有成本
            expr.addTerm(O[k, t], k_price[k - 1] * C_sp[2])  # 分组C缺货成本
        if k - 1 in veryhigh:
            expr.addTerm(IE[k, t], k_price[k - 1] * C_sh[3])  # 分组D持有成本
            expr.addTerm(O[k, t], k_price[k - 1] * C_sp[3])  # 分组D缺货成本
        if sum(D_kt[k - 1]) != 0:
            expr.addTerm(IB[k, 1], (7.5 / sum(D_kt[k - 1])))  # 库存周转天数
            expr.addTerm(IE[k, 15], (7.5 / sum(D_kt[k - 1])))  # 库存周转天数
        # expr.addTerms(k_price[k - 1], buhuo[k, t])  # 补货费用
model.setObjective(expr, COPT.MINIMIZE)

# 约束条件
# constraint1 - 库存约束
for t in T:
    if t >= 4:
        for k in K:
            model.addConstr(kucun[k, t] * big_M >= IB[k, t] + x[k, t - TL] - D_kt[k - 1][t - 1], name='库存量平衡约束1')
            model.addConstr((1 - kucun[k, t]) * big_M >= D_kt[k - 1][t - 1] - IB[k, t] - x[k, t - TL], name='库存量平衡约束1')
            model.addGenConstrIndicator(kucun[k, t], False, IE[k, t] == 0)
            model.addGenConstrIndicator(kucun[k, t], True, IE[k, t] == IB[k, t] + x[k, t - TL] - D_kt[k - 1][t - 1])
            model.addGenConstrIndicator(kucun[k, t], False, O[k, t] == D_kt[k - 1][t - 1] - IB[k, t] - x[k, t - TL])

    else:
        for k in K:
            model.addConstr(kucun[k, t] * big_M >= IB[k, t] - D_kt[k - 1][t - 1], name='库存量平衡约束2')
            model.addConstr((1 - kucun[k, t]) * big_M >= D_kt[k - 1][t - 1] - IB[k, t], name='库存量平衡约束2')
            model.addGenConstrIndicator(kucun[k, t], False, IE[k, t] == 0)
            model.addGenConstrIndicator(kucun[k, t], True, IE[k, t] == IB[k, t] - D_kt[k - 1][t - 1])
            model.addGenConstrIndicator(kucun[k, t], False, O[k, t] == D_kt[k - 1][t - 1] - IB[k, t])
# constraint2 - 期初库存约束
for k in K:
    model.addConstr(IB[k, 1] == 5, name='期初库存约束')
# constraint3 - 期初库存和期末库存约束
for k in K:
    for t in T:
        if t > 1:
            model.addConstr(IE[k, t - 1] == IB[k, t], name='期初期末库存约束')
for t in T:
    for k in K:
        # 使用内置函数
        model.addConstr(buhuo[k, t] * big_M >= v_s[k] - IE[k, t])
        model.addConstr((1 - buhuo[k, t]) * big_M >= IE[k, t] - v_s[k])
        model.addGenConstrIndicator(buhuo[k, t], True, x[k, t] == v_S[k] - IE[k, t])
        model.addGenConstrIndicator(buhuo[k, t], False, x[k, t] == 0)

# constraint 安全库存约束
for k in K:
    model.addConstr(v_S[k] >= v_s[k])
# 优化求解
model.solve()
print(model.objVal)
for sku_idx in K:
    print('IB:', list(map(int, [IB.select(sku_idx, i)[0].x for i in range(1, 16)])))
    print('IE:', list(map(int, [IE.select(sku_idx, i)[0].x for i in range(1, 16)])))
    print('D:', [i for i in D_kt[sku_idx - 1][:15]])
    print('x:', list(map(int, [x.select(sku_idx, i)[0].x for i in range(1, 16)])))
    print('v_s:', v_s.select(sku_idx)[0].x)
    print('v_S:', v_S.select(sku_idx)[0].x)
    print('O:', list(map(int, [O.select(sku_idx, i)[0].x for i in range(1, 16)])))
    print('补货:', list(map(int, [buhuo.select(sku_idx, i)[0].x for i in range(1, 16)])))
    # print('缺货:', list(map(int, [model.getVarByName('add_var2[{},{}]'.format(sku_idx, i)).x for i in range(1, 16)])))
    # print('库存:', list(map(int, [model.getVarByName('add_var3[{},{}]'.format(sku_idx, i)).x for i in range(1, 16)])))
    print('-----------------------------')
