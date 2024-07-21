import pandas as pd
from gurobipy import *
from datetime import datetime, timedelta


def get_data(path):
    df = pd.read_excel(path)
    T = 15
    TL = 3
    C_sh = [0.2, 0.2, 0.2, 0.2]  # 4个不同种类的持有成本比率
    C_sp = [0.8, 0.8, 0.8, 0.8]  # 4个不同种类的缺货成本比率
    center_wh_capacity = 20000 * 0.8
    district_wh_capacity = 12000 * 0.8
    k_price = list(df['price'])
    D_kt_col = [i for i in range(1, 19)]
    ss_k_col = ["Mean_for_ss"]
    D_kt = df.loc[:, D_kt_col].values
    ss_k = df.loc[:, ss_k_col].values
    category = df["Category_for_Price"].to_list()
    wh = df["warehouse_no"].to_list()
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
K = [i for i in range(1, len(df_data))]
S = [1, 2, 3, 4]
T = [i for i in range(1, 16)]
model = Model()
big_M = 10000
# 决策变量

IB = model.addVars(K, T, lb=0, ub=GRB.INFINITY, vtype=GRB.INTEGER, name='IB')
IE = model.addVars(K, T, lb=0, ub=GRB.INFINITY, vtype=GRB.INTEGER, name='IE')
x = model.addVars(K, T, lb=0, ub=GRB.INFINITY, vtype=GRB.INTEGER, name='x')
v_s = model.addVars(K, lb=0, ub=GRB.INFINITY, vtype=GRB.INTEGER, name='v_s')
v_S = model.addVars(K, lb=0, ub=GRB.INFINITY, vtype=GRB.INTEGER, name='v_S')
O = model.addVars(K, T, lb=0, ub=GRB.INFINITY, vtype=GRB.INTEGER, name='O')
# additional varbial
buhuo = model.addVars(K, T, vtype=GRB.BINARY, name='buhuo')
quehuo = model.addVars(K, T, vtype=GRB.BINARY, name='quehuo')
kucun = model.addVars(K, T, vtype=GRB.BINARY, name='kucun')
# 目标函数
obj = LinExpr()
obj1 = LinExpr()
for t in T:
    for k in K:
        if k - 1 in low:
            obj.addTerms(k_price[k - 1] * C_sh[0], IE[k, t])  # 分组A持有成本
            obj.addTerms(k_price[k - 1] * C_sp[0], O[k, t])  # 分组A缺货成本
        if k - 1 in medium:
            obj.addTerms(k_price[k - 1] * C_sh[1], IE[k, t])  # 分组B持有成本
            obj.addTerms(k_price[k - 1] * C_sp[1], O[k, t])  # 分组B缺货成本
        if k - 1 in high:
            obj.addTerms(k_price[k - 1] * C_sh[2], IE[k, t])  # 分组C持有成本
            obj.addTerms(k_price[k - 1] * C_sp[2], O[k, t])  # 分组C缺货成本
        if k - 1 in veryhigh:
            obj.addTerms(k_price[k - 1] * C_sh[3], IE[k, t])  # 分组D持有成本
            obj.addTerms(k_price[k - 1] * C_sp[3], O[k, t])  # 分组D缺货成本
        obj.addTerms(k_price[k - 1] * 0.8, buhuo[k, t])  # 补货费用乘以系数.
for k in K:
    if sum(D_kt[k - 1]) != 0:
        obj1.addTerms((7.5 / sum(D_kt[k - 1])), IB[k, 1])  # 库存周转天数
        obj1.addTerms((7.5 / sum(D_kt[k - 1])), IE[k, 15])  # 库存周转天数
# model.setObjective(obj+obj1,GRB.MINIMIZE)
model.setObjectiveN(obj1, index=0, priority=1, weight=1)
model.setObjectiveN(obj, index=1, priority=1, weight=1)

# 约束条件
# constraint1 - 库存约束
for t in T:
    if t >= 4:
        for k in K:
            model.addConstr(kucun[k, t] * big_M >= IB[k, t] + x[k, t - TL] - D_kt[k - 1][t - 1], name='库存量平衡约束1')
            model.addConstr((1 - kucun[k, t]) * big_M >= D_kt[k - 1][t - 1] - IB[k, t] - x[k, t - TL],
                            name='库存量平衡约束1')
            model.addGenConstrIndicator(kucun[k, t], False, IE[k, t] == 0, name='库存量平衡约束1')
            model.addGenConstrIndicator(kucun[k, t], True, IE[k, t] == IB[k, t] + x[k, t - TL] - D_kt[k - 1][t - 1],
                                        name='库存量平衡约束1')
            model.addGenConstrIndicator(kucun[k, t], False, O[k, t] == D_kt[k - 1][t - 1] - IB[k, t] - x[k, t - TL],
                                        name='缺货量约束1')

    else:
        for k in K:
            model.addConstr(kucun[k, t] * big_M >= IB[k, t] - D_kt[k - 1][t - 1], name='库存量平衡约束2')
            model.addConstr((1 - kucun[k, t]) * big_M >= D_kt[k - 1][t - 1] - IB[k, t], name='库存量平衡约束2')
            model.addGenConstrIndicator(kucun[k, t], False, IE[k, t] == 0, name='库存量平衡约束2')
            model.addGenConstrIndicator(kucun[k, t], True, IE[k, t] == IB[k, t] - D_kt[k - 1][t - 1],
                                        name='库存量平衡约束2')
            model.addGenConstrIndicator(kucun[k, t], False, O[k, t] == D_kt[k - 1][t - 1] - IB[k, t],
                                        name='缺货量约束2')
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

model.write('model.lp')
model.optimize()
total_obj = 0
for i in range(model.NumObj):
    model.setParam(GRB.Param.ObjNumber, i)
    print('Obj%d = ' % (i + 1), model.ObjNVal)
    total_obj += model.ObjNVal
print('Total_obj = ', total_obj)
result_dict = {
    'seller_no': [],
    'product_no': [],
    'warehouse_no': [],
    'date': [],
    'lower_s': [],
    'upper_s': [],
    'inventory_begin': [],
    'inventory_end': [],
    'forecast_qty': [],
    'replenish_qty': []
}

start_date = datetime(2023, 5, 16)
end_date = datetime(2023, 5, 30)
date_list = []
current_date = start_date
while current_date <= end_date:
    date_list.append(current_date.strftime('%Y-%m-%d'))
    current_date += timedelta(days=1)

for sku_idx in K:
    IB_ls = list(map(int, [model.getVarByName('IB[{},{}]'.format(sku_idx, i)).x for i in range(1, 16)]))
    IE_ls = list(map(int, [model.getVarByName('IE[{},{}]'.format(sku_idx, i)).x for i in range(1, 16)]))
    D_ls = [i for i in D_kt[sku_idx - 1]]
    x_ls = list(map(int, [model.getVarByName('x[{},{}]'.format(sku_idx, i)).x for i in range(1, 16)]))
    v_s_ls = list(map(int, [model.getVarByName('v_s[{}]'.format(sku_idx)).x] * 15))
    v_S_ls = list(map(int, [model.getVarByName('v_S[{}]'.format(sku_idx)).x] * 15))
    O_ls = list(map(int, [model.getVarByName('O[{},{}]'.format(sku_idx, i)).x for i in range(1, 16)]))
    start_date = datetime(2023, 5, 16)
    end_date = datetime(2023, 5, 30)
    for t in range(len(IB_ls)):
        seller_no, product_no, warehouse_no = df_data.iloc[sku_idx - 1, [1, 2, 3]].to_list()
        result_dict['seller_no'].append(seller_no)
        result_dict['product_no'].append(product_no)
        result_dict['warehouse_no'].append(warehouse_no)
        result_dict['date'].append(date_list[t])
        result_dict['lower_s'].append(v_s_ls[t])
        result_dict['upper_s'].append(v_S_ls[t])
        result_dict['inventory_begin'].append(IB_ls[t])
        result_dict['inventory_end'].append(IE_ls[t])
        result_dict['forecast_qty'].append(D_ls[t])
        result_dict['replenish_qty'].append(x_ls[t])
result_df = pd.DataFrame(result_dict)
print(result_df)
# result_df.to_excel('output.xlsx')
