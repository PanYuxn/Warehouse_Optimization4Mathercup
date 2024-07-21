import pandas as pd
from gurobipy import *
import numpy as np


def get_data(path):
    df = pd.read_excel(path)
    T = 15
    TL = 3
    center_wh_capacity = 20000 * 0.8
    district_wh_capacity = 12000 * 0.8
    k_price = list(df['price'])
    D_kt_col = [i for i in range(1, 19)]
    ss_k_col = ["Mean_for_ss"]
    D_kt = df.loc[:, D_kt_col].values
    ss_k = df.loc[:, ss_k_col].values
    category = df["Category_for_Price"].to_list()
    wh = df["warehouse_no"].to_list()
    return df, T, TL, k_price, D_kt, ss_k, category, wh, center_wh_capacity, district_wh_capacity


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


def MIP_model(path, C_sh, C_sp, C_order):
    df_data, Time, TL, k_price, D_kt, ss_k, category, wh, center_wh_capacity, district_wh_capacity = get_data(
        path)
    low, medium, high, veryhigh = get_category(category)
    # K = [i for i in range(1, len(k_price) + 1)]
    # K = [i for i in range(1, len(k_price) + 1)]
    K = [i for i in range(1, 100)]
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
            obj.addTerms(k_price[k - 1] * C_order, buhuo[k, t])  # 补货费用乘以系数.
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
                model.addConstr(kucun[k, t] * big_M >= IB[k, t] + x[k, t - TL] - D_kt[k - 1][t - 1],
                                name='库存量平衡约束1')
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

    model.setParam('OutputFlag', 0)
    model.write('model.lp')
    model.optimize()
    return model, df_data, K


def get_result(model, df_data, K, C_sh, C_sp, C_order, price_t):
    obj_ls = []
    for i in range(model.NumObj):
        model.setParam(GRB.Param.ObjNumber, i)
        obj_ls.append(model.ObjNVal)
    total_obj = sum(obj_ls)
    obj_ls.append(total_obj)
    total_inventory_cycle_day, total_hold_less_cost, total_obj = obj_ls
    result_dict = {
        'seller_no': [],
        'product_no': [],
        'warehouse_no': [],
        'O': [],
        'IE': [],
        'type': [],
        'demand': [],
        'less_price': [],
        'hold_price': [],
        'bu_huo_num': [],
        'order_price': [],
        'IE_last': []
    }

    price_type_map = {'low': 0, 'medium': 1, 'high': 2, 'veryhigh': 3}
    for sku_idx in K:
        IE_ls = list(map(int, [model.getVarByName('IE[{},{}]'.format(sku_idx, i)).x for i in range(1, 16)]))
        O_ls = list(map(int, [model.getVarByName('O[{},{}]'.format(sku_idx, i)).x for i in range(1, 16)]))
        buhuo = list(map(int, [model.getVarByName('buhuo[{},{}]'.format(sku_idx, i)).x for i in range(1, 16)]))
        seller_no, product_no, warehouse_no, price, price_type = df_data.iloc[sku_idx - 1, [1, 2, 3, 4, -2]].to_list()
        result_dict['demand'].append(sum(df_data.iloc[sku_idx - 1, 5:20].to_list()))
        result_dict['IE_last'].append(IE_ls[-1])
        result_dict['bu_huo_num'].append(sum(buhuo))
        result_dict['seller_no'].append(seller_no)
        result_dict['product_no'].append(product_no)
        result_dict['warehouse_no'].append(warehouse_no)
        result_dict['type'].append(price_type)
        result_dict['order_price'].append(price * C_order)
        result_dict['less_price'].append(price * C_sh[price_type_map[price_type]])
        result_dict['hold_price'].append(price * C_sp[price_type_map[price_type]])
        result_dict['O'].append(sum(O_ls))
        result_dict['IE'].append(sum(IE_ls))
    result_df = pd.DataFrame(result_dict)
    result_df['hold_cost'] = result_df['hold_price'] * result_df['IE']
    result_df['less_cost'] = result_df['less_price'] * result_df['O']
    result_df['order_cost'] = result_df['bu_huo_num'] * result_df['order_price']
    total_mean_service_level = (sum(result_df['demand'].to_list()) - sum(result_df['O'].to_list())) / sum(
        result_df['demand'].to_list())
    result_df['total_mean_service_level'] = total_mean_service_level
    total_hold_less_cost = int(sum(list(result_df['hold_cost']))) + int(sum(list(result_df['less_cost'])))
    df_price = result_df[result_df['type'] == price_t]
    hold_cost = int(sum(list(df_price['hold_cost'])))
    less_cost = int(sum(list(df_price['less_cost'])))
    order_sum = int(sum(list(df_price['order_cost'])))
    inventory_cycle_day = (5 * len(df_price) + sum(df_price['IE_last'].to_list())) * (
            15 / sum(df_price['demand'].to_list()))
    service_level = np.mean(((df_price['demand'] - df_price['O']) / df_price['demand']).to_list())
    result_df[f'{price_t}_hold_cost'] = hold_cost
    result_df[f'{price_t}_less_cost'] = less_cost
    result_df[f'{price_t}_hold_and_less_cost'] = less_cost + hold_cost
    result_df[f'{price_t}_order_cost'] = order_sum
    result_df[f'{price_t}_service_level'] = service_level
    result_df[f'{price_t}_inventory_cycle_day'] = inventory_cycle_day
    result_df['total_mean_inventory_cycle_day'] = total_inventory_cycle_day / len(K)
    result_df['total_inventory_cycle_day'] = total_inventory_cycle_day
    result_df['total_hold_less_cost'] = total_hold_less_cost
    result_df['total_obj'] = total_obj
    result_df['total_order_cost'] = int(sum(list(result_df['order_cost'])))
    result_df[f'{price_t}_C_sh'] = C_sh[price_type_map[price_t]]
    result_df[f'{price_t}_C_sp'] = C_sp[price_type_map[price_t]]
    metric_col = [f'{price_t}_C_sh', f'{price_t}_C_sp', 'total_obj',
                  'total_hold_less_cost', f'{price_t}_hold_and_less_cost', 'total_order_cost', f'{price_t}_order_cost',
                  'total_mean_service_level',
                  f'{price_t}_service_level', f'{price_t}_inventory_cycle_day', 'total_inventory_cycle_day',
                  'total_mean_inventory_cycle_day']
    result_df = result_df.loc[:, metric_col]
    return result_df[metric_col].iloc[0, :].values


if __name__ == "__main__":
    path = r'C:\Users\Lenovo\Python_jupyter\竞赛\MotherCup\MotherCupAgain\src\data_Category_for_Price.xlsx'
    price_t_ls = ['low', 'medium', 'high', 'veryhigh']
    rate = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    for idx, price_t in enumerate(price_t_ls):
        C_sh = [0.2, 0.2, 0.2, 0.2]  # 4个不同种类的持有成本比率,分别为low,medium,high,very_high
        C_sp = [0.8, 0.8, 0.8, 0.8]  # 4个不同种类的缺货成本比率
        C_order = 0.8
        print(f'price种类为:{price_t}')
        val_ls = []
        for i in range(len(rate)):
            C_sh[idx] = rate[i]
            C_sp[idx] = 1 - rate[i]
            print(f'当前持有成本为{C_sh}')
            print(f'当前缺货成本为{C_sp}')
            model, df_data, K = MIP_model(path, C_sh, C_sp, C_order)
            val = get_result(model, df_data, K, C_sh, C_sp, C_order, price_t)
            val_ls.append(val)
        val_df = pd.DataFrame(val_ls, columns=[f'{price_t}_C_sh', f'{price_t}_C_sp', 'total_obj',
                                               'total_hold_less_cost', f'{price_t}_hold_and_less_cost',
                                               'total_order_cost', f'{price_t}_order_cost',
                                               'total_mean_service_level',
                                               f'{price_t}_service_level', f'{price_t}_inventory_cycle_day',
                                               'total_inventory_cycle_day',
                                               'total_mean_inventory_cycle_day'])
        val_df.to_excel(f'{price_t}-持有成本系数分析.xlsx')
    C_order_ls = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    c_order_val = np.array([])
    for c_order in C_order_ls:
        C_sh = [0.2, 0.2, 0.2, 0.2]  # 4个不同种类的持有成本比率,分别为low,medium,high,very_high
        C_sp = [0.8, 0.8, 0.8, 0.8]  # 4个不同种类的缺货成本比率
        model, df_data, K = MIP_model(path, C_sh, C_sp, c_order)
        val = get_result(model, df_data, K, C_sh, C_sp, C_order, price_t)
        c_order_val.append(val)
    c_order_df = pd.DataFrame(c_order_val, columns=['C_order', 'total_obj',
                                  'total_hold_less_cost', f'{price_t}_hold_and_less_cost',
                                  'total_order_cost', f'{price_t}_order_cost',
                                  'total_mean_service_level',
                                  f'{price_t}_service_level', f'{price_t}_inventory_cycle_day',
                                  'total_inventory_cycle_day',
                                  'total_mean_inventory_cycle_day'])
    c_order_df.to_excel('订单成本系数分析.xlsx')