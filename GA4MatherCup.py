import random
import pandas as pd
from deap import base, creator, tools, algorithms

df = pd.read_excel('src/data_category.xlsx')
# 定义问题参数
DAYS = 15
PRODUCTS = len(df)

# 定义遗传算法参数
POPULATION_SIZE = 100
P_CROSSOVER = 0.8
P_MUTATION = 0.1
MAX_GENERATIONS = 50


# 定义个体生成函数
def create_individual():
    individual = []
    for i in range(PRODUCTS):
        data = df.iloc[i, :]
        demand = data[[i for i in range(1, 16)]].to_list()
        s = max(0, (data['Mean_for_ss'] + random.randint(-5, 5)))
        S = random.randint(s, sum(sorted(demand)[-3:]) + s)
        if s > S:  # 避免出现s>S的情况.
            s = 0
        total_demand = sum(demand[3:])  # 计算除前3天外的所有需要补货量.
        demand_ub = max(0, sum(sorted(demand[3:])[-3:]) - 5)  # 生成x_values时可能存在的upper bound
        s_S_gap = S - s
        nums = [0] * 12
        current_sum = 0
        while current_sum < total_demand:
            for i in range(12):
                # Randomly choose how much to add to each element, without exceeding demand_ub
                add = random.randint(0, max(s_S_gap, demand_ub) + 5)
                nums[i] += add
                current_sum = sum(nums)
                if current_sum > total_demand:
                    break
        x_values = nums + [0, 0, 0]
        individual.extend([s, S] + x_values)
    return creator.Individual(individual)


def evaluate_individual(sub_individual, count):
    # 在这里实现成本计算逻辑
    ts_flag = 1  # 用来标记选择那个类别, 1-ts分类,2-price分类
    price_para = {'low': 0.6, 'medium': 0.6, 'high': 0.6, 'veryhigh': 0.6, 'intermittent': 0.6, 'erratic': 0.6,
                  'lumpy': 0.6, 'smooth': 0.6, 'order_rate': 0.8}
    s, S, x_values = sub_individual[0], sub_individual[1], sub_individual[2:]
    data = df.iloc[count, :]
    demand = data[[i for i in range(1, 16)]].to_list()
    if ts_flag == 1:
        price_less = data['price'] * price_para[data['category']]  # 缺货成本.
        price_hold = data['price'] * (1 - price_para[data['category']])  # 持有成本价格.
    else:
        price_less = data['price'] * price_para[data['category']]  # 缺货成本.
        price_hold = data['price'] * (1 - price_para[data['Category_for_Price']])  # 持有成本价格.
    # 更新1-15天的库存量和缺货量数据，用来计算cost
    lead_time = 3
    IB = [0] * 15  # 期初库存
    IE = [0] * 15  # 期末库存
    O = [0] * 15  # 缺货量
    incoming_supply = [0] * 15  # 来货量
    IB[0] = 5
    for i in range(15):
        # 加入订货提前期的处理
        if i >= lead_time:
            incoming_supply[i] = x_values[i - lead_time]
        # 更新期初库存
        if i > 0:
            IB[i] = IE[i - 1]
        # 更新缺货量
        available_inventory = IB[i] + incoming_supply[i] - demand[i]
        O[i] = max(0, -available_inventory)
        # 更新期末库存
        IE[i] = max(0, available_inventory)
    # 开始计算各个成本
    cost1 = sum(IE) * price_hold + sum(O) * price_less  # 持有成本
    cost2 = sum([1 for i in x_values if i >= 1]) * data['price'] * price_para['order_rate']  # 订购成本
    cost3 = ((5 + IE[-1]) / 2) * (15 / sum(demand)) if sum(demand) != 0 else 0  # 库存周转天数
    cost = cost1 + cost2 + cost3
    return cost,


# 定义适应度函数
def evaluate(individual):
    total_cost = 0  # 记录总成本.
    count = 0  # 利用count记录第几个SKU单品.
    for i in range(0, len(individual), 17):
        sub_individual = individual[i:i + 17]  # 拆分成单个列表进行分析
        total_cost += evaluate_individual(sub_individual, count)[0]
        count += 1
    return total_cost,


def myCrossover(ind1, ind2):
    # 将两个个体当中以17为长度进行交叉
    size = min(len(ind1), len(ind2))
    unit_size = 17  # 单个产品的编码长度

    # 确保交叉点是单元的边界
    cxpoint1 = random.randint(0, size // unit_size) * unit_size
    cxpoint2 = random.randint(0, size // unit_size) * unit_size

    if cxpoint2 >= cxpoint1:
        cxpoint2 += unit_size
    else:
        cxpoint1, cxpoint2 = cxpoint2, cxpoint1 + unit_size

    ind1[cxpoint1:cxpoint2], ind2[cxpoint1:cxpoint2] = ind2[cxpoint1:cxpoint2], ind1[cxpoint1:cxpoint2]

    return ind1, ind2


def myMutation(individual):
    unit_size = 17  # 单个产品的编码长度
    num_units = len(individual) // unit_size

    # 随机选择一个单元进行变异
    unit_to_mutate = random.randint(0, num_units - 1)
    start = unit_size * unit_to_mutate
    end = start + unit_size
    # 加载当前SKU基本信息
    data = df.iloc[unit_to_mutate, :]
    demand = data[[i for i in range(1, 16)]].to_list()
    # 变异 s 和 S
    s = individual[start]
    S = individual[start + 1]
    s_new = random.randint(s, s + data['Mean_for_ss'])
    S_new = random.randint(S, sum(sorted(demand)[-3:]) + S)
    # 后续库存量修改
    total_demand = sum(demand[3:])  # 计算除前3天外的所有需要补货量.
    demand_ub = max(0, sum(sorted(demand[3:])[-3:]) - 5)  # 生成x_values时可能存在的upper bound
    s_S_gap = S - s
    nums = [0] * 12
    current_sum = 0
    while current_sum < total_demand:
        for i in range(12):
            # Randomly choose how much to add to each element, without exceeding demand_ub
            add = random.randint(0, max(s_S_gap, demand_ub) + 5)
            nums[i] += add
            current_sum = sum(nums)
            if current_sum > total_demand:
                break
    x_values = nums + [0, 0, 0]
    # 替换当前个体中的数字
    individual[start] = s_new
    individual[start + 1] = S_new
    individual[start + 2:end] = x_values  # 确保补货量在新的 s 和 S 之间

    return individual,


# 定义个体类
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)
# 初始化遗传算法工具箱
toolbox = base.Toolbox()
toolbox.register("individualCreator", create_individual)
toolbox.register("populationCreator", tools.initRepeat, list, toolbox.individualCreator)
# 定义算法中各个函数主体
toolbox.register("evaluate", evaluate)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", myCrossover)
toolbox.register("mutate", myMutation)

# 创建初始种群并运行遗传算法
population = toolbox.populationCreator(n=POPULATION_SIZE)
stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("max", max)
# stats.register("avg", lambda x: sum(x) / len(x) if x else 0)
stats.register("avg", lambda x: sum(fit[0] for fit in x) / len(x) if x else 0)
hall_of_fame = tools.HallOfFame(maxsize=10)

population, logbook = algorithms.eaSimple(population, toolbox, cxpb=P_CROSSOVER, mutpb=P_MUTATION, ngen=MAX_GENERATIONS,
                                          stats=stats, halloffame=hall_of_fame, verbose=True)

# 输出结果
best_individual = tools.selBest(population, k=1)[0]
print(f"Best Individual: {best_individual}, Fitness: {best_individual.fitness.values[0]}")


# 结果可视化
def get_total_info(initial_inventory, demand, lead_time, s, S, x_values):
    # 初始化变量
    IB = [0] * 15  # 期初库存
    IE = [0] * 15  # 期末库存
    O = [0] * 15  # 缺货量
    incoming_supply = [0] * 15  # 来货量
    # 计算
    IB[0] = initial_inventory
    for i in range(15):
        # 加入订货提前期的处理
        if i >= lead_time:
            incoming_supply[i] = x_values[i - lead_time]
        # 更新期初库存
        if i > 0:
            IB[i] = IE[i - 1]
        # 更新缺货量
        available_inventory = IB[i] + incoming_supply[i] - demand[i]
        O[i] = max(0, -available_inventory)
        # 更新期末库存
        IE[i] = max(0, available_inventory)
    return IB, IE, O


unit_size = 17
# 分割成100个子列表
sublists = []
for i in range(0, len(best_individual), unit_size):
    sublist = best_individual[i:i + unit_size]
    sublists.append(sublist)
for idx, data in enumerate(sublists):
    seller_no, product_no, warehouse_no = df.loc[idx, ['seller_no', 'product_no', 'warehouse_no']]
    s, S, x_values = data[0], data[1], data[2:]
    demand = df.iloc[idx, :][[i for i in range(1, 16)]].to_list()
    IB, IE, O = get_total_info(initial_inventory=5, demand=demand, lead_time=3, s=s, S=S, x_values=x_values)
    print(f'当前产品为：{seller_no},{product_no},{warehouse_no}')
    print('IB:', IB)
    print('ID:', IE)
    print('D:', demand)
    print('x:', x_values)
    print('s:', s)
    print('S:', S)