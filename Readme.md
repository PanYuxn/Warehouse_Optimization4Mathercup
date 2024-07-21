
# 代码介绍
`copt_mode.py`:使用coptpy完成对模型的构建，这里没有设置多目标.

`GA4MatherCup.py`:用遗传算法求解仓储优化问题，目标好像就定了一个还是加权忘记了。

`gurobi_model.py`:使用gurobi完成对模型的搭建，这里设置了多目标，具体设了几个忘记了。

`ss_for_model.py`:为了比赛凑页数的灵敏度分析，没啥用不用看。

`src/...`：数据部分和赛题，好像就用了其中几个数据。这里新增了两列进行一下介绍。


| 列名                 | 含义                                                                       |
| ------------------ | ------------------------------------------------------------------------ |
| Category_for_Price | 对所有SKU价格进行排序后均分对价格量化成四种类型，分别为low、medium<br>、high和veryhigh.               |
| category           | 有intermittent、smooth、erratic和lumpy四种，这个意思是指<br>时间序列的特征，具体含义自己看，当时是为了第一题。 |



# 模型部分

## Notation

| 集合             | 集合含义                        |
| -------------- | --------------------------- |
| $T=15$         | 时间集合,$\{1,2,3...15\}$       |
| $K=29$         | 零售商集合,$\{1,2,3...29\}$      |
| $P=99$         | 产品集合,$\{1,2,3...99\}$       |
| $W=36$         | 仓库集合,$\{1,2,3,...36\}$      |
| **参数**         | **参数含义**                    |
| $NRT$          | 审查期，没NRT天对库存进行一次盘点          |
| $LT$           | 提前期，商品需要经过LT天到货             |
| $SKU_{k,p,w}$  | 表示将零售商K售卖的仓库W的商品P视为一个SKU    |
| $IC_{k,p,w}$   | 表示$SKU_{k,p,w}$的持有成本率       |
| $SC_{k,p,w}$   | 表示$SKU_{k,p,w}$的缺货成本率       |
| $P_{p}$        | 表示商品p的价格                    |
| $SP_{k,p,w}$   | 表示订购一次$SKU_{k,p,w}$的订购费用    |
| $D_{k,p,w,t}$  | 表示$SKU_{k,p,w}$在第t天的需求      |
| $capa_{w}$     | 表示仓库w的最大库存容量                |
| M              | 极大数                         |
| **决策变量**       | **变量含义**                    |
| $IB_{k,p,w,t}$ | 表示$SKU_{k,p,w}$在第t天的期初库存水平  |
| $IE_{k,p,w,t}$ | 表示$SKU_{k,p,w}$在第t天的期末库存水平  |
| $O_{k,p,w,t}$  | 表示$SKU_{k,p,w}$在第t天的缺货量     |
| $Q_{k,p,w,t}$  | 表示$SKU_{k,p,w}$在第t天的补货量     |
| $S_{k,p,w}$    | 表示$SKU_{k,p,w}$在周期内的最大库存    |
| $s_{k,p,w}$    | 表示$SKU_{k,p,w}$在周期内的订货点     |
| $il_{k,p,w,t}$ | 表示$SKU_{k,p,w}$在第t天是否处于缺货状态 |
| $ip_{k,p,w,t}$ | 表示$SKU_{k,p,w}$在第天天是否进行补货   |

## Asuumption
该模型的假设主要包括成本、审查期、提前期、库存四个方面，具体假设如下：

假设1：商品的持有成本和缺货成本均与商品的价格呈正比；

假设2：每种商品的订购成本与商品的价格呈正比；

假设3：商家在期末进行库存盘点，并决定是否进行采购；

假设4：交货期是确定的，不存在订单交叉现象；

假设5：每次订购的商品经过提前期时间，在期初到货；

假设6：前一天的期末库存等于后一天的期初库存；

假设7：本问考虑的库存优化问题为单阶段库存问题，不考虑商品物流存在多个运输节点。

## Model Formulation
**目标函数**
1. 持有成本和缺货成本

$$f_{1}=\sum_{t}\sum_{k}\sum_{p}\sum_{w}P_{p}(IC_{k,p,w}IE_{k,p,w,t}+SC_{k,p,w}O_{k,p,w})$$

2. 库存周转天数

$$f_{2}=\sum_{k}\sum_{p}\sum_{w}\frac{(IB_{k,p,w,l}+IE_{k,p,w,t})}{2}\cdot\frac{T}{\sum_{t}D_{k,p,w,t}}$$

3. 订购成本

$$f_{3}=\sum_{k}\sum_{p}\sum_{w}SP_{p}\sum_{t}ip_{k,p,w,t}$$

**约束条件**
1. 期初库存约束

$$IB_{k,p,w,1}=5,\forall k\in K,\forall p \in P ,\forall w \in W$$

2. 商品缺货量约束

![img.png](src%2Fimg.png)

3. 库存更新约束

![img_1.png](src%2Fimg_1.png)

4. 商品采购约束

![img_2.png](src%2Fimg_2.png)

5. 仓库容量约束

$$\sum_{k}\sum_{p}(IE_{k,p,w,t}+Q_{k,p,w,t})\le capa_{w},\forall t\in T,\forall w \in W$$

