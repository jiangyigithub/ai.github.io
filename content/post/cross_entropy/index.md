(4)    # 数字
(4,)   # 元组

logits 是模型对每个类别的“打分”

softmax 才把打分变成概率

PyTorch 要用元组
.item() 把只有一个元素的 tensor 变成普通 Python 数字

torch.randn(4,) # 一维
torch.randn(4, 10) # 二维
torch.randn(2, 3, 5) # 三维

CrossEntropyLoss(logits, target) = - log_softmax(logits)[target]
LogSoftmax = logits − LogSumExp
LogSoftmax(z)=z−LogSumExp(z)

torch.randint(low, high, size)

```python
import torch
def cross_entropy_loss(logits, targets):
    log_probs = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
    return -log_probs[torch.arange(targets.shape[0]), targets].mean()

logits = torch.randn(4, 10)
targets = torch.randint(0, 10, (4,))
print('Loss:', cross_entropy_loss(logits, targets).item())
print('Ref: ', torch.nn.functional.cross_entropy(logits, targets).item())
```


好，我们**不用公式**，直接用一个具体数字例子手算一遍，你马上就懂了。

---

## 🎯 假设有一个 3 分类问题

某一个 batch 里有 2 个样本。

模型输出（logits）是：

```python
logits =
[[2.0, 1.0, 0.1],
 [0.5, 2.5, 0.3]]
```

真实标签：

```python
targets = [0, 1]
```

意思是：

* 第 1 个样本 → 正确类别是 0
* 第 2 个样本 → 正确类别是 1

---

# 第一步：把 logits 变成概率（softmax）

### 第一个样本

```
[2.0, 1.0, 0.1]
```

softmax 后大概是：

```
[0.659, 0.242, 0.099]
```

意思是：

* 类别 0 → 65.9%
* 类别 1 → 24.2%
* 类别 2 → 9.9%

因为真实标签是 0
所以我们只关心 **0.659**

---

### 第二个样本

```
[0.5, 2.5, 0.3]
```

softmax 后大概是：

```
[0.106, 0.787, 0.107]
```

真实标签是 1
所以我们只关心 **0.787**

---

# 第二步：取 log

交叉熵就是：

> 真实类别概率越大 → loss 越小
> 真实类别概率越小 → loss 越大

公式其实就是：

```
loss = - log(真实类别的概率)
```

算一下：

```
样本1: -log(0.659) ≈ 0.417
样本2: -log(0.787) ≈ 0.239
```

---

# 第三步：求平均

```
(0.417 + 0.239) / 2 = 0.328
```

这就是最终 loss。

---

# 🔥 那你的代码在干嘛？

它干了这 4 件事：

1. logits → 变成 log_softmax
2. 只拿真实类别那一列
3. 取负号
4. 求平均

---

# 🧠 直觉理解

交叉熵本质就是一句话：

> “模型对正确答案的自信程度有多高？”

* 如果正确类别概率 = 0.99 → loss 很小
* 如果正确类别概率 = 0.01 → loss 巨大

---

# 🎯 再用一句人话解释代码

```python
return -log_probs[torch.arange(targets.shape[0]), targets].mean()
```

等价于
```python
for i in range(batch_size):
    result[i] = log_probs[i][targets[i]]
```
意思是：

> 把每个样本正确类别的 log 概率拿出来，
> 变成负数，然后算平均。

---


好问题 👍
“向量化索引”一开始确实容易晕，但其实有一条**非常简单的记忆规则**。

我给你一个 **一条核心口诀 + 三个固定套路**，以后基本不会错。

---

# 🌟 一条核心口诀

> **两个索引张量形状相同 → 就是“成对取坐标”**

```python
A[rows, cols]
```

等价于：

```python
for i in range(len(rows)):
    A[rows[i], cols[i]]
```

---

# 一、交叉熵那句怎么记？

```python
log_probs[torch.arange(N), targets]
```

记住：

```text
arange(N)  = 行号
targets    = 列号
```

意思是：

```text
第 i 行，取 targets[i] 列
```

---

# 二、最常见的 3 种向量化规则

---

## ✅ 规则 1：单维索引（选行）

```python
A[rows]
```

等价于：

```python
A[rows, :]
```

例子：

```python
A[[0,2]]
```

选第 0 行和第 2 行。

---

## ✅ 规则 2：两个索引 → 成对取元素

```python
A[rows, cols]
```

如果：

```python
rows = [0,1,2]
cols = [2,0,1]
```

那取：

```python
A[0,2]
A[1,0]
A[2,1]
```

记忆法：

> “拉链式匹配”

---

## ✅ 规则 3：广播规则

如果一个是标量或长度1：

```python
A[rows, 1]
```

等价于：

```python
for i in rows:
    A[i, 1]
```

---

# 三、一个万能记忆图

想象：

```text
行索引 = 竖着的指针
列索引 = 横着的指针
```

当你写：

```python
A[rows, cols]
```

就是：

```text
rows[i] 和 cols[i] 组成一个坐标
```

---

# 四、交叉熵为什么必须这样？

因为我们要做：

[
\log p_{y_i}
]

也就是：

```text
第 i 个样本
取 第 y_i 类
```

向量化后变成：

```python
log_probs[arange(N), targets]
```

---

# 五、终极记忆公式

把这句背下来：

> **A[x, y] → 取 (x[i], y[i]) 这些坐标点**

只要 x 和 y 是同形状向量，就是“配对取值”。

---

# 六、如果你想更牢固记住

每次看到：

```python
A[rows, cols]
```

就在脑子里自动翻译成：

```python
for i in range(len(rows)):
    A[rows[i], cols[i]]
```

这样你永远不会错。

---

如果你愿意，我可以再给你出 3 个小练习（非常短），你做完就彻底掌握向量化索引。



