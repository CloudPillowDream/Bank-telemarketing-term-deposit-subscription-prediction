# 葡萄牙银行电话营销：定期存款认购预测

基于 UCI Bank Marketing 数据集，预测客户是否会认购银行定期存款（`y = yes/no`）。

完整实验流程见 Notebook：`2024111712_章泽俊_期中作业.ipynb`

---

## 背景

- **任务**：二分类，预测客户是否认购定期存款。
- **数据**：4521 条样本，16 个特征，无缺失、无重复。
- **难点**：正类仅占 11.52%，Accuracy 会失真（全预测 "no" 也有 88.48%），因此主要看 Recall、F1、PR-AUC。
- **业务约束**：`duration`（通话时长）区分度最强，但只有打完电话才能获得，无法用于拨号前筛选，项目中做了专门对照。

---

## 数据探索要点

- `duration` 区分度最强：认购客户平均通话 553 秒，未认购 226 秒。
- `pdays = -1`（81.95%）表示"从未被联系过"，与 `poutcome = unknown` 完全对应，因此构造二元特征 `previously_contacted`。
- `balance`、`previous`、`campaign`、`duration`、`pdays` 偏度较大（> 2.5），建模前做幂变换。
- 类别特征中，退休人员 / 学生认购率最高（约 23%），蓝领最低（7.3%）；`default` 几乎无区分度。

---

## 预处理

- 8:2 分层划分，`random_state = 42`。
- 训练集 3616 条（正类 11.53%），测试集 905 条（正类 11.49%）。
- 所有拟合只在训练集上进行，预处理放入 `Pipeline`，交叉验证时每折独立拟合。
- 偏态数值特征用 Yeo-Johnson 变换，其余数值特征标准化，类别特征独热编码。

---

## 交叉验证

- RepeatedStratifiedKFold，5 折 × 3 次重复，共 15 次划分。
- 模型与参数选择只看训练集交叉验证，测试集仅用于最终报告。

---

## 基线模型结果（含 duration）

**训练集交叉验证**

| 模型 | CV F1 | CV ROC-AUC | CV PR-AUC | Train F1 |
|---|---|---|---|---|
| Logistic Regression | **0.4478** | **0.8942** | **0.5271** | 0.4826 |
| Decision Tree | 0.4433 | 0.6885 | 0.2619 | 1.0000 |
| SVM (RBF) | 0.2870 | 0.8782 | 0.5211 | 0.4212 |

**测试集**

| 模型 | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.5556 | 0.2885 | 0.3797 | **0.8918** | 0.4822 |
| Decision Tree | 0.3730 | **0.4519** | **0.4087** | 0.6766 | 0.2316 |
| SVM (RBF) | **0.6897** | 0.1923 | 0.3008 | 0.8740 | **0.4978** |

- 决策树训练 F1 = 1.0，CV F1 仅 0.44，明显过拟合。
- 逻辑回归排序能力最好；SVM Precision 高但 Recall 低。
- 默认阈值 0.5 下 Recall 普遍偏低，问题主要在阈值，而非排序能力。

---

## 决策树深度实验

候选深度：`1, 2, 3, 4, 5, 7, 10, 15, None`

- CV F1 最高为深度 2（0.4594）。
- 按 1-SE 规则（下界 0.4180），选最浅的 **`max_depth = 2`**，Train-CV gap 仅 0.02。
- 深度越大，训练 F1 升到 1.0，但 CV F1 停在 0.42–0.46，过拟合加剧。

> 若只看排序能力，深度 4 的 ROC-AUC / PR-AUC 最好；本作业关注默认阈值下的分类效果，故选深度 2。

**学习曲线（max_depth = 2）**

- 样本量 ≥ 1150 后，训练 F1 与验证 F1 基本重合，无明显过拟合。
- 验证 F1 随数据量缓慢上升后趋于平台，瓶颈在模型复杂度而非数据量。

---

## 实验二：duration 与类别权重

| 特征集 | 权重 | LR F1 | DT F1 | SVM F1 |
|---|---|---|---|---|
| 含 duration | original | 0.4478 | 0.4594 | 0.2870 |
| 含 duration | balanced | 0.5110 | 0.3632 | **0.5378** |
| 不含 duration | original | 0.2484 | 0.2522 | 0.2307 |
| 不含 duration | balanced | 0.3229 | 0.2465 | 0.3494 |

**结论**

1. **`duration` 影响巨大**：去掉后所有模型 ROC-AUC 从 0.88–0.89 降到 0.70–0.72，F1 从 0.45–0.54 降到 0.23–0.35。说明模型很大程度依赖通话结果，**拨号前无法使用 duration**。
2. **`class_weight="balanced"` 能显著提升 Recall**（LR 0.35 → 0.81，SVM 0.18 → 0.78），F1 也有提升，但 Precision 明显下降。本质上等价于阈值下移。

---

## 结论

1. 数据不平衡是核心挑战，评估必须以 Recall / F1 / PR-AUC 为主。
2. `duration` 最强但拨号前不可用，去掉后性能大幅下降。
3. `pdays = -1` 需构造 `previously_contacted` 单独处理。
4. 决策树 `max_depth = 2` 是稳健选择，学习曲线显示数据量已足够。
5. 阈值下移或 balanced 权重是提升 Recall 的关键手段。
6. 后续可尝试随机森林 / 梯度提升，并针对拨号前场景做特征工程。

---

## 项目结构

```
.
├── bank.csv
├── 2024111712_章泽俊_期中作业.ipynb
└── README.md
```

---

## 环境与复现

```
python >= 3.9
numpy pandas matplotlib seaborn scikit-learn jupyter
```

```bash
pip install numpy pandas matplotlib seaborn scikit-learn jupyter
jupyter notebook 2024111712_章泽俊_期中作业.ipynb
```

将 `bank.csv` 放在 Notebook 同目录下，按顺序执行全部单元格即可复现，所有随机过程固定 `random_state = 42`。
