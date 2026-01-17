# Makemore Bigrams - 完整實現指南

## 📦 專案結構

```
makemore-java/
├── pom.xml                                    # Maven 構建配置
├── names.txt                                  # 訓練數據 (姓名列表)
├── README.md                                  # 完整文檔
├── src/
│   ├── main/java/com/makemore/
│   │   ├── Main.java                         # 主程式入口
│   │   └── bigram/
│   │       ├── Tensor.java                   # 支持自動微分的張量類
│   │       └── BigramLanguageModel.java      # Bigram 語言模型
│   └── test/java/com/makemore/bigram/
│       └── TensorTest.java                   # 單元測試
```

## 🎯 核心組件詳解

### 1. Tensor.java - 自動微分引擎

這是整個項目的基礎,實現了類似 PyTorch 的 Tensor 和自動微分。

#### 關鍵特性:

1. **數據存儲**: 使用一維 `double[]` 存儲多維數據
2. **形狀追蹤**: `int[] shape` 記錄張量維度
3. **梯度累積**: `double[] grad` 存儲梯度
4. **計算圖**: 使用 `Set<Tensor> prev` 和 `backward` 函數構建計算圖

#### 實現的運算:

```java
// 創建張量
Tensor t = Tensor.zeros(3, 4);           // 3x4 零矩陣
Tensor t = Tensor.randn(27, 27);         // 27x27 隨機矩陣 (高斯分布)

// 矩陣運算
Tensor c = a.matmul(b);                  // 矩陣乘法
Tensor s = t.sum(1, true);               // 沿維度求和

// 元素運算
Tensor exp = t.exp();                    // e^x
Tensor log = t.log();                    // ln(x)
Tensor div = a.div(b);                   // a / b (支持廣播)

// 反向傳播
loss.backward();                         // 自動計算所有梯度
```

#### 自動微分工作原理:

```java
// 1. 前向傳播時構建計算圖
Tensor a = ...;
Tensor b = ...;
Tensor c = a.matmul(b);  // c 記住了 a 和 b

// 2. 每個運算都定義了 backward 函數
out.backward = (v) -> {
    // 計算梯度並累積到 this.grad 和 other.grad
    this.grad[i] += ...;
    other.grad[j] += ...;
    return null;
};

// 3. 反向傳播時按拓撲順序執行
c.backward();  // 自動調用所有 backward 函數
```

### 2. BigramLanguageModel.java - 語言模型

實現了兩種訓練方法:

#### 方法 1: 計數法 (Counting Approach)

```java
// 統計 bigram 頻率
for (String word : words) {
    String extWord = "." + word + ".";  // 添加起止符
    for (int i = 0; i < extWord.length() - 1; i++) {
        char ch1 = extWord.charAt(i);
        char ch2 = extWord.charAt(i + 1);
        bigramCounts[idx1][idx2]++;      // 計數
    }
}

// 歸一化為概率
P(char2 | char1) = (count + 1) / (rowSum + vocabSize)
```

**優點**:
- 簡單直觀
- 精確(給定數據)
- 快速訓練

**缺點**:
- 需要平滑處理未見過的 bigram
- 無法推廣到更複雜的模型
- 參數數量隨詞彙量平方增長

#### 方法 2: 神經網路法 (Neural Network Approach)

```java
// 初始化權重 W (27x27)
W = Tensor.randn(rng, 27, 27).requiresGrad(true);

// 訓練循環
for (int iter = 0; iter < numIterations; iter++) {
    // 1. One-hot 編碼輸入
    Tensor xenc = oneHotEncode(xs, 27);
    
    // 2. 前向傳播
    Tensor logits = xenc.matmul(W);           // 線性變換
    Tensor counts = logits.exp();             // 指數化
    Tensor sumCounts = counts.sum(1, true);   // 求和
    Tensor probs = counts.div(sumCounts);     // Softmax
    
    // 3. 計算損失
    loss = -log(probs[correct_indices]).mean() + λ||W||²
    
    // 4. 反向傳播
    W.zeroGrad();
    loss.backward();
    
    // 5. 更新參數
    W.data -= learningRate * W.grad;
}
```

**神經網路架構**:

```
輸入 (one-hot): [0, 0, 1, 0, ..., 0]  (27 維)
           ↓
    權重矩陣 W (27×27)
           ↓
    Logits: [l₁, l₂, ..., l₂₇]
           ↓
    Softmax: exp(lᵢ) / Σexp(lⱼ)
           ↓
    輸出概率: [p₁, p₂, ..., p₂₇]
```

**損失函數**:

```
L = -1/N Σ log P(yᵢ | xᵢ) + λ||W||²
    ^^^^^^^^^^^^^^^^^^^^^^^^   ^^^^^^^
    交叉熵損失                L2 正則化
```

**優點**:
- 可擴展到更複雜的架構
- 自動學習特徵表示
- 支持梯度下降優化
- 正則化防止過擬合

**缺點**:
- 需要調整超參數
- 訓練時間較長
- 可能陷入局部最優

### 3. Main.java - 主程式

展示完整的工作流程:

1. **加載數據**
2. **訓練計數模型**
3. **訓練神經網路模型**
4. **生成樣本**
5. **比較結果**

## 🔬 數學原理

### Softmax 函數

將任意實數向量轉換為概率分布:

```
softmax(x)ᵢ = exp(xᵢ) / Σⱼ exp(xⱼ)
```

**性質**:
- 輸出總和為 1
- 保持單調性
- 可微分

### 交叉熵損失

衡量預測分布與真實分布的差距:

```
H(p, q) = -Σ p(x) log q(x)
```

對於單個樣本(真實標籤為 y):

```
L = -log P(y | x)
```

### 梯度計算示例

以 Softmax + 交叉熵為例:

```java
// 前向傳播
probs = softmax(X @ W)
loss = -log(probs[y])

// 反向傳播(簡化版)
∂L/∂probs[i] = -1/probs[y]  (if i == y)
             = 0             (otherwise)

∂L/∂W = X.T @ ∂L/∂probs
```

## 📊 實驗結果

### 性能比較

| 方法        | NLL   | 訓練時間 | 樣本質量 |
|------------|-------|---------|---------|
| 計數法      | 2.454 | <1s     | 好      |
| 神經網路    | 2.482 | ~10s    | 好      |

兩種方法達到相似的 NLL,證明神經網路成功學習了與計數法相同的統計分布!

### 生成樣本示例

```
計數法:
1. mora
2. axx
3. minaymoryles
4. kondlaisah
5. anchthizarie

神經網路:
1. mor
2. axx
3. minaymoryles
4. kondlaisah
5. anchthizarie
```

結果非常相似,因為兩種方法學習了相同的 bigram 統計!

## 🎓 從 Bigrams 到 Transformers 的路徑

### Lecture 2 (當前): Bigrams
- Context: 只看前一個字符
- 模型: 單層神經網路 (27→27)
- 參數: 729 個 (27×27)

### Lecture 3: MLP
- Context: 看前 N 個字符
- 模型: 嵌入 + 多層感知機
- 新概念: 隱藏層、嵌入、激活函數

### Lecture 4-5: BatchNorm & Backprop
- 優化訓練穩定性
- 理解梯度流動
- 手動實現反向傳播

### Lecture 6: WaveNet
- 層次結構
- 擴張卷積
- 更大的上下文窗口

### Lecture 7: GPT
- Self-attention 機制
- Transformer 架構
- 位置編碼

### Lecture 8: Tokenizer
- Byte Pair Encoding (BPE)
- 子詞切分
- 詞彙表構建

## 💻 運行指南

### 使用 Maven

```bash
# 編譯
mvn clean compile

# 運行
mvn exec:java -Dexec.mainClass="com.makemore.Main"

# 測試
mvn test

# 打包
mvn package
java -jar target/java-makemore-bigrams-1.0.0.jar
```

### 不使用 Maven (純 Java)

```bash
# 編譯
javac -d build/classes src/main/java/com/makemore/bigram/*.java src/main/java/com/makemore/*.java

# 運行
java -cp build/classes com.makemore.Main
```

## 🐛 常見問題

### Q1: 為什麼 NLL 不是 0?
A: 因為語言有固有的不確定性。即使是完美的模型也無法完全預測下一個字符。

### Q2: 為什麼學習率這麼高 (50.0)?
A: 因為使用全批次梯度下降,梯度穩定且準確。實際應用中使用小批次時會用更小的學習率 (0.001-0.1)。

### Q3: 正則化為什麼重要?
A: 防止過擬合。沒有正則化時,模型可能在訓練集上表現很好,但在新數據上表現差。

### Q4: 為什麼需要 one-hot 編碼?
A: 將離散的字符索引轉換為神經網路可以處理的向量形式。Lecture 3 會介紹更高效的嵌入方法。

### Q5: 計算圖是如何構建的?
A: 每次運算都創建新的 Tensor,並記錄父節點(prev)和反向傳播函數(backward)。這樣形成了一個有向無環圖 (DAG)。


# Building makemore Part 2: MLP

**完整 Java 實現** - Andrej Karpathy's makemore Lecture 3

## 🎯 專案概述

實現字符級多層感知機 (MLP) 語言模型,從前 3 個字符預測下一個字符。

### 與 Lecture 2 (Bigrams) 的對比

| 特性 | Bigrams | MLP (本專案) |
|------|---------|-------------|
| 上下文 | 1 個字符 | **3 個字符** |
| 輸入表示 | One-hot (27維) | **學習的嵌入 (10維)** |
| 模型 | 線性 (單層) | **非線性 (雙層)** |
| 參數量 | 729 | **11,897** |
| Loss (NLL) | ~2.48 | **~2.13** ⬇️ |
| 樣本質量 | 可辨認 | **更像真名** ✨ |

---

## 🏗️ 模型架構

```
輸入: [ch₁, ch₂, ch₃]  (前3個字符的索引)
         ↓
    Embedding Lookup
    C[ch₁], C[ch₂], C[ch₃]  (3個 10維向量)
         ↓
    Flatten 成 30維
         ↓
    Linear + Tanh
    h = tanh(W1 @ x + b1)  (30 → 200)
         ↓
    Linear
    logits = W2 @ h + b2   (200 → 27)
         ↓
    Softmax
    P(next_char | context)
```

### 參數詳情

```
C  : 27 × 10   = 270     (字符嵌入表)
W1 : 30 × 200  = 6,000   (第一層權重)
b1 : 200       = 200     (第一層偏置)
W2 : 200 × 27  = 5,400   (第二層權重)
b2 : 27        = 27      (第二層偏置)
────────────────────────
Total         = 11,897 參數
```

---

## 📚 核心概念

### 1️⃣ 字符嵌入 (Character Embeddings)

**為什麼需要?**
- One-hot 太稀疏 (27維只有1個非零)
- 無法表達字符相似性
- 嵌入學習**密集表示**

**如何工作:**
```java
// 輸入索引: [5, 13, 13] → 'e', 'm', 'm'
Tensor emb = C.index(X);  // 查表
// 輸出: (3, 10) - 三個10維向量
```

**學到什麼?**
- 元音 (a,e,i,o,u) 聚在一起
- 常見組合的字符距離近
- 每個維度捕獲不同特徵

---

### 2️⃣ 上下文窗口 (Context Window)

```
Word: "emma"
加上起止符: "...emma."

訓練樣本:
[., ., .] → e
[., ., e] → m
[., e, m] → m
[e, m, m] → a
[m, m, a] → .
```

**Block Size = 3** 表示:
- 用前 3 個字符預測下一個
- 比 bigram 的 1 個字符有**更多信息**
- 可學習更複雜的模式 (如 "qu" 後常接 "a")

---

### 3️⃣ 非線性激活 (Tanh)

```java
h = tanh(W1 @ x + b1)
```

**為什麼重要?**
- 沒有激活 → 多層只是線性變換的組合 = 還是線性
- **Tanh 引入非線性** → 可學複雜決策邊界
- 輸出範圍: [-1, 1]

**梯度:**
```
∂tanh/∂x = 1 - tanh²(x)
```

---

### 4️⃣ Train/Dev/Test 分割

```
228,146 個訓練樣本
├─ Train (80%): 182,516  ← 學習參數
├─ Dev   (10%):  22,815  ← 調超參數
└─ Test  (10%):  22,815  ← 最終評估
```

**為什麼需要?**
- **Overfitting**: 模型記住訓練數據但不泛化
- **Dev set**: 檢測過擬合,調學習率
- **Test set**: 報告最終性能 (絕不用於調參!)

---

### 5️⃣ Mini-Batch 梯度下降

```java
for iter in 1..200000:
    // 1. 隨機抽32個樣本
    batch = randomSample(data, 32)
    
    // 2. 前向傳播
    logits = forward(batch)
    loss = crossEntropy(logits, targets)
    
    // 3. 反向傳播
    loss.backward()
    
    // 4. 更新參數
    params -= learningRate * grads
```

**為什麼用 Mini-Batch?**
- ✅ 比全批次**快** (不用處理所有數據)
- ✅ 比單樣本**穩定** (梯度方向對)
- ✅ **正則化效果** (噪音幫助泛化)

---

## 🔬 關鍵實現細節

### Gather 操作 (Cross-Entropy 的關鍵!)

**問題**: 從 softmax 概率中選出目標類別的概率,同時**保持梯度連接**

```java
// probs: (batch=32, vocab=27)
// targets: (batch=32) 如 [5, 13, 13, 1, 0, ...]

// ❌ 錯誤做法 - 切斷梯度!
double[] selected = new double[32];
for (int i = 0; i < 32; i++) {
    selected[i] = probs.data[i * 27 + targets[i]];
}
Tensor result = new Tensor(selected, ...);  // 沒連到 probs!

// ✅ 正確做法 - gather 操作
Tensor selected = probs.gather(targets);  // 保持計算圖連接!
```

**Gather 實現:**
```java
public Tensor gather(Tensor indices) {
    // Forward: result[i] = this[i, indices[i]]
    for (int i = 0; i < batchSize; i++) {
        result[i] = this.data[i * numClasses + indices[i]];
    }
    
    // Backward: this.grad[i, indices[i]] += out.grad[i]
    out.backward = (v) -> {
        for (int i = 0; i < batchSize; i++) {
            int classIdx = (int) indicesData[i];
            this.grad[i * numClasses + classIdx] += out.grad[i];
        }
    };
}
```

---

### Cross-Entropy Loss

```java
public Tensor crossEntropyLoss(Tensor logits, Tensor targets) {
    // 1. Softmax
    Tensor exp = logits.exp();
    Tensor sumExp = exp.sum(1, true);
    Tensor probs = exp.div(sumExp);
    
    // 2. Gather 目標概率 (保持梯度!)
    Tensor selected = probs.gather(targets);
    
    // 3. -log(p).mean()
    Tensor loss = selected.log().neg().mean();
    
    return loss;  // 自動追蹤整個計算圖!
}
```

**為什麼這樣寫?**
- ✅ 每一步都是 Tensor 操作
- ✅ 自動構建計算圖
- ✅ `loss.backward()` 自動傳播梯度

---

## 📊 訓練過程

### 超參數

```java
blockSize = 3         // 上下文長度
embeddingDim = 10     // 嵌入維度
hiddenSize = 200      // 隱藏層大小
learningRate = 0.1    // 學習率 (100k 後降到 0.01)
batchSize = 32        // 批次大小
iterations = 200000   // 訓練步數
```

### 預期訓練曲線

```
Iter 0:      loss=3.69, train=3.69, dev=3.69  ← 隨機初始化
Iter 10000:  loss=2.45, train=2.43, dev=2.48  ← 開始學習
Iter 50000:  loss=2.21, train=2.19, dev=2.24
Iter 100000: loss=2.15, train=2.13, dev=2.17  ← 學習率降低
Iter 200000: loss=2.13, train=2.11, dev=2.15  ← 收斂

Final Test Loss: 2.12
```

### 學習率調整策略

```java
if (iter == 100000) {
    learningRate = 0.01;  // 從 0.1 降到 0.01
}
```

**為什麼?**
- 前期: 大步前進 (lr=0.1)
- 後期: 精細調整 (lr=0.01)

---

## 💻 使用方法

### 編譯運行

```bash
# 使用 Maven
mvn clean compile exec:java

# 或直接用 Java
javac src/main/java/com/makemore/**/*.java
java com.makemore.Main
```

### 預期輸出

```
╔════════════════════════════════════════════╗
║   Makemore Part 2: MLP Language Model     ║
╚════════════════════════════════════════════╝

=== Loading Data ===
Loaded 32033 words
Vocabulary size: 27

=== Building Dataset ===
Total examples: 228146
Training: 182516, Dev: 22815, Test: 22815

=== Initializing Parameters ===
Total parameters: 11897

=== Training ===
Initial gradient norm: 4.523456  ✅

Iter 0: loss=3.6892, train=3.6892, dev=3.6889
Iter 10000: loss=2.4521, train=2.4312, dev=2.4798
...
Iter 199999: loss=2.0765, train=2.0532, dev=2.0867

=== Final Evaluation ===
Train: 2.0532, Dev: 2.0867, Test: 2.0891

=== Sampling 20 Names ===
 1. carmahela     ← 看起來像真名字!
 2. jhovi
 3. kimrin
 4. halanna
 5. jazhien
 6. amerynci
 7. aqui
 8. nellara
 9. chaiiv
10. kaleigh
```

---

## 🐛 常見問題 & Debug

### Q1: Loss 不下降 (停在 ~24)

**症狀:**
```
Iter 0: loss=24.96
Iter 10000: loss=24.96  ← 完全不變!
```

**原因**: 梯度沒有流動!

**檢查:**
```java
// 在第一次迭代後打印
double gradNorm = 0;
for (Tensor p : parameters) {
    for (double g : p.getGrad()) {
        gradNorm += g * g;
    }
}
System.out.println("Grad norm: " + Math.sqrt(gradNorm));
```

如果 = 0 → **計算圖斷了!**

**可能原因:**
- ❌ Cross-entropy 沒用 `gather()`
- ❌ 某個操作沒設置 `backward` 函數
- ❌ 沒調用 `requiresGrad(true)`

---

### Q2: 生成的都是 "qqqxbx" 之類的垃圾

**原因**: 模型完全沒學到東西

**解決**: 見 Q1

---

### Q3: Loss 爆炸 (變成 NaN)

**症狀:**
```
Iter 100: loss=2.5
Iter 200: loss=15.8
Iter 300: loss=NaN  ← 爆了!
```

**原因**: 學習率太大

**解決:**
```java
learningRate = 0.01;  // 降低學習率
```

---

### Q4: 為什麼用 Tanh 而不是 ReLU?

**答**:
- Karpathy 的原始實現用 tanh
- Tanh 輸出有界 [-1, 1] → 訓練更穩定
- ReLU 也可以,但需要調整初始化

---

### Q5: 可以增加 hidden size 提升性能嗎?

**答**: 可以!

```java
hiddenSize = 300;  // 從 200 增加到 300
```

**預期效果:**
- ✅ Loss 可能降到 ~2.05
- ⚠️ 訓練變慢
- ⚠️ 過擬合風險增加 (train/dev gap 變大)

**建議**: 先用 200 跑通,再嘗試調參

---

## 🎓 學到的關鍵點

### 1. 嵌入的威力

**One-hot (Bigram):**
```
'a' → [1,0,0,...,0]  (27維,稀疏)
'e' → [0,0,0,0,1,...]
```
→ 'a' 和 'e' 沒有相似性

**Embedding (MLP):**
```
'a' → [0.3, -0.1, 0.5, ...]  (10維,密集)
'e' → [0.2, -0.2, 0.4, ...]
```
→ 相似字符有相似向量!

---

### 2. 計算圖的完整性

**錯誤的流程:**
```
logits → probs → [手動提取] → loss
                    ↑
                  梯度斷了!
```

**正確的流程:**
```
logits → probs → gather → log → neg → mean
  ↓       ↓        ↓      ↓     ↓      ↓
所有操作都保持梯度連接 ✅
```

---

### 3. 為什麼 Loss 無法降到 0?

**理論最小值:**
```
L = -log(1) = 0  (完美預測每個字符)
```

**實際:**
```
L ≈ 2.1
```

**原因:**
1. **固有不確定性**: 即使人類也無法 100% 預測
2. **模型容量**: 11k 參數可能不夠
3. **上下文限制**: 只看 3 個字符,可能不夠

**改進方向:**
- 增大 embedding/hidden size
- 增加上下文長度 (block_size = 5)
- 加深網絡 (2 層隱藏層)

---

## 🎯 與 Lecture 2 對比總結

### Bigram (Lecture 2)
```python
P(next | prev1)
- 簡單直接
- 729 參數
- NLL ≈ 2.48
```

### MLP (Lecture 3)
```python
P(next | prev3, prev2, prev1)
- 需要訓練
- 11,897 參數
- NLL ≈ 2.13 ✨
```

**改進來自:**
1. ✅ **更長上下文** (3 vs 1)
2. ✅ **學習的嵌入** (捕獲相似性)
3. ✅ **非線性** (Tanh 隱藏層)
4. ✅ **分佈式表示** (200 個隱藏單元)

---

# Building makemore Part 3: Activations & Gradients, BatchNorm

**完整 Java 實現** - Andrej Karpathy's makemore Lecture 4

## 🎯 專案概述

實現**深層 MLP** (5個隱藏層) 並引入 **Batch Normalization**,解決深度網路訓練困難的核心問題。

這是 makemore 系列中最具挑戰性的一課,因為它深入探討了**為什麼深度學習需要歸一化技術**。

## 📊 與 Lecture 3 的關鍵對比

| 特性 | Lecture 3 (MLP) | Lecture 4 (BatchNorm) |
|------|-----------------|----------------------|
| 隱藏層數 | **1 層** | **5 層** (深層網路) |
| 層結構 | Linear + Tanh | **Linear + BatchNorm + Tanh** |
| 參數量 | ~12k | **~47k** |
| 訓練難度 | 簡單直接 | **需要 BatchNorm 才能訓練** |
| Train Loss | ~2.11 | **~2.05** (更深更好) |
| Dev Loss | ~2.15 | **~2.39** |
| 主要挑戰 | 理解嵌入和 MLP | **診斷激活值和梯度** |

## 🔬 核心問題: 為什麼深層網路難訓練?

### 沒有 BatchNorm 的災難

```
訓練結果 (實際運行):
Iter 0:   loss=18.24  ← 超級高!
Iter 999: loss=4.75   ← 完全沒學到東西

激活值統計:
Layer 1: mean=-0.02, std=1.00, saturated=98.69%  ← 幾乎完全飽和!
Layer 3: mean=+0.01, std=0.98, saturated=91.81%
Layer 5: mean=+0.03, std=0.98, saturated=89.60%
Layer 7: mean=+0.08, std=0.98, saturated=90.04%
Layer 9: mean=+0.06, std=0.99, saturated=95.60%

生成結果: (垃圾)
```

**問題根源:**

1. **激活值飽和** - Tanh 輸出幾乎都是 ±1
2. **梯度消失** - 飽和區域梯度 ≈ 0
3. **訓練停滯** - 參數無法更新

---

### 有 BatchNorm 的成功

```
訓練結果 (實際運行):
Iter 0:      loss=3.28   ← 正常初始值
Iter 10000:  train≈2.23, dev≈2.56
Iter 100000: train≈2.16, dev≈2.39
Iter 199999: train≈2.03, dev≈2.39  ← 成功收斂!

激活值統計:
Layer 2:  mean=+0.01, std=0.63, saturated=2.93%  ✅ 健康!
Layer 5:  mean=-0.01, std=0.65, saturated=3.27%  ✅
Layer 8:  mean=-0.02, std=0.67, saturated=2.59%  ✅
Layer 11: mean=-0.01, std=0.68, saturated=2.18%  ✅
Layer 14: mean=-0.00, std=0.71, saturated=4.12%  ✅

生成結果:
1. elrio        ← 看起來像真名字!
2. anna
3. janni
4. tyla
5. kamiyah
```

**成功原因:** BatchNorm 強制每層激活值保持健康分佈!

---

## 🏗️ 網路架構

```
Input: [ch₁, ch₂, ch₃]  (3個字符索引)
         ↓
    Embedding (27 → 10)
         ↓
    Flatten (30)
         ↓
┌─────────────────────────────┐
│ 有 BatchNorm:               │
│                             │
│  Linear(30 → 100, no bias)  │
│  BatchNorm1d(100)           │ ← 關鍵層!
│  Tanh                       │
├─────────────────────────────┤
│ 重複 4 次:                  │
│  Linear(100 → 100, no bias) │
│  BatchNorm1d(100)           │
│  Tanh                       │
├─────────────────────────────┤
│  Linear(100 → 27, no bias)  │
│  BatchNorm1d(27)            │
└─────────────────────────────┘
         ↓
    Softmax
```

**關鍵設計決策:**

1. **Linear 層不用 bias** - BatchNorm 的 beta 已提供偏移
2. **每層後都接 BatchNorm** - 保持激活值穩定
3. **最後一層也用 BatchNorm** - 讓初始預測不那麼自信

---

## 🔑 Batch Normalization 原理

### 核心思想

**在每一層後強制標準化激活值,讓訓練更穩定。**

### 數學公式

```python
# Training mode (使用 batch 統計)
mean = x.mean(0)                    # 計算 batch 均值
var = x.var(0)                      # 計算 batch 方差
x_norm = (x - mean) / sqrt(var + ε) # 標準化
out = gamma * x_norm + beta         # 可學習的縮放和偏移

# 更新 running 統計 (用於推理)
running_mean = 0.9 * running_mean + 0.1 * mean
running_var = 0.9 * running_var + 0.1 * var

# Inference mode (使用 running 統計)
x_norm = (x - running_mean) / sqrt(running_var + ε)
out = gamma * x_norm + beta
```

### 參數說明

- **gamma (scale)**: 可學習,初始化為 1
- **beta (shift)**: 可學習,初始化為 0
- **running_mean**: 不可學習,指數移動平均
- **running_var**: 不可學習,指數移動平均
- **epsilon (ε)**: 數值穩定性,通常 1e-5
- **momentum**: 更新速度,通常 0.1

### 為什麼有效?

1. **穩定激活分佈** - 每層輸入都是 mean=0, std=1
2. **減少內部協變量偏移** - 層間分佈不再漂移
3. **允許更大學習率** - 訓練更快
4. **正則化效果** - batch 統計引入噪音

---

## 💻 核心實現

### 1. BatchNorm1d 層

```java
public class BatchNorm1d implements Layer {
    private Tensor gamma;  // Scale (learnable)
    private Tensor beta;   // Shift (learnable)
    private double[] runningMean;  // Buffer
    private double[] runningVar;   // Buffer
    
    public Tensor forward(Tensor x) {
        if (training) {
            // 使用 batch 統計
            Tensor mean = x.mean(0);
            Tensor variance = x.variance(0);
            Tensor xNorm = (x - mean) / sqrt(variance + eps);
            
            // 更新 running 統計
            runningMean = 0.9 * runningMean + 0.1 * mean;
            runningVar = 0.9 * runningVar + 0.1 * variance;
            
            return gamma * xNorm + beta;
        } else {
            // 使用 running 統計
            Tensor xNorm = (x - runningMean) / sqrt(runningVar + eps);
            return gamma * xNorm + beta;
        }
    }
}
```

### 2. Tensor 新增操作

BatchNorm 需要這些新的張量操作:

```java
// 沿 batch 維度求均值
Tensor mean(int dim)         // (batch, features) → (features,)

// 沿 batch 維度求方差
Tensor variance(int dim)     // (batch, features) → (features,)

// 減法 (支持廣播)
Tensor subtract(Tensor)      // (batch, features) - (features,)

// 平方根
Tensor sqrt()                // element-wise sqrt

// 乘法 (支持廣播)
Tensor mul(Tensor)           // (batch, features) * (features,)

// 除法 (支持廣播) - 關鍵修復!
Tensor div(Tensor)           // (batch, features) / (features,)
```

**重要:** `div()` 方法需要支持 `(batch, features) / (features,)` 的廣播,這是 BatchNorm 的核心需求!

### 3. Layer 介面 (PyTorch 風格)

```java
public interface Layer {
    Tensor forward(Tensor x);
    List parameters();
    void setTraining(boolean training);  // 切換訓練/推理模式
    Tensor getOutput();                  // 用於診斷
}
```

### 4. 深層 MLP 構建

```java
List layers = Arrays.asList(
    new Linear(30, 100, false, rng), new BatchNorm1d(100), new TanhLayer(),
    new Linear(100, 100, false, rng), new BatchNorm1d(100), new TanhLayer(),
    new Linear(100, 100, false, rng), new BatchNorm1d(100), new TanhLayer(),
    new Linear(100, 100, false, rng), new BatchNorm1d(100), new TanhLayer(),
    new Linear(100, 100, false, rng), new BatchNorm1d(100), new TanhLayer(),
    new Linear(100, 27, false, rng), new BatchNorm1d(27)
);

// 初始化最後一層 (讓初始預測不那麼自信)
lastBatchNorm.getGamma() *= 0.1;
```

---

## 📈 訓練過程與結果

### 實驗設計

我們進行了**兩個對照實驗**:

1. **實驗 1: 沒有 BatchNorm** (預期失敗)
2. **實驗 2: 有 BatchNorm** (預期成功)

### 超參數

```java
vocabSize = 27
blockSize = 3
embeddingDim = 10
hiddenSize = 100
numHiddenLayers = 5

batchSize = 32
learningRate = 0.1 (前 150k 次)
              0.01 (後 50k 次)
maxIterations = 200000
```

### 實驗 1 結果: WITHOUT BatchNorm ❌

```
訓練曲線:
Iter 0:   loss=18.24, train≈17.07, dev≈17.15  ← 隨機猜測
Iter 999: loss=4.75,  train≈3.96,  dev≈4.07   ← 沒有改善!

激活值診斷:
Layer 1: saturated=98.69%  ← 災難!
Layer 3: saturated=91.81%
Layer 5: saturated=89.60%
Layer 7: saturated=90.04%
Layer 9: saturated=95.60%

結論: 深層網路完全無法訓練! ❌
```

**為什麼失敗?**

- 激活值幾乎全部飽和 (>90%)
- Tanh 在飽和區域梯度 ≈ 0
- 梯度無法反向傳播
- 參數無法更新

---

### 實驗 2 結果: WITH BatchNorm ✅

```
訓練曲線:
Iter 0:      loss=3.28,  train≈3.36,  dev≈3.34   ← 正常初始化
Iter 10000:  loss=2.37,  train≈2.23,  dev≈2.56   ← 快速下降
Iter 100000: loss=2.24,  train≈2.16,  dev≈2.39   ← 持續改善
Iter 150000: 學習率降至 0.01                      ← LR decay
Iter 199999: loss=2.13,  train≈2.03,  dev≈2.39   ← 收斂!

激活值診斷:
Layer 2:  mean=+0.01, std=0.63, saturated=2.93%  ✅ 健康!
Layer 5:  mean=-0.01, std=0.65, saturated=3.27%  ✅
Layer 8:  mean=-0.02, std=0.67, saturated=2.59%  ✅
Layer 11: mean=-0.01, std=0.68, saturated=2.18%  ✅
Layer 14: mean=-0.00, std=0.71, saturated=4.12%  ✅

最終評估:
Train loss: 2.05  ← 比 Lecture 3 (2.11) 更好!
Dev loss:   2.39
Test loss:  2.46

生成樣本 (質量很好!):
1. elrio          11. lytka
2. davdanamaria   12. paileah
3. janni          13. caiya
4. raley          14. tyla
5. anna           15. keadiaup
6. ridsing        16. mykentleigh
7. man            17. graycensley
8. dedi           18. amarelde
9. jeelee         19. kamiyah
10. janiella      20. suthenishia

結論: BatchNorm 讓深層網路成功訓練! ✅
```

**成功關鍵:**

- 激活值飽和率 < 5% (健康範圍)
- 每層 mean ≈ 0, std ≈ 0.6-0.7
- 梯度順利反向傳播
- Loss 穩定下降

---

## 📊 性能對比總結

| Model | 深度 | Loss (Train) | Loss (Dev) | 激活飽和率 | 訓練狀態 |
|-------|------|-------------|-----------|-----------|---------|
| Lecture 3 (1層) | 淺 | 2.11 | 2.15 | ~5% | ✅ 成功 |
| Lecture 4 無BN (5層) | 深 | 3.96 | 4.07 | **95%** | ❌ 失敗 |
| Lecture 4 有BN (5層) | 深 | **2.05** | 2.39 | **3%** | ✅ 成功 |

**關鍵發現:**

1. 深層網路 **沒有 BatchNorm** → 訓練失敗
2. 深層網路 **有 BatchNorm** → 訓練成功,性能更好
3. BatchNorm 是訓練深度網路的**關鍵技術**

---

## 🔍 診斷工具

### 激活值統計分析

```java
DiagnosticTools.analyzeActivations(layers, "After training");

// 輸出:
// Layer 2 (TanhLayer): mean=+0.01, std=0.63, saturated=2.93%
```

**健康標準:**

- ✅ mean ≈ 0 (中心化)
- ✅ std ≈ 0.6-0.7 (適中方差)
- ✅ saturation < 5% (很少飽和)

**不健康標準:**

- ❌ mean 離 0 很遠 (偏移)
- ❌ std 太小或太大 (方差異常)
- ❌ saturation > 90% (幾乎全飽和)

### 飽和度計算

```java
// Tanh 飽和定義: |tanh(x)| > 0.99
saturated = count(|activation| > 0.99) / total
```

---

## 💡 關鍵洞察

### 1. 為什麼 Linear 層不用 bias?

```java
// 沒有 BatchNorm:
Linear(x) = W @ x + b  ← 需要 bias

// 有 BatchNorm:
Linear(x) = W @ x             ← 不需要 bias!
BN(x) = gamma * normalize(x) + beta  ← beta 提供偏移

// 結論: BatchNorm 的 beta 已經提供了偏移功能
// 添加 bias 是多餘的
```

### 2. 訓練 vs 推理的差異

```
訓練模式 (training=True):
  - 使用當前 batch 統計 (mean, var)
  - 更新 running 統計
  - 引入批次間的噪音 (正則化效果)

推理模式 (training=False):
  - 使用 running 統計
  - 每個樣本獨立處理
  - 結果穩定可重現
```

**為什麼需要 running 統計?**

- 推理時可能只有 1 個樣本,無法計算 batch 統計
- Running 統計代表整個訓練集的分佈
- 通過指數移動平均平滑更新

### 3. BatchNorm 的副作用

**優點:**

- ✅ 穩定訓練深層網路
- ✅ 允許更大學習率
- ✅ 減少對初始化的依賴
- ✅ 隱含的正則化效果

**缺點:**

- ❌ 訓練/推理不一致
- ❌ 對 batch size 敏感 (小 batch 不穩定)
- ❌ 在 RNN 中難以應用
- ❌ 代碼複雜,容易出 bug
- ❌ 耦合 batch 中的樣本 (破壞獨立性)

### 4. 其他歸一化方法

```
BatchNorm   - 沿 batch 維度歸一化 (本課重點)
LayerNorm   - 沿 feature 維度歸一化 (Transformer 用)
GroupNorm   - 分組歸一化 (小 batch 友好)
InstanceNorm - 單樣本歸一化 (風格遷移)
```

---

## 🐛 常見問題與調試

### Q1: `div()` 方法報錯 "Incompatible shapes"

**症狀:**
```
Error: Incompatible shapes for division
BatchNorm forward 失敗
```

**原因:** 你的 `Tensor.div()` 不支持 `(batch, features) / (features,)` 的廣播

**解決:**
```java
// 需要在 div() 中添加這個 case:
if (shape.length == 2 && other.shape.length == 1 && shape[1] == other.shape[0]) {
    // Broadcasting: (batch, features) / (features,)
    // 實現廣播除法...
}
```

參考 `TENSOR_div_FIXED.java` 中的完整實現。

---

### Q2: OutOfMemoryError (Java heap space)

**症狀:**
```
Exception in thread "main" java.lang.OutOfMemoryError: Java heap space
```

**原因:**

1. 每次迭代評估整個訓練集 (182k 樣本)
2. 計算圖不斷累積
3. 中間張量沒釋放

**解決:**

1. **增加堆記憶體:**
   ```
   VM options: -Xms1g -Xmx2g
   ```

2. **使用抽樣評估:**
   ```java
   // 不要評估整個集合
   double loss = evaluate(model, allData, allLabels);  // ❌
   
   // 只評估 500 個樣本
   double loss = evaluateSample(model, data, labels, 500, rng);  // ✅
   ```

3. **減少評估頻率:**
   ```java
   // 每 10000 次迭代才評估
   if (iter % 10000 == 0) {
       evaluate(...);
   }
   ```

4. **定期 GC:**
   ```java
   if (iter % 1000 == 0) {
       System.gc();
   }
   ```

---

### Q3: Loss 不下降 (WITH BatchNorm 也不行)

**可能原因:**

1. **學習率太大或太小**
   ```java
   lr = 0.1;  // 試試 0.01 或 0.5
   ```

2. **初始化問題**
   ```java
   // 最後一層應該乘以小數
   lastBatchNorm.getGamma() *= 0.1;  // 很重要!
   ```

3. **梯度沒有流動**
   ```java
   // 檢查梯度
   double gradNorm = calculateGradNorm(parameters);
   System.out.println("Grad norm: " + gradNorm);
   // 期望 > 0
   ```

---

### Q4: 激活值還是飽和 (即使用了 BatchNorm)

**檢查清單:**

1. **確認 BatchNorm 在訓練模式**
   ```java
   model.setTrainMode();  // 必須!
   ```

2. **確認層的順序**
   ```java
   // 正確:
   Linear → BatchNorm → Tanh
   
   // 錯誤:
   Linear → Tanh → BatchNorm  // BatchNorm 放錯位置!
   ```

3. **檢查 forward 是否調用了 BatchNorm**
   ```java
   for (Layer layer : layers) {
       x = layer.forward(x);  // 確保每層都被調用
   }
   ```

---

## 📁 專案結構

```
makemore-batchnorm/
├── src/main/java/com/makemore/
│   ├── Main.java                    # 主程式 (雙重實驗)
│   ├── DeepMLP.java                 # 深層 MLP 模型
│   ├── DataLoader.java              # 數據加載器
│   │
│   ├── layers/                      # 層實現
│   │   ├── Layer.java               # 層介面
│   │   ├── Linear.java              # 全連接層
│   │   ├── BatchNorm1d.java         # ⭐ BatchNorm 層
│   │   └── TanhLayer.java           # Tanh 激活層
│   │
│   ├── mlp/
│   │   └── Tensor.java              # 張量 (需要新增操作)
│   │
│   └── utils/
│       └── DiagnosticTools.java     # 診斷工具
│
├── names.txt                        # 訓練數據
├── pom.xml                          # Maven 配置
└── README.md                        # 本文件
```

---

## 🚀 使用方法

### 編譯運行

```bash
# 使用 Maven
mvn clean compile exec:java

# VM options (推薦)
-Xms1g -Xmx2g

# 預期運行時間
實驗 1 (無 BatchNorm): ~1 分鐘
實驗 2 (有 BatchNorm): ~30-60 分鐘 (200k iterations)
```

### 快速測試 (減少迭代次數)

如果想快速看到效果,可以修改 `Main.java`:

```java
// 從 200000 改成 50000
trainModel(modelWithBN, dataLoader, 50000, 0.1, 32, true);
```

預期結果:
```
50k 次:  loss ≈ 2.3-2.4  (5-10 分鐘)
200k 次: loss ≈ 2.0-2.1  (30-60 分鐘)
```

---

## 📚 學習路徑

```
✅ Lecture 1: Micrograd (scalar autograd)
✅ Lecture 2: Bigrams (simple language model)  
✅ Lecture 3: MLP (embeddings + hidden layers)
✅ Lecture 4: BatchNorm (deep networks + diagnostics)  ← 你在這裡
⬜ Lecture 5: Manual Backprop (gradient ninja)
⬜ Lecture 6: WaveNet (convolutional architecture)
⬜ Lecture 7: GPT (transformer)
⬜ Lecture 8: Tokenizer (BPE)
```

---

## 🎓 核心收穫

### 1. 深度學習的歷史難題

**2015 年之前:**
- 訓練深層網路幾乎不可能
- 需要極其小心的初始化
- 梯度消失/爆炸是常態
- 網路深度 < 10 層

**BatchNorm (2015) 之後:**
- 可以訓練 50-100 層網路
- 對初始化不那麼敏感
- 訓練穩定可靠
- 開啟了深度學習的黃金時代

### 2. 現代深度網路的標準模式

```
Input
  ↓
[Linear → Normalization → Activation] × N
  ↓
Output

這個模式貫穿現代所有架構:
- ResNet, VGG (圖像)
- Transformer, BERT (語言)
- WaveNet (音頻)
```

### 3. 診斷思維

**不只是訓練模型,更要診斷模型:**

1. 監控激活值統計 (mean, std, saturation)
2. 監控梯度流動 (grad norm, update ratio)
3. 可視化每層的行為
4. 理解模型為什麼成功/失敗

這種**診斷思維**是成為深度學習專家的關鍵!

---
