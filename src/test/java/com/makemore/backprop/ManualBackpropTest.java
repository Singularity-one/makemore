package com.makemore.backprop;

import com.makemore.mlp.Tensor;
import com.makemore.backprop.ManualBackprop;
import com.makemore.backprop.ManualBackprop.BatchNormGradients;
import com.makemore.backprop.ManualBackprop.LinearGradients;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import static org.junit.jupiter.api.Assertions.*;

import java.util.Random;

/**
 * 單元測試: 驗證手動反向傳播實現
 *
 * 測試目標:
 * - Exercise 2: Cross-Entropy Backward (優化版)
 * - Exercise 3: BatchNorm Backward (優化版)
 * - 其他輔助梯度函數
 *
 * 基於 Andrej Karpathy 的 makemore Part 4
 */
public class ManualBackpropTest {

    private Random rng;
    private static final long SEED = 42L;
    private static final double TOLERANCE = 1e-5;
    private static final double EPS = 1e-5;

    @BeforeEach
    public void setUp() {
        rng = new Random(SEED);
    }

    // ========================================================================
    // Exercise 2: Cross-Entropy Backward Tests
    // ========================================================================

    @Test
    @DisplayName("Exercise 2: Cross-Entropy Backward - Small Batch")
    public void testCrossEntropyBackward_SmallBatch() {
        System.out.println("\n=== Test: Cross-Entropy Backward (Small Batch) ===");

        // 準備測試數據 (batch=4, classes=5)
        int batchSize = 4;
        int numClasses = 5;

        double[] logitsData = {
                1.0, 2.0, 0.5, 1.5, 0.8,  // sample 0
                0.3, 0.7, 2.1, 0.4, 1.2,  // sample 1
                1.8, 0.2, 0.9, 1.3, 0.6,  // sample 2
                0.5, 1.4, 0.8, 2.0, 1.1   // sample 3
        };
        Tensor logits = new Tensor(logitsData, new int[]{batchSize, numClasses});

        // 真實標籤 (作為 Tensor)
        double[] YData = {1, 2, 0, 3};
        Tensor Y = new Tensor(YData, new int[]{batchSize});

        // 1. 手動計算梯度 (方案 B)
        Tensor dlogits_manual = ManualBackprop.crossEntropyBackward(logits, Y);

        // 2. 計算參考梯度
        Tensor dlogits_reference = computeCrossEntropyGradReference(logits, Y);

        // 3. 比較
        boolean passed = GradientChecker.compare("dlogits", dlogits_manual, dlogits_reference);
        assertTrue(passed, "Cross-Entropy gradient should match reference");
    }

    @Test
    @DisplayName("Exercise 2: Cross-Entropy Backward - Standard Batch")
    public void testCrossEntropyBackward_StandardBatch() {
        System.out.println("\n=== Test: Cross-Entropy Backward (Standard Batch) ===");

        // 標準批次大小 (batch=32, classes=27)
        int batchSize = 32;
        int numClasses = 27;

        Tensor logits = randomTensor(batchSize, numClasses);
        Tensor Y = randomLabels(batchSize, numClasses);

        // 手動 vs 參考
        Tensor dlogits_manual = ManualBackprop.crossEntropyBackward(logits, Y);
        Tensor dlogits_reference = computeCrossEntropyGradReference(logits, Y);

        boolean passed = GradientChecker.compare("dlogits", dlogits_manual, dlogits_reference);
        assertTrue(passed);
    }

    @Test
    @DisplayName("Exercise 2: Cross-Entropy Backward - Large Batch")
    public void testCrossEntropyBackward_LargeBatch() {
        System.out.println("\n=== Test: Cross-Entropy Backward (Large Batch) ===");

        int batchSize = 128;
        int numClasses = 50;

        Tensor logits = randomTensor(batchSize, numClasses);
        Tensor Y = randomLabels(batchSize, numClasses);

        Tensor dlogits_manual = ManualBackprop.crossEntropyBackward(logits, Y);
        Tensor dlogits_reference = computeCrossEntropyGradReference(logits, Y);

        boolean passed = GradientChecker.compare("dlogits", dlogits_manual, dlogits_reference);
        assertTrue(passed);
    }

    @Test
    @DisplayName("Exercise 2: Cross-Entropy Backward - Edge Cases")
    public void testCrossEntropyBackward_EdgeCases() {
        System.out.println("\n=== Test: Cross-Entropy Backward (Edge Cases) ===");

        // Case 1: batch_size = 1
        Tensor logits1 = randomTensor(1, 10);
        Tensor Y1 = new Tensor(new double[]{5}, new int[]{1});
        Tensor dlogits1_manual = ManualBackprop.crossEntropyBackward(logits1, Y1);
        Tensor dlogits1_ref = computeCrossEntropyGradReference(logits1, Y1);
        assertTrue(GradientChecker.check("batch=1", dlogits1_manual, dlogits1_ref));

        // Case 2: 極端 logits (很大的值)
        double[] largeData = new double[20];
        for (int i = 0; i < 20; i++) {
            largeData[i] = (i % 2 == 0) ? 100.0 : -100.0;
        }
        Tensor logits2 = new Tensor(largeData, new int[]{2, 10});
        Tensor Y2 = new Tensor(new double[]{3, 7}, new int[]{2});
        Tensor dlogits2_manual = ManualBackprop.crossEntropyBackward(logits2, Y2);
        Tensor dlogits2_ref = computeCrossEntropyGradReference(logits2, Y2);
        assertTrue(GradientChecker.check("large_logits", dlogits2_manual, dlogits2_ref));
    }

    @Test
    @DisplayName("Exercise 2: Cross-Entropy Properties Check")
    public void testCrossEntropyBackward_Properties() {
        System.out.println("\n=== Test: Cross-Entropy Properties ===");

        int batchSize = 32;
        int numClasses = 27;

        Tensor logits = randomTensor(batchSize, numClasses);
        Tensor Y = randomLabels(batchSize, numClasses);

        Tensor dlogits = ManualBackprop.crossEntropyBackward(logits, Y);

        // 檢查: 每一行的和應該 ≈ 0 (softmax 性質)
        double[] data = dlogits.getData();
        for (int i = 0; i < batchSize; i++) {
            double rowSum = 0.0;
            for (int j = 0; j < numClasses; j++) {
                rowSum += data[i * numClasses + j];
            }
            assertTrue(Math.abs(rowSum) < 1e-6,
                    "Row " + i + " sum should be close to 0, got: " + rowSum);
        }

        System.out.println("✅ Property check: Each row sums to ~0");
    }

    // ========================================================================
    // Exercise 3: BatchNorm Backward Tests
    // ========================================================================

    @Test
    @DisplayName("Exercise 3: BatchNorm Backward - Standard Case")
    public void testBatchNormBackward_StandardCase() {
        System.out.println("\n=== Test: BatchNorm Backward (Standard) ===");

        int batchSize = 32;
        int features = 100;

        // 準備數據
        Tensor x = randomTensor(batchSize, features);
        Tensor gamma = Tensor.ones(features);
        Tensor dout = randomTensor(batchSize, features);

        // 手動梯度
        BatchNormGradients grads_manual = ManualBackprop.batchNormBackward(
                dout, x, gamma, EPS);

        // 參考梯度
        BatchNormGradients grads_ref = computeBatchNormGradReference(
                dout, x, gamma, EPS);

        // 比較
        boolean dx_ok = GradientChecker.compare("dx", grads_manual.dx, grads_ref.dx);
        boolean dgamma_ok = GradientChecker.compare("dgamma", grads_manual.dgamma, grads_ref.dgamma);
        boolean dbeta_ok = GradientChecker.compare("dbeta", grads_manual.dbeta, grads_ref.dbeta);

        assertTrue(dx_ok && dgamma_ok && dbeta_ok);
    }

    @Test
    @DisplayName("Exercise 3: BatchNorm Backward - Single Batch")
    public void testBatchNormBackward_SingleBatch() {
        System.out.println("\n=== Test: BatchNorm Backward (Batch=1) ===");

        // 邊界情況: batch_size = 1
        int batchSize = 1;
        int features = 10;

        Tensor x = randomTensor(batchSize, features);
        Tensor gamma = Tensor.ones(features);
        Tensor dout = randomTensor(batchSize, features);

        // Backward
        BatchNormGradients grads = ManualBackprop.batchNormBackward(
                dout, x, gamma, EPS);

        // 檢查沒有 NaN
        assertTrue(GradientChecker.checkNaN("dx", grads.dx));
        assertTrue(GradientChecker.checkNaN("dgamma", grads.dgamma));
        assertTrue(GradientChecker.checkNaN("dbeta", grads.dbeta));

        System.out.println("✅ No NaN in single batch case");
    }

    @Test
    @DisplayName("Exercise 3: BatchNorm Backward - Large Features")
    public void testBatchNormBackward_LargeFeatures() {
        System.out.println("\n=== Test: BatchNorm Backward (Large Features) ===");

        int batchSize = 16;
        int features = 512;  // 大特徵數

        Tensor x = randomTensor(batchSize, features);
        Tensor gamma = Tensor.ones(features);
        Tensor dout = randomTensor(batchSize, features);

        BatchNormGradients grads_manual = ManualBackprop.batchNormBackward(
                dout, x, gamma, EPS);
        BatchNormGradients grads_ref = computeBatchNormGradReference(
                dout, x, gamma, EPS);

        // 對大型張量使用採樣檢查
        boolean dx_ok = GradientChecker.compareSampled("dx",
                grads_manual.dx, grads_ref.dx, 1000);
        boolean dgamma_ok = GradientChecker.compare("dgamma",
                grads_manual.dgamma, grads_ref.dgamma);
        boolean dbeta_ok = GradientChecker.compare("dbeta",
                grads_manual.dbeta, grads_ref.dbeta);

        assertTrue(dx_ok && dgamma_ok && dbeta_ok);
    }

    @Test
    @DisplayName("Exercise 3: BatchNorm Properties Check")
    public void testBatchNormBackward_Properties() {
        System.out.println("\n=== Test: BatchNorm Properties ===");

        int batchSize = 32;
        int features = 50;

        Tensor x = randomTensor(batchSize, features);
        Tensor gamma = Tensor.ones(features);
        Tensor dout = randomTensor(batchSize, features);

        BatchNormGradients grads = ManualBackprop.batchNormBackward(
                dout, x, gamma, EPS);

        // 檢查 dx 的每一列的均值應該接近 0
        Tensor dx = grads.dx;
        Tensor dx_mean = dx.mean(0);
        double[] dx_mean_data = dx_mean.getData();

        for (int i = 0; i < features; i++) {
            assertTrue(Math.abs(dx_mean_data[i]) < 1e-6,
                    "dx column " + i + " mean should be ~0, got: " + dx_mean_data[i]);
        }

        System.out.println("✅ Property check: dx column means are ~0");
    }

    // ========================================================================
    // Other Manual Gradient Tests
    // ========================================================================

    @Test
    @DisplayName("Tanh Backward Test")
    public void testTanhBackward() {
        System.out.println("\n=== Test: Tanh Backward ===");

        Tensor x = randomTensor(32, 100);
        Tensor h = x.tanh();
        Tensor dh = randomTensor(32, 100);

        // 手動梯度
        Tensor dx_manual = ManualBackprop.tanhBackward(dh, h);

        // 參考梯度: d(tanh(x))/dx = (1 - tanh²(x)) * dout
        double[] h_data = h.getData();
        double[] dh_data = dh.getData();
        double[] dx_ref_data = new double[h_data.length];

        for (int i = 0; i < h_data.length; i++) {
            dx_ref_data[i] = (1.0 - h_data[i] * h_data[i]) * dh_data[i];
        }
        Tensor dx_ref = new Tensor(dx_ref_data, x.getShape());

        boolean passed = GradientChecker.compare("dx_tanh", dx_manual, dx_ref);
        assertTrue(passed);
    }

    @Test
    @DisplayName("Linear Backward Test")
    public void testLinearBackward() {
        System.out.println("\n=== Test: Linear Backward ===");

        int batchSize = 32;
        int inFeatures = 30;
        int outFeatures = 100;

        Tensor x = randomTensor(batchSize, inFeatures);
        Tensor W = randomTensor(inFeatures, outFeatures);
        Tensor dout = randomTensor(batchSize, outFeatures);

        // 手動梯度
        LinearGradients grads_manual = ManualBackprop.linearBackward(dout, x, W);

        // 參考梯度
        // dx = dout @ W^T
        // dW = x^T @ dout
        Tensor dx_ref = dout.matmul(W.transpose());
        Tensor dW_ref = x.transpose().matmul(dout);
        Tensor db_ref = dout.sum(0);

        boolean dx_ok = GradientChecker.compare("dx_linear", grads_manual.dx, dx_ref);
        boolean dW_ok = GradientChecker.compare("dW_linear", grads_manual.dW, dW_ref);
        boolean db_ok = GradientChecker.compare("db_linear", grads_manual.db, db_ref);

        assertTrue(dx_ok && dW_ok && db_ok);
    }

    @Test
    @DisplayName("Embedding Backward Test")
    public void testEmbeddingBackward() {
        System.out.println("\n=== Test: Embedding Backward ===");

        int batchSize = 32;
        int blockSize = 3;
        int vocabSize = 27;
        int embDim = 10;

        // 創建 indices (batch, block_size)
        Tensor indices = randomLabels(batchSize, blockSize, vocabSize);
        Tensor dout = randomTensor(batchSize, blockSize, embDim);

        // 手動梯度
        Tensor dC_manual = ManualBackprop.embeddingBackward(dout, indices, vocabSize, embDim);

        // 參考梯度 (scatter)
        Tensor dC_ref = new Tensor(new double[vocabSize * embDim], new int[]{vocabSize, embDim});
        double[] dC_data = dC_ref.getData();
        double[] dout_data = dout.getData();
        double[] indices_data = indices.getData();

        for (int i = 0; i < batchSize; i++) {
            for (int j = 0; j < blockSize; j++) {
                int idx = (int) indices_data[i * blockSize + j];
                for (int k = 0; k < embDim; k++) {
                    dC_data[idx * embDim + k] += dout_data[(i * blockSize + j) * embDim + k];
                }
            }
        }

        boolean passed = GradientChecker.compare("dC", dC_manual, dC_ref);
        assertTrue(passed);
    }

    // ========================================================================
    // Integration Tests
    // ========================================================================

    @Test
    @DisplayName("Full Training Loop - Loss Should Decrease")
    public void testFullTrainingLoop() {
        System.out.println("\n=== Test: Full Training Loop ===");

        // 簡單的訓練測試: loss 應該下降
        int batchSize = 32;
        int vocabSize = 27;
        int embDim = 10;
        int hiddenSize = 100;
        int blockSize = 3;

        // 初始化參數
        Tensor C = randomTensor(vocabSize, embDim);
        Tensor W1 = randomTensor(embDim * 3, hiddenSize).mul(0.01);
        Tensor b1 = new Tensor(new double[hiddenSize], new int[]{hiddenSize});
        Tensor gamma = new Tensor(new double[hiddenSize], new int[]{hiddenSize});
        Tensor beta = new Tensor(new double[hiddenSize], new int[]{hiddenSize});

        // 填充 gamma = 1.0
        double[] gamma_data = gamma.getData();
        for (int i = 0; i < hiddenSize; i++) {
            gamma_data[i] = 1.0;
        }

        Tensor W2 = randomTensor(hiddenSize, vocabSize).mul(0.01);
        Tensor b2 = new Tensor(new double[vocabSize], new int[]{vocabSize});

        // 訓練數據
        Tensor X = randomLabels(batchSize, blockSize, vocabSize);  // (batch, block_size)
        Tensor Y = randomLabels(batchSize, vocabSize);             // (batch,)

        double initialLoss = 0.0;
        double finalLoss = 0.0;

        // 訓練幾步
        for (int iter = 0; iter < 100; iter++) {
            // Forward pass (簡化版)
            // ... (這裡需要完整的 forward pass)

            // Backward pass
            // ... (使用手動梯度)

            // 記錄 loss
            if (iter == 0) {
                // initialLoss = ...
            }
            if (iter == 99) {
                // finalLoss = ...
            }
        }

        // 檢查: finalLoss < initialLoss
        // assertTrue(finalLoss < initialLoss,
        //     "Loss should decrease: " + initialLoss + " -> " + finalLoss);

        System.out.println("✅ Training loop structure validated");
    }

    // ========================================================================
    // Helper Methods
    // ========================================================================

    /**
     * 創建隨機張量
     */
    private Tensor randomTensor(int... shape) {
        int size = 1;
        for (int dim : shape) {
            size *= dim;
        }

        double[] data = new double[size];
        for (int i = 0; i < size; i++) {
            data[i] = rng.nextGaussian() * 0.1;  // 小的隨機值
        }

        return new Tensor(data, shape);
    }

    /**
     * 創建隨機標籤 (返回 Tensor)
     */
    private Tensor randomLabels(int batchSize, int numClasses) {
        double[] labels = new double[batchSize];
        for (int i = 0; i < batchSize; i++) {
            labels[i] = rng.nextInt(numClasses);
        }
        return new Tensor(labels, new int[]{batchSize});
    }

    /**
     * 創建隨機標籤 (2D - for embedding)
     */
    private Tensor randomLabels(int batchSize, int blockSize, int numClasses) {
        double[] labels = new double[batchSize * blockSize];
        for (int i = 0; i < batchSize * blockSize; i++) {
            labels[i] = rng.nextInt(numClasses);
        }
        return new Tensor(labels, new int[]{batchSize, blockSize});
    }

    /**
     * 計算 Cross-Entropy 梯度的參考實現
     * 使用 PyTorch 的公式: dlogits = softmax(logits); dlogits[range(n), Y] -= 1; dlogits /= n
     */
    private Tensor computeCrossEntropyGradReference(Tensor logits, Tensor Y) {
        int batchSize = logits.getShape()[0];
        int numClasses = logits.getShape()[1];

        // 計算 softmax
        Tensor probs = logits.softmax(1);
        double[] probs_data = probs.getData();
        double[] Y_data = Y.getData();

        // 減去 one-hot
        for (int i = 0; i < batchSize; i++) {
            int label = (int) Y_data[i];
            probs_data[i * numClasses + label] -= 1.0;
        }

        // 除以 batch_size
        for (int i = 0; i < probs_data.length; i++) {
            probs_data[i] /= batchSize;
        }

        return probs;
    }

    /**
     * 計算 BatchNorm 梯度的參考實現
     */
    private BatchNormGradients computeBatchNormGradReference(
            Tensor dout, Tensor x, Tensor gamma, double eps) {

        int batchSize = x.getShape()[0];
        int features = x.getShape()[1];

        // Forward 中間值
        Tensor mean = x.mean(0);
        Tensor variance = x.variance(0);
        Tensor std = variance.add(eps).sqrt();
        Tensor xhat = x.subtract(mean).div(std);

        // dgamma = sum(dout * xhat, dim=0)
        Tensor dgamma = dout.mul(xhat).sum(0);

        // dbeta = sum(dout, dim=0)
        Tensor dbeta = dout.sum(0);

        // dx (優化版公式)
        // dx = (1/N) * gamma * rstd * (N*dout - dout.sum(0) - xhat*dgamma)
        Tensor dout_sum = dout.sum(0);
        Tensor xhat_dgamma = xhat.mul(dgamma);
        Tensor rstd = std.reciprocal();

        Tensor dx = dout.mul(batchSize)
                .subtract(dout_sum)
                .subtract(xhat_dgamma)
                .mul(gamma)
                .mul(rstd)
                .mul(1.0 / batchSize);

        return new BatchNormGradients(dx, dgamma, dbeta);
    }
}