package com.makemore.wavenet;

import com.makemore.mlp.Tensor;
import org.junit.jupiter.api.*;

import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import static org.junit.jupiter.api.Assertions.*;

/**
 * WaveNet Performance & Quality Tests (JUnit 5)
 */
@TestMethodOrder(MethodOrderer.OrderAnnotation.class)
public class WaveNetPerformanceTest {

    // =========================================================================
    // Test 1: Quick Training (Smoke Test)
    // =========================================================================

    @Test
    @Order(1)
    @DisplayName("Quick training should reduce loss")
    void testQuickTraining() throws IOException {
        // ✅ 增大模型
        WaveNetLanguageModel model = new WaveNetLanguageModel(4, 24, 64);

        model.loadData("names.txt");
        model.buildDataset();
        model.buildModel();

        double initialLoss = evaluateLoss(model, model.getXtr(), model.getYtr(), 32);

        List<Tensor> params = model.getParameters();
        double lr = 0.001;  // ✅ 降低 LR

        // ✅ 增加步數
        for (int step = 0; step < 15000; step++) {
            MiniBatch batch = getMiniBatch(model.getXtr(), model.getYtr(), 32);
            Tensor logits = model.forward(batch.X);
            Tensor loss = crossEntropyLossTensor(logits, batch.Y);

            zeroGrad(params);
            loss.backward();

            for (Tensor p : params) {
                double[] d = p.getData();
                double[] g = p.getGrad();
                for (int i = 0; i < d.length; i++) {
                    d[i] -= lr * g[i];
                }
            }

            // ✅ 可選：每 500 步印一次 loss 觀察
            if (step % 500 == 0) {
                System.out.printf("Step %d: loss=%.4f\n", step, loss.item());
            }
        }

        double finalLoss = evaluateLoss(model, model.getXtr(), model.getYtr(), 32);

        System.out.println("Initial loss = " + initialLoss);
        System.out.println("Final loss   = " + finalLoss);
        System.out.println("Improvement  = " + (initialLoss - finalLoss));

        assertTrue(finalLoss < initialLoss,
                "Loss should decrease after training");
        assertTrue(initialLoss - finalLoss > 0.1,
                "Loss improvement should be meaningful");
    }

    // =========================================================================
    // Test 2: Generation Quality (Does not crash)
    // =========================================================================

    @Test
    @Order(2)
    @DisplayName("Model should generate samples without crashing")
    void testGenerationQuality() throws IOException {
        WaveNetLanguageModel model =
                new WaveNetLanguageModel(8, 24, 128);

        model.loadData("names.txt");
        model.buildDataset();
        model.buildModel();

        List<Tensor> params = model.getParameters();
        double lr = 0.01;

        for (int step = 0; step < 3000; step++) {
            MiniBatch batch = getMiniBatch(model.getXtr(), model.getYtr(), 32);
            Tensor logits = model.forward(batch.X);
            Tensor loss = crossEntropyLossTensor(logits, batch.Y);

            zeroGrad(params);
            loss.backward();

            for (Tensor p : params) {
                double[] d = p.getData();
                double[] g = p.getGrad();
                for (int i = 0; i < d.length; i++) {
                    d[i] -= lr * g[i];
                }
            }
        }

        for (int i = 0; i < 5; i++) {
            String sample = model.sample(20);
            System.out.println("Sample: " + sample);

            assertNotNull(sample);
            assertFalse(sample.isEmpty());
        }
    }

    // =========================================================================
    // Test 3: Loss Convergence
    // =========================================================================

    @Test
    @Order(3)
    @DisplayName("Validation loss should trend downward")
    void testLossConvergence() throws IOException {
        WaveNetLanguageModel model =
                new WaveNetLanguageModel(8, 24, 128);

        model.loadData("names.txt");
        model.buildDataset();
        model.buildModel();

        List<Tensor> params = model.getParameters();
        double lr = 0.01;

        double prevLoss = Double.MAX_VALUE;

        for (int epoch = 0; epoch < 5; epoch++) {
            for (int step = 0; step < 1000; step++) {
                MiniBatch batch = getMiniBatch(model.getXtr(), model.getYtr(), 32);
                Tensor logits = model.forward(batch.X);
                Tensor loss = crossEntropyLossTensor(logits, batch.Y);

                zeroGrad(params);
                loss.backward();

                for (Tensor p : params) {
                    double[] d = p.getData();
                    double[] g = p.getGrad();
                    for (int i = 0; i < d.length; i++) {
                        d[i] -= lr * g[i];
                    }
                }
            }

            double valLoss =
                    evaluateLoss(model, model.getXdev(), model.getYdev(), 32);

            System.out.println("Epoch " + epoch + " val loss = " + valLoss);

            assertTrue(valLoss < prevLoss || epoch == 0,
                    "Validation loss should not increase sharply");

            prevLoss = valLoss;
        }
    }

    // =========================================================================
    // Helper Methods (unchanged)
    // =========================================================================

    private static double evaluateLoss(WaveNetLanguageModel model,
                                       Tensor X, Tensor Y, int batchSize) {
        int n = X.getShape()[0];
        double totalLoss = 0.0;
        int totalSamples = 0;

        System.out.println("\n=== EVALUATING ===");
        System.out.println("Total samples: " + n);
        System.out.println("Batch size: " + batchSize);

        for (int i = 0; i < n; i += batchSize) {
            int end = Math.min(i + batchSize, n);
            int currentBatchSize = end - i;

            double[] batchX = Arrays.copyOfRange(X.getData(),
                    i * X.getShape()[1], end * X.getShape()[1]);
            double[] batchY = Arrays.copyOfRange(Y.getData(), i, end);

            Tensor batchXT = new Tensor(batchX, new int[]{currentBatchSize, X.getShape()[1]});
            Tensor batchYT = new Tensor(batchY, new int[]{currentBatchSize});

            Tensor logits = model.forward(batchXT);
            Tensor loss = crossEntropyLossTensor(logits, batchYT);

            double batchLoss = loss.item();

            // ✅ Debug 前 3 個 batch
            if (i / batchSize < 3) {
                System.out.printf("Batch %d: size=%d, loss=%.4f, weighted=%.4f\n",
                        i / batchSize, currentBatchSize, batchLoss, batchLoss * currentBatchSize);
            }

            totalLoss += batchLoss * currentBatchSize;
            totalSamples += currentBatchSize;
        }

        double avgLoss = totalLoss / totalSamples;
        System.out.printf("Total loss: %.4f, Total samples: %d, Avg: %.4f\n",
                totalLoss, totalSamples, avgLoss);
        System.out.println("=================\n");

        return avgLoss;
    }



    private static MiniBatch getMiniBatch(Tensor X, Tensor Y, int batchSize) {
        int numSamples = X.getShape()[0];
        int blockSize = X.getShape()[1];

        Random rng = new Random(); // 固定 seed
        int[] indices = new int[batchSize];

        for (int i = 0; i < batchSize; i++) {
            indices[i] = rng.nextInt(numSamples);
        }

        double[] xData = X.getData();
        double[] yData = Y.getData();

        double[] batchX = new double[batchSize * blockSize];
        double[] batchY = new double[batchSize];

        for (int i = 0; i < batchSize; i++) {
            int idx = indices[i];
            System.arraycopy(xData, idx * blockSize,
                    batchX, i * blockSize, blockSize);
            batchY[i] = yData[idx];
        }

        return new MiniBatch(
                new Tensor(batchX, new int[]{batchSize, blockSize}),
                new Tensor(batchY, new int[]{batchSize})
        );
    }

    private static MiniBatch sliceBatch(Tensor X, Tensor Y, int start, int end) {
        int batchSize = end - start;
        int blockSize = X.getShape()[1];

        double[] batchX = new double[batchSize * blockSize];
        double[] batchY = new double[batchSize];

        System.arraycopy(X.getData(), start * blockSize,
                batchX, 0, batchSize * blockSize);
        System.arraycopy(Y.getData(), start,
                batchY, 0, batchSize);

        return new MiniBatch(
                new Tensor(batchX, new int[]{batchSize, blockSize}),
                new Tensor(batchY, new int[]{batchSize})
        );
    }

    private static void zeroGrad(List<Tensor> params) {
        for (Tensor p : params) {
            p.zeroGrad();
        }
    }

    private static class MiniBatch {
        Tensor X;
        Tensor Y;
        MiniBatch(Tensor X, Tensor Y) {
            this.X = X;
            this.Y = Y;
        }
    }

    private static Tensor crossEntropyLossTensor(Tensor logits, Tensor Y) {
        int batchSize = logits.getShape()[0];
        int vocabSize = logits.getShape()[1];

        double[] logitsData = logits.getData();
        double[] yData = Y.getData();

        // ✅ 使用 log-sum-exp trick
        double totalLoss = 0.0;

        for (int i = 0; i < batchSize; i++) {
            int target = (int) yData[i];

            // 找最大值（數值穩定性）
            double maxLogit = Double.NEGATIVE_INFINITY;
            for (int j = 0; j < vocabSize; j++) {
                maxLogit = Math.max(maxLogit, logitsData[i * vocabSize + j]);
            }

            // log-sum-exp
            double sumExp = 0.0;
            for (int j = 0; j < vocabSize; j++) {
                sumExp += Math.exp(logitsData[i * vocabSize + j] - maxLogit);
            }
            double logSumExp = maxLogit + Math.log(sumExp);

            // Cross-entropy
            double loss = -(logitsData[i * vocabSize + target] - logSumExp);
            totalLoss += loss;
        }

        double lossValue = totalLoss / batchSize;

        Tensor lossTensor = new Tensor(new double[]{lossValue}, new int[]{1}, true);
        lossTensor.prev.add(logits);
        lossTensor.op = "cross_entropy";

        lossTensor.backward = (v) -> {
            if (logits.isRequiresGrad()) {
                double[] logitsGrad = logits.getGrad();

                for (int i = 0; i < batchSize; i++) {
                    int target = (int) yData[i];

                    // 重新計算數值穩定的 softmax
                    double maxLogit = Double.NEGATIVE_INFINITY;
                    for (int j = 0; j < vocabSize; j++) {
                        maxLogit = Math.max(maxLogit, logitsData[i * vocabSize + j]);
                    }

                    double sumExp = 0.0;
                    for (int j = 0; j < vocabSize; j++) {
                        sumExp += Math.exp(logitsData[i * vocabSize + j] - maxLogit);
                    }

                    // Gradient: softmax - one_hot
                    for (int j = 0; j < vocabSize; j++) {
                        double softmax = Math.exp(logitsData[i * vocabSize + j] - maxLogit) / sumExp;
                        double grad = softmax;
                        if (j == target) {
                            grad -= 1.0;
                        }
                        grad /= batchSize;

                        logitsGrad[i * vocabSize + j] += grad;
                    }
                }
            }
            return null;
        };

        return lossTensor;
    }
}
