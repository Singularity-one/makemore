package com.makemore;

import com.makemore.backprop.ManualBackprop;
import com.makemore.backprop.ManualBackprop.*;
import com.makemore.mlp.Tensor;
import com.makemore.wavenet.WaveNetLanguageModel;

import java.io.*;
import java.nio.file.*;
import java.util.*;

/**
 * WaveNet Training Main Program
 *
 * Trains the WaveNet language model on character-level sequences.
 *
 * Based on Karpathy's makemore Part 5 (WaveNet)
 */
public class Main {

    public static void main(String[] args) throws IOException {
        System.out.println("=".repeat(80));
        System.out.println("WaveNet Character-Level Language Model");
        System.out.println("Based on Andrej Karpathy's makemore Part 5");
        System.out.println("=".repeat(80) + "\n");

        // ========================================================================
        // Hyperparameters
        // ========================================================================

        int blockSize = 8;        // Context length (MUST be power of 2)
        int embeddingDim = 24;    // Embedding dimension
        int hiddenSize = 128;     // Hidden layer size

        // Training hyperparameters
        int maxSteps = 50000;    // Training iterations
        int batchSize = 32;       // Mini-batch size //test use 16
        double learningRate = 0.01; // Initial learning rate
        double lrDecay = 0.99999; // Learning rate decay per step

        System.out.println("Hyperparameters:");
        System.out.println("  Block size: " + blockSize);
        System.out.println("  Embedding dim: " + embeddingDim);
        System.out.println("  Hidden size: " + hiddenSize);
        System.out.println("  Batch size: " + batchSize);
        System.out.println("  Max steps: " + maxSteps);
        System.out.println("  Learning rate: " + learningRate);
        System.out.println("  LR decay: " + lrDecay);
        System.out.println();

        // ========================================================================
        // 1. Initialize Model
        // ========================================================================

        System.out.println("Initializing model...");
        WaveNetLanguageModel model = new WaveNetLanguageModel(
                blockSize, embeddingDim, hiddenSize
        );

        // ========================================================================
        // 2. Load Data
        // ========================================================================

        System.out.println("Loading data...");
        model.loadData("names.txt");

        System.out.println("\nBuilding dataset...");
        model.buildDataset();

        // ========================================================================
        // 3. Build Model
        // ========================================================================

        System.out.println("\nBuilding model architecture...");
        model.buildModel();

        List<Tensor> parameters = model.getParameters();
        System.out.println("\nTotal parameters: " +
                parameters.stream().mapToInt(Tensor::getSize).sum());

        // ========================================================================
        // 4. Initial Evaluation
        // ========================================================================

        System.out.println("\n" + "=".repeat(80));
        System.out.println("Initial Evaluation (before training)");
        System.out.println("=".repeat(80));

        double trainLoss = evaluateSplit(model, model.getXtr(), model.getYtr(), batchSize);
        double valLoss = evaluateSplit(model, model.getXdev(), model.getYdev(), batchSize);

        System.out.printf("Train loss: %.4f\n", trainLoss);
        System.out.printf("Val loss:   %.4f\n", valLoss);

        System.out.println("\nSample generations (before training):");
        for (int i = 0; i < 5; i++) {
            System.out.println("  " + model.sample(20));
        }

        // ========================================================================
        // 5. Training Loop
        // ========================================================================

        System.out.println("\n" + "=".repeat(80));
        System.out.println("Training...");
        System.out.println("=".repeat(80) + "\n");

        long startTime = System.currentTimeMillis();
        double currentLR = learningRate;

        for (int step = 0; step < maxSteps; step++) {
            MiniBatch batch = getMiniBatch(model.getXtr(), model.getYtr(), batchSize);

            Tensor logits = model.forward(batch.X);

            // ✅ 計算 loss
            Tensor lossTensor = crossEntropyLossTensor(logits, batch.Y);
            double loss = lossTensor.item();  // ← 直接叫 loss

            // ✅ Backward
            zeroGrad(parameters);
            lossTensor.backward();

            // ✅ 檢查 logits 的梯度
            if (step == 0) {
                double[] logitsGrad = logits.getGrad();
                double gradSum = 0.0;
                for (double g : logitsGrad) {
                    gradSum += Math.abs(g);
                }
                System.out.println("Logits grad sum: " + gradSum);
                System.out.println("Loss tensor grad: " + java.util.Arrays.toString(lossTensor.getGrad()));
            }

            // ✅ 第一個 step 檢查梯度
            if (step == 0) {
                System.out.println("\n=== GRADIENT CHECK ===");
                List<Tensor> params = model.getParameters();
                for (int i = 0; i < params.size(); i++) {
                    Tensor p = params.get(i);
                    double[] grad = p.getGrad();

                    double gradSum = 0.0;
                    double gradMax = 0.0;
                    for (double g : grad) {
                        gradSum += Math.abs(g);
                        gradMax = Math.max(gradMax, Math.abs(g));
                    }

                    System.out.printf("Param %d: grad_sum=%.6f, grad_max=%.6f, shape=%s\n",
                            i, gradSum, gradMax, java.util.Arrays.toString(p.getShape()));
                }
                System.out.println("======================\n");
            }

            // SGD update
            for (Tensor param : parameters) {
                double[] paramData = param.getData();
                double[] paramGrad = param.getGrad();
                for (int i = 0; i < paramData.length; i++) {
                    paramData[i] -= currentLR * paramGrad[i];
                }
            }

            currentLR *= lrDecay;

            // ✅ 每 2000 步採樣一次，觀察進步
            if (step % 2000 == 0) {
                String sample = model.sample(20);
                System.out.println("Step " + step + " sample: " + sample);
            }


            // Logging
            if (step % 10000 == 0 || step == maxSteps - 1) {
                long elapsed = System.currentTimeMillis() - startTime;
                double stepsPerSec = (step + 1) / (elapsed / 1000.0);

                System.out.printf("Step %6d/%d | Loss: %.4f | LR: %.6f | %.1f steps/sec\n",
                        step, maxSteps, loss, currentLR, stepsPerSec);
            }

            // Periodic evaluation
            if (step % 50000 == 0 && step > 0) {
                System.out.println("\n" + "-".repeat(80));
                System.out.println("Checkpoint at step " + step);
                System.out.println("-".repeat(80));

                trainLoss = evaluateSplit(model, model.getXtr(), model.getYtr(), batchSize);
                valLoss = evaluateSplit(model, model.getXdev(), model.getYdev(), batchSize);

                System.out.printf("Train loss: %.4f\n", trainLoss);
                System.out.printf("Val loss:   %.4f\n", valLoss);

                System.out.println("\nSample generations:");
                for (int i = 0; i < 5; i++) {
                    System.out.println("  " + model.sample(20));
                }
                System.out.println();
            }
        }

        // ========================================================================
        // 6. Final Evaluation
        // ========================================================================

        System.out.println("\n" + "=".repeat(80));
        System.out.println("Final Evaluation");
        System.out.println("=".repeat(80));

        trainLoss = evaluateSplit(model, model.getXtr(), model.getYtr(), batchSize);
        valLoss = evaluateSplit(model, model.getXdev(), model.getYdev(), batchSize);

        System.out.printf("Train loss: %.4f\n", trainLoss);
        System.out.printf("Val loss:   %.4f\n", valLoss);

        // ========================================================================
        // 7. Generate Samples
        // ========================================================================

        System.out.println("\n" + "=".repeat(80));
        System.out.println("Generated Samples");
        System.out.println("=".repeat(80) + "\n");

        for (int i = 0; i < 20; i++) {
            String name = model.sample(20);
            System.out.println((i + 1) + ". " + name);
        }

        // ========================================================================
        // 8. Summary
        // ========================================================================

        long totalTime = System.currentTimeMillis() - startTime;
        System.out.println("\n" + "=".repeat(80));
        System.out.println("Training Complete!");
        System.out.println("=".repeat(80));
        System.out.printf("Total time: %.2f minutes\n", totalTime / 60000.0);
        System.out.printf("Final validation loss: %.4f\n", valLoss);
        System.out.println("\n🎉 WaveNet training finished!");
    }

    // ============================================================================
    // Helper Methods
    // ============================================================================

    /**
     * Evaluate loss on a dataset split
     */
    private static double evaluateSplit(WaveNetLanguageModel model,
                                        Tensor X, Tensor Y, int batchSize) {
        model.getModel().setTraining(false);

        int numSamples = X.getShape()[0];
        int numBatches = (numSamples + batchSize - 1) / batchSize;
        double totalLoss = 0.0;
        int totalSamples = 0;

        for (int i = 0; i < numBatches; i++) {
            int start = i * batchSize;
            int end = Math.min(start + batchSize, numSamples);

            if (end - start < 1) continue;

            MiniBatch batch = sliceBatch(X, Y, start, end);

            Tensor logits = model.forward(batch.X);
            double loss = crossEntropyLoss(logits, batch.Y);

            totalLoss += loss * (end - start);
            totalSamples += (end - start);
        }

        model.getModel().setTraining(true);
        return totalLoss / totalSamples;
    }

    /**
     * Get a random mini-batch
     */
    private static MiniBatch getMiniBatch(Tensor X, Tensor Y, int batchSize) {
        int numSamples = X.getShape()[0];
        int[] indices = new int[batchSize];

        java.util.Random rng = new java.util.Random();
        for (int i = 0; i < batchSize; i++) {
            indices[i] = rng.nextInt(numSamples);
        }

        return selectBatch(X, Y, indices);
    }

    /**
     * Select specific samples by indices
     */
    private static MiniBatch selectBatch(Tensor X, Tensor Y, int[] indices) {
        int batchSize = indices.length;
        int[] xShape = X.getShape();
        int blockSize = xShape[1];

        double[] xData = X.getData();
        double[] yData = Y.getData();

        double[] batchXData = new double[batchSize * blockSize];
        double[] batchYData = new double[batchSize];

        for (int i = 0; i < batchSize; i++) {
            int idx = indices[i];

            // Copy X
            System.arraycopy(xData, idx * blockSize, batchXData, i * blockSize, blockSize);

            // Copy Y
            batchYData[i] = yData[idx];
        }

        Tensor batchX = new Tensor(batchXData, new int[]{batchSize, blockSize});
        Tensor batchY = new Tensor(batchYData, new int[]{batchSize});

        return new MiniBatch(batchX, batchY);
    }

    /**
     * Slice a batch from dataset
     */
    private static MiniBatch sliceBatch(Tensor X, Tensor Y, int start, int end) {
        int batchSize = end - start;
        int blockSize = X.getShape()[1];

        double[] xData = X.getData();
        double[] yData = Y.getData();

        double[] batchXData = new double[batchSize * blockSize];
        double[] batchYData = new double[batchSize];

        System.arraycopy(xData, start * blockSize, batchXData, 0, batchSize * blockSize);
        System.arraycopy(yData, start, batchYData, 0, batchSize);

        Tensor batchX = new Tensor(batchXData, new int[]{batchSize, blockSize});
        Tensor batchY = new Tensor(batchYData, new int[]{batchSize});

        return new MiniBatch(batchX, batchY);
    }

    /**
     * Cross-entropy loss
     */
    private static double crossEntropyLoss(Tensor logits, Tensor Y) {
        return crossEntropyLossTensor(logits, Y).item();
    }

    /**
     * Zero all gradients
     */
    private static void zeroGrad(List<Tensor> parameters) {
        for (Tensor param : parameters) {
            param.zeroGrad();
        }
    }

    /**
     * Mini-batch container
     */
    private static class MiniBatch {
        Tensor X;
        Tensor Y;

        MiniBatch(Tensor X, Tensor Y) {
            this.X = X;
            this.Y = Y;
        }
    }

    public static Tensor crossEntropyLossTensor(Tensor logits, Tensor Y) {
        int batchSize = logits.getShape()[0];
        int vocabSize = logits.getShape()[1];

        double[] logitsData = logits.getData();
        double[] yData = Y.getData();

        // ✅ 使用 log-sum-exp trick 保證數值穩定
        double totalLoss = 0.0;

        for (int i = 0; i < batchSize; i++) {
            int target = (int) yData[i];

            // 找到這個樣本的最大 logit（數值穩定性）
            double maxLogit = Double.NEGATIVE_INFINITY;
            for (int j = 0; j < vocabSize; j++) {
                maxLogit = Math.max(maxLogit, logitsData[i * vocabSize + j]);
            }

            // log-sum-exp: log(sum(exp(x))) = max + log(sum(exp(x - max)))
            double sumExp = 0.0;
            for (int j = 0; j < vocabSize; j++) {
                sumExp += Math.exp(logitsData[i * vocabSize + j] - maxLogit);
            }
            double logSumExp = maxLogit + Math.log(sumExp);

            // Cross-entropy: -log(softmax[target]) = -(logit[target] - log_sum_exp)
            double loss = -(logitsData[i * vocabSize + target] - logSumExp);
            totalLoss += loss;
        }

        double lossValue = totalLoss / batchSize;

        // 創建 loss tensor
        Tensor lossTensor = new Tensor(new double[]{lossValue}, new int[]{1}, true);
        lossTensor.prev.add(logits);
        lossTensor.op = "cross_entropy";

        // ✅ Backward（使用數值穩定的 softmax）
        lossTensor.backward = (v) -> {
            if (logits.isRequiresGrad()) {
                double[] logitsGrad = logits.getGrad();

                for (int i = 0; i < batchSize; i++) {
                    int target = (int) yData[i];

                    // 計算數值穩定的 softmax
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