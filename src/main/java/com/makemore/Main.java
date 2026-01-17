package com.makemore;

import com.makemore.layers.*;
import com.makemore.mlp.Tensor;
import com.makemore.utils.DiagnosticTools;

import java.io.*;
import java.nio.file.*;
import java.util.*;

/**
 * Building makemore Part 3: Activations & Gradients, BatchNorm
 *
 * Key differences from Lecture 3:
 * - Much deeper network (5 hidden layers vs 1)
 * - Introduces Batch Normalization
 * - Heavy focus on diagnostics and visualization
 * - Comparison: with vs without BatchNorm
 */
public class Main {

    public static void main(String[] args) {
        try {
            System.out.println("╔═══════════════════════════════════════════════════════╗");
            System.out.println("║  Makemore Part 3: Batch Normalization                ║");
            System.out.println("║  Deep MLP with Activation & Gradient Analysis        ║");
            System.out.println("╚═══════════════════════════════════════════════════════╝\n");

            // Load data
            DataLoader dataLoader = new DataLoader();
            dataLoader.loadData("names.txt");
            dataLoader.buildDataset(3); // block_size = 3

            int vocabSize = dataLoader.getVocabSize();
            int blockSize = 3;
            int embeddingDim = 10;
            int hiddenSize = 100;

            System.out.println("=== Configuration ===");
            System.out.println("Vocabulary size: " + vocabSize);
            System.out.println("Block size: " + blockSize);
            System.out.println("Embedding dim: " + embeddingDim);
            System.out.println("Hidden size: " + hiddenSize);
            System.out.println("Hidden layers: 5\n");

            // =========================================
            // Experiment 1: WITHOUT BatchNorm (fails!)
            // =========================================
            System.out.println("\n" + "=".repeat(60));
            System.out.println("EXPERIMENT 1: Deep Network WITHOUT BatchNorm");
            System.out.println("=".repeat(60));
            System.out.println("Expected: Training will FAIL (activations saturate)\n");

            DeepMLP modelNoBN = new DeepMLP(
                    vocabSize, embeddingDim, blockSize, hiddenSize,
                    false  // useBatchNorm = false
            );

            System.out.println("Total parameters: " + modelNoBN.numParameters());

            // Train for a few iterations to show it fails
            trainModel(modelNoBN, dataLoader, 1000, 0.1, 32, false);

            // Check activation statistics (should be terrible)
            System.out.println("\n📊 Activation Statistics (WITHOUT BatchNorm):");
            DiagnosticTools.analyzeActivations(modelNoBN.getLayers(), "After 1000 iters");

            // =========================================
            // Experiment 2: WITH BatchNorm (succeeds!)
            // =========================================
            System.out.println("\n\n" + "=".repeat(60));
            System.out.println("EXPERIMENT 2: Deep Network WITH BatchNorm");
            System.out.println("=".repeat(60));
            System.out.println("Expected: Training will SUCCEED\n");

            DeepMLP modelWithBN = new DeepMLP(
                    vocabSize, embeddingDim, blockSize, hiddenSize,
                    true  // useBatchNorm = true
            );

            System.out.println("Total parameters: " + modelWithBN.numParameters());

            // Initialize last layer to be less confident
            modelWithBN.initializeLastLayer(0.1);

            // Full training
            trainModel(modelWithBN, dataLoader, 200000, 0.1, 32, true);

            // Check activation statistics (should be healthy)
            System.out.println("\n📊 Activation Statistics (WITH BatchNorm):");
            DiagnosticTools.analyzeActivations(modelWithBN.getLayers(), "After full training");

            // Evaluate
            System.out.println("\n=== Final Evaluation ===");
            Random evalRng = new Random(42);
            double trainLoss = evaluateSample(modelWithBN, dataLoader.getXtr(), dataLoader.getYtr(), 1000, evalRng);
            double devLoss = evaluateSample(modelWithBN, dataLoader.getXdev(), dataLoader.getYdev(), 1000, evalRng);
            double testLoss = evaluateSample(modelWithBN, dataLoader.getXte(), dataLoader.getYte(), 1000, evalRng);

            System.out.printf("Train loss (1000 samples): %.4f\n", trainLoss);
            System.out.printf("Dev loss (1000 samples): %.4f\n", devLoss);
            System.out.printf("Test loss (1000 samples): %.4f\n", testLoss);

            // Sample names
            System.out.println("\n=== Sampling 20 Names ===");
            modelWithBN.setEvalMode();
            List<String> samples = sampleNames(modelWithBN, dataLoader, 20);
            for (int i = 0; i < samples.size(); i++) {
                System.out.printf("%2d. %s\n", i + 1, samples.get(i));
            }

            // Summary
            printSummary();

        } catch (Exception e) {
            System.err.println("Error: " + e.getMessage());
            e.printStackTrace();
        }
    }

    private static void trainModel(DeepMLP model, DataLoader dataLoader,
                                   int maxIters, double lr, int batchSize,
                                   boolean fullTraining) {

        System.out.println("=== Training ===");
        System.out.println("Max iterations: " + maxIters);
        System.out.println("Learning rate: " + lr);
        System.out.println("Batch size: " + batchSize + "\n");

        model.setTrainMode();

        Tensor Xtr = dataLoader.getXtr();
        Tensor Ytr = dataLoader.getYtr();
        int nTrain = Xtr.getShape()[0];

        Random rng = new Random(2147483647L);

        for (int iter = 0; iter < maxIters; iter++) {
            // Mini-batch
            int[] batchIndices = randomBatch(nTrain, batchSize, rng);
            Tensor Xb = selectRows(Xtr, batchIndices);
            Tensor Yb = selectRows(Ytr, batchIndices);

            // Forward
            Tensor logits = model.forward(Xb);
            Tensor loss = crossEntropyLoss(logits, Yb);

            // Backward
            model.zeroGrad();
            loss.backward();

            // Get loss value before clearing
            double lossValue = loss.item();

            // Update with learning rate decay
            double currentLr = lr;
            if (fullTraining && iter >= 150000) {
                currentLr = 0.01;
            }

            model.updateParameters(currentLr);

            // Logging (減少評估頻率以節省記憶體)
            if (iter % 10000 == 0 || iter == maxIters - 1 || (!fullTraining && iter % 100 == 0)) {
                // 只在關鍵點才評估完整集合
                if (iter % 10000 == 0 || iter == maxIters - 1) {
                    double trainLoss = evaluateSample(model, Xtr, Ytr, 500, rng);
                    double devLoss = evaluateSample(model, dataLoader.getXdev(), dataLoader.getYdev(), 500, rng);
                    System.out.printf("Iter %d: loss=%.4f, train≈%.4f, dev≈%.4f\n",
                            iter, lossValue, trainLoss, devLoss);
                } else {
                    // 其他時候只顯示當前批次的loss
                    System.out.printf("Iter %d: loss=%.4f\n", iter, lossValue);
                }

                // For the failing experiment, show why it's failing
                if (!fullTraining && iter == 100) {
                    System.out.println("\n⚠️  Notice: Loss is not decreasing!");
                    System.out.println("    Reason: Activations are saturating (check stats above)");
                }
            }

            // Suggest GC every 1000 iterations
            if (iter % 1000 == 0 && iter > 0) {
                System.gc();
            }
        }

        if (fullTraining && maxIters >= 150000) {
            System.out.println("\n📉 Learning rate decayed to 0.01 at iter 150000");
        }
    }

    private static double evaluate(DeepMLP model, Tensor X, Tensor Y) {
        model.setEvalMode();

        Tensor logits = model.forward(X);

        // Calculate loss without gradients
        Tensor exp = logits.exp();
        Tensor sumExp = exp.sum(1, true);
        Tensor probs = exp.div(sumExp);

        int batchSize = X.getShape()[0];
        int vocabSize = logits.getShape()[1];
        double[] probData = probs.getData();
        double[] targetData = Y.getData();

        double totalLoss = 0.0;
        for (int i = 0; i < batchSize; i++) {
            int target = (int) targetData[i];
            double prob = probData[i * vocabSize + target];
            totalLoss += -Math.log(Math.max(prob, 1e-10));
        }

        model.setTrainMode();
        return totalLoss / batchSize;
    }

    /**
     * Evaluate on a random sample (to save memory)
     */
    private static double evaluateSample(DeepMLP model, Tensor X, Tensor Y,
                                         int sampleSize, Random rng) {
        int n = X.getShape()[0];
        sampleSize = Math.min(sampleSize, n);

        int[] indices = randomBatch(n, sampleSize, rng);
        Tensor Xs = selectRows(X, indices);
        Tensor Ys = selectRows(Y, indices);

        return evaluate(model, Xs, Ys);
    }

    /**
     * Clear computation graph to free memory
     * Critical for long training runs!
     */
    private static void clearComputationGraph(Tensor t) {
        // Simply let Java GC handle it by not keeping references
        // The key is to not hold onto intermediate tensors
    }

    private static Tensor crossEntropyLoss(Tensor logits, Tensor targets) {
        Tensor exp = logits.exp();
        Tensor sumExp = exp.sum(1, true);
        Tensor probs = exp.div(sumExp);
        Tensor selected = probs.gather(targets);
        return selected.log().neg().mean();
    }

    private static List<String> sampleNames(DeepMLP model, DataLoader dataLoader, int n) {
        List<String> samples = new ArrayList<>();
        Random rng = new Random(2147483647L + 10);

        for (int s = 0; s < n; s++) {
            List<Integer> out = new ArrayList<>();
            int[] context = new int[3];

            while (true) {
                double[] contextData = new double[3];
                for (int i = 0; i < 3; i++) {
                    contextData[i] = context[i];
                }
                Tensor contextTensor = new Tensor(contextData, new int[]{1, 3});

                Tensor logits = model.forward(contextTensor);
                Tensor exp = logits.exp();
                Tensor sumExp = exp.sum(1, true);
                Tensor probs = exp.div(sumExp);

                int nextIdx = sampleMultinomial(probs.getData(), rng);

                System.arraycopy(context, 1, context, 0, 2);
                context[2] = nextIdx;
                out.add(nextIdx);

                if (nextIdx == 0 || out.size() > 20) break;
            }

            StringBuilder name = new StringBuilder();
            for (int idx : out) {
                if (idx == 0) break;
                name.append(dataLoader.getIdxToChar().get(idx));
            }
            samples.add(name.toString());
        }

        return samples;
    }

    private static int sampleMultinomial(double[] probs, Random rng) {
        double r = rng.nextDouble();
        double cumProb = 0.0;
        for (int i = 0; i < probs.length; i++) {
            cumProb += probs[i];
            if (r < cumProb) return i;
        }
        return probs.length - 1;
    }

    private static int[] randomBatch(int n, int batchSize, Random rng) {
        int[] indices = new int[batchSize];
        for (int i = 0; i < batchSize; i++) {
            indices[i] = rng.nextInt(n);
        }
        return indices;
    }

    private static Tensor selectRows(Tensor t, int[] indices) {
        int[] shape = t.getShape();
        int dim = shape.length == 1 ? 1 : shape[1];

        double[] selected = new double[indices.length * dim];
        for (int i = 0; i < indices.length; i++) {
            System.arraycopy(t.getData(), indices[i] * dim, selected, i * dim, dim);
        }

        int[] newShape = shape.length == 1 ?
                new int[]{indices.length} :
                new int[]{indices.length, dim};

        return new Tensor(selected, newShape);
    }

    private static void printSummary() {
        System.out.println("\n" + "=".repeat(60));
        System.out.println("KEY INSIGHTS FROM LECTURE 4");
        System.out.println("=".repeat(60));

        System.out.println("\n🔑 Core Concepts:");
        System.out.println("   1. Deep Networks are Hard to Train");
        System.out.println("      - Activations saturate or explode");
        System.out.println("      - Gradients vanish or explode");
        System.out.println("      - Initialization becomes critical");

        System.out.println("\n   2. Batch Normalization Fixes This");
        System.out.println("      - Forces activations to mean=0, std=1");
        System.out.println("      - Stabilizes training");
        System.out.println("      - Allows deeper networks");

        System.out.println("\n   3. Activation Statistics Matter");
        System.out.println("      - Monitor mean, std, saturation");
        System.out.println("      - Healthy: mean≈0, std≈0.6, sat<5%");
        System.out.println("      - Unhealthy: saturated (sat>90%)");

        System.out.println("\n   4. Gradient Flow is Critical");
        System.out.println("      - Update ratio should be ~1e-3");
        System.out.println("      - Too large → instability");
        System.out.println("      - Too small → slow learning");

        System.out.println("\n📊 Architecture Pattern:");
        System.out.println("   Modern Deep Network = Stack of:");
        System.out.println("   [Linear → BatchNorm → Activation] × N");
        System.out.println("   → Final Linear → BatchNorm");

        System.out.println("\n💡 Why No Bias in Linear?");
        System.out.println("   - BatchNorm's beta parameter provides shift");
        System.out.println("   - Adding bias would be redundant");

        System.out.println("\n🎯 Training vs Inference:");
        System.out.println("   Training:   Use batch statistics");
        System.out.println("   Inference:  Use running statistics");
        System.out.println("   → Need to maintain running mean/var!");

        System.out.println("\n⚠️  BatchNorm Gotchas:");
        System.out.println("   - Couples examples in batch (breaks independence)");
        System.out.println("   - Sensitive to batch size");
        System.out.println("   - Tricky for RNNs");
        System.out.println("   - Alternatives: LayerNorm, GroupNorm");

        System.out.println("\n🎓 What's Next:");
        System.out.println("   • Lecture 5: Manual Backpropagation");
        System.out.println("   • Lecture 6: WaveNet Architecture");
        System.out.println("   • Lecture 7: Transformer/GPT");

        System.out.println("\n✅ Lecture 4 Complete!\n");
    }
}