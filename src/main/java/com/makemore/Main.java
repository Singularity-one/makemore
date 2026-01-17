package com.makemore;

import com.makemore.mlp.MLPLanguageModel;
import java.util.List;

/**
 * Main class for MLP Language Model
 *
 * Implements Andrej Karpathy's makemore Lecture 3:
 * "Building makemore Part 2: MLP"
 */
public class Main {

    public static void main(String[] args) {
        try {
            System.out.println("╔════════════════════════════════════════════════════════════╗");
            System.out.println("║   Makemore Part 2: Multi-Layer Perceptron (MLP)          ║");
            System.out.println("║          Character-Level Language Model                   ║");
            System.out.println("║              Based on Karpathy's Lecture 3                ║");
            System.out.println("╚════════════════════════════════════════════════════════════╝\n");

            // Hyperparameters
            int blockSize = 3;        // Context length (how many chars to predict next one)
            int embeddingDim = 10;    // Embedding dimension
            int hiddenSize = 200;     // Hidden layer size

            // Training hyperparameters
            int numIterations = 200000;
            double learningRate = 0.1;
            int batchSize = 32;

            System.out.println("=== Model Configuration ===");
            System.out.println("Block size (context): " + blockSize);
            System.out.println("Embedding dimension: " + embeddingDim);
            System.out.println("Hidden layer size: " + hiddenSize);

            // Create model
            MLPLanguageModel model = new MLPLanguageModel(blockSize, embeddingDim, hiddenSize);

            // Load data
            String dataPath = "names.txt";
            System.out.println("\n=== Loading Data ===");
            model.loadData(dataPath);

            // Build dataset
            model.buildDataset();

            // Initialize parameters
            model.initializeParameters();

            // Train
            model.train(numIterations, learningRate, batchSize);

            // Sample names
            System.out.println("\n=== Sampling 20 Names ===");
            List<String> samples = model.sample(20);
            for (int i = 0; i < samples.size(); i++) {
                System.out.printf("%2d. %s\n", i + 1, samples.get(i));
            }

            // Summary
            System.out.println("\n" + "=".repeat(60));
            System.out.println("KEY INSIGHTS FROM LECTURE 3");
            System.out.println("=".repeat(60));

            System.out.println("\n🔑 Core Concepts Learned:");
            System.out.println("   1. Embeddings: Characters → Dense Vectors");
            System.out.println("      - Transforms discrete chars into continuous space");
            System.out.println("      - Learns semantic relationships");

            System.out.println("\n   2. Context Window:");
            System.out.println("      - Uses previous " + blockSize + " characters");
            System.out.println("      - Sliding window over text");

            System.out.println("\n   3. Hidden Layer:");
            System.out.println("      - Non-linear transformation (tanh)");
            System.out.println("      - Learns complex patterns");

            System.out.println("\n   4. Train/Dev/Test Split:");
            System.out.println("      - Training: 80% - Learn parameters");
            System.out.println("      - Dev: 10% - Tune hyperparameters");
            System.out.println("      - Test: 10% - Final evaluation");

            System.out.println("\n   5. Mini-Batch Training:");
            System.out.println("      - Batch size: " + batchSize);
            System.out.println("      - Faster training, better generalization");

            System.out.println("\n📊 Architecture Diagram:");
            System.out.println("   Input (context of " + blockSize + " chars)");
            System.out.println("        ↓");
            System.out.println("   Embedding Layer (" + embeddingDim + "D vectors)");
            System.out.println("        ↓");
            System.out.println("   Flatten (" + (blockSize * embeddingDim) + " dims)");
            System.out.println("        ↓");
            System.out.println("   Hidden Layer (" + hiddenSize + " neurons + tanh)");
            System.out.println("        ↓");
            System.out.println("   Output Layer (" + model.getVocabSize() + " logits)");
            System.out.println("        ↓");
            System.out.println("   Softmax → Probability Distribution");

            System.out.println("\n💡 Improvements over Lecture 2 (Bigrams):");
            System.out.println("   ✓ Larger context (3 chars vs 1 char)");
            System.out.println("   ✓ Learned embeddings (vs one-hot)");
            System.out.println("   ✓ Hidden layer (non-linear combinations)");
            System.out.println("   ✓ Much better quality names!");

            System.out.println("\n🎯 What's Next:");
            System.out.println("   • Lecture 4: BatchNorm & Gradient Analysis");
            System.out.println("   • Lecture 5: Manual Backpropagation");
            System.out.println("   • Lecture 6: WaveNet Architecture");
            System.out.println("   • Lecture 7: Transformer/GPT");

            System.out.println("\n✅ Lecture 3 Complete!\n");

        } catch (Exception e) {
            System.err.println("Error: " + e.getMessage());
            e.printStackTrace();
        }
    }
}