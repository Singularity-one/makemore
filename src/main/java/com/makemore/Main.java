package com.makemore;

import com.makemore.bigram.BigramLanguageModel;
import java.util.List;

/**
 * Main class demonstrating makemore bigram language modeling
 *
 * This implements Andrej Karpathy's makemore Lecture 2:
 * "The spelled-out intro to language modeling: building makemore"
 *
 * Two approaches are demonstrated:
 * 1. Counting-based: Simple bigram frequency counting
 * 2. Neural Network: Gradient descent with single-layer network
 */
public class Main {

    public static void main(String[] args) {
        try {
            System.out.println("╔══════════════════════════════════════════════════════════╗");
            System.out.println("║     Makemore: Bigram Character-Level Language Model     ║");
            System.out.println("║          Based on Andrej Karpathy's Lecture 2           ║");
            System.out.println("╚══════════════════════════════════════════════════════════╝\n");

            // Initialize model
            BigramLanguageModel model = new BigramLanguageModel();

            // Load data
            String dataPath = "names.txt";
            System.out.println("Loading data from: " + dataPath);
            model.loadData(dataPath);

            // 2. 建立 bigramCounts
            model.trainCounting();

            // Print some statistics
            model.printTopBigrams(30);

            // ===== APPROACH 1: COUNTING METHOD =====
            System.out.println("\n" + "=".repeat(60));
            System.out.println("APPROACH 1: COUNTING-BASED BIGRAM MODEL");
            System.out.println("=".repeat(60));

            model.trainCounting();

            System.out.println("\nSampling 10 names using counting method:");
            List<String> countingSamples = model.sampleCounting(10);
            for (int i = 0; i < countingSamples.size(); i++) {
                System.out.printf("%2d. %s\n", i + 1, countingSamples.get(i));
            }

            // ===== APPROACH 2: NEURAL NETWORK METHOD =====
            System.out.println("\n" + "=".repeat(60));
            System.out.println("APPROACH 2: NEURAL NETWORK BIGRAM MODEL");
            System.out.println("=".repeat(60));

            // Hyperparameters
            int numIterations = 100;
            double learningRate = 50.0;
            double regularization = 0.01;

            System.out.printf("\nHyperparameters:\n");
            System.out.printf("  - Iterations: %d\n", numIterations);
            System.out.printf("  - Learning Rate: %.1f\n", learningRate);
            System.out.printf("  - L2 Regularization: %.4f\n", regularization);

            model.trainNeuralNet(numIterations, learningRate, regularization);

            System.out.println("\nSampling 10 names using neural network:");
            List<String> nnSamples = model.sampleNeuralNet(10);
            for (int i = 0; i < nnSamples.size(); i++) {
                System.out.printf("%2d. %s\n", i + 1, nnSamples.get(i));
            }

            // ===== COMPARISON =====
            System.out.println("\n" + "=".repeat(60));
            System.out.println("COMPARISON & INSIGHTS");
            System.out.println("=".repeat(60));

            System.out.println("\n🔍 Key Observations:");
            System.out.println("   1. Both methods learn character-level patterns");
            System.out.println("   2. Counting is exact but needs smoothing for unseen bigrams");
            System.out.println("   3. Neural network learns distributed representations");
            System.out.println("   4. Neural network supports gradient-based optimization");
            System.out.println("   5. Neural network generalizes better with regularization");

            System.out.println("\n💡 What's Next:");
            System.out.println("   • Lecture 3: MLP with hidden layers and embeddings");
            System.out.println("   • Lecture 4: Batch normalization and activation analysis");
            System.out.println("   • Lecture 5: Manual backpropagation (Backprop Ninja)");
            System.out.println("   • Lecture 6: WaveNet with hierarchical structure");
            System.out.println("   • Lecture 7: Building GPT from scratch");

            System.out.println("\n✅ Lecture 2 Complete!\n");

        } catch (Exception e) {
            System.err.println("Error: " + e.getMessage());
            e.printStackTrace();
        }
    }
}