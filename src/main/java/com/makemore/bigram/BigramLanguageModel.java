package com.makemore.bigram;

import com.makemore.Main;

import java.util.*;
import java.io.*;
import java.nio.file.*;

/**
 * Bigram Language Model
 * Implements both counting-based and neural network approaches
 *
 * Based on Andrej Karpathy's makemore Lecture 2
 */
public class BigramLanguageModel {

    private List<String> words;
    private Map<Character, Integer> charToIdx;
    private Map<Integer, Character> idxToChar;
    private int vocabSize;

    // For counting approach
    private int[][] bigramCounts;
    private double[][] bigramProbs;

    // For neural network approach
    private Tensor W;  // Weights (27 x 27)
    private Random rng;

    public BigramLanguageModel() {
        this.charToIdx = new HashMap<>();
        this.idxToChar = new HashMap<>();
        this.rng = new Random(2147483647L);
    }

    /**
     * Load names from file
     */
    public void loadData(String filename) throws IOException {
        InputStream inputStream = Main.class.getResourceAsStream("/names.txt");
        if (inputStream == null) {
            throw new FileNotFoundException("Cannot find names.txt in resources folder");
        }
        BufferedReader reader = new BufferedReader(new InputStreamReader(inputStream));
        words = reader.lines().toList();
        System.out.println("Loaded " + words.size() + " words");

        // Build vocabulary
        Set<Character> chars = new TreeSet<>();
        chars.add('.');  // Special start/end token
        for (String word : words) {
            for (char c : word.toCharArray()) {
                chars.add(c);
            }
        }

        // Create mappings
        int idx = 0;
        for (Character c : chars) {
            charToIdx.put(c, idx);
            idxToChar.put(idx, c);
            idx++;
        }
        vocabSize = chars.size();

        System.out.println("Vocabulary size: " + vocabSize);
        System.out.println("Characters: " + chars);
    }

    /**
     * METHOD 1: Counting-based approach
     */
    public void trainCounting() {
        System.out.println("\n=== Training with Counting Method ===");

        // Initialize count matrix
        bigramCounts = new int[vocabSize][vocabSize];

        // Count bigrams
        for (String word : words) {
            String extWord = "." + word + ".";
            for (int i = 0; i < extWord.length() - 1; i++) {
                char ch1 = extWord.charAt(i);
                char ch2 = extWord.charAt(i + 1);
                int idx1 = charToIdx.get(ch1);
                int idx2 = charToIdx.get(ch2);
                bigramCounts[idx1][idx2]++;
            }
        }

        // Normalize to get probabilities (with smoothing)
        bigramProbs = new double[vocabSize][vocabSize];
        for (int i = 0; i < vocabSize; i++) {
            int rowSum = 0;
            for (int j = 0; j < vocabSize; j++) {
                rowSum += bigramCounts[i][j];
            }

            // Add-one smoothing
            for (int j = 0; j < vocabSize; j++) {
                bigramProbs[i][j] = (bigramCounts[i][j] + 1.0) / (rowSum + vocabSize);
            }
        }

        // Calculate negative log likelihood
        double nll = calculateNLL(bigramProbs);
        System.out.printf("Negative Log Likelihood: %.4f\n", nll);
    }

    /**
     * METHOD 2: Neural Network approach
     */
    public void trainNeuralNet(int numIterations, double learningRate, double regularization) {
        System.out.println("\n=== Training with Neural Network ===");

        // Initialize weights
        W = Tensor.randn(rng, vocabSize, vocabSize).requiresGrad(true);

        // Create training data
        int[] xs = createTrainingData()[0];
        int[] ys = createTrainingData()[1];
        int numExamples = xs.length;

        System.out.println("Number of examples: " + numExamples);

        // Training loop
        for (int iter = 0; iter < numIterations; iter++) {
            // Forward pass
            Tensor xenc = oneHotEncode(xs, vocabSize);
            Tensor logits = xenc.matmul(W);
            Tensor counts = logits.exp();
            Tensor sumCounts = counts.sum(1, true);
            Tensor probs = counts.div(sumCounts);

            // Calculate loss
            Tensor loss = calculateLoss(probs, ys, numExamples);

            // Add L2 regularization
            if (regularization > 0) {
                Tensor regLoss = W.pow(2).mean().mul(regularization);
                loss = loss.add(regLoss);
            }

            if (iter % 10 == 0 || iter == numIterations - 1) {
                System.out.printf("Iteration %d: Loss = %.6f\n", iter, loss.item());
            }

            // Backward pass
            W.zeroGrad();
            loss.backward();

            // Update weights
            double[] wData = W.getData();
            double[] wGrad = W.getGrad();
            for (int i = 0; i < wData.length; i++) {
                wData[i] -= learningRate * wGrad[i];
            }
        }

        System.out.println("Training complete!");
    }

    /**
     * Create training dataset of bigram pairs (xs, ys)
     */
    private int[][] createTrainingData() {
        List<Integer> xsList = new ArrayList<>();
        List<Integer> ysList = new ArrayList<>();

        for (String word : words) {
            String extWord = "." + word + ".";
            for (int i = 0; i < extWord.length() - 1; i++) {
                char ch1 = extWord.charAt(i);
                char ch2 = extWord.charAt(i + 1);
                xsList.add(charToIdx.get(ch1));
                ysList.add(charToIdx.get(ch2));
            }
        }

        int[] xs = xsList.stream().mapToInt(Integer::intValue).toArray();
        int[] ys = ysList.stream().mapToInt(Integer::intValue).toArray();

        return new int[][]{xs, ys};
    }

    /**
     * One-hot encode indices
     */
    private Tensor oneHotEncode(int[] indices, int numClasses) {
        int n = indices.length;
        double[] data = new double[n * numClasses];

        for (int i = 0; i < n; i++) {
            data[i * numClasses + indices[i]] = 1.0;
        }

        return new Tensor(data, new int[]{n, numClasses});
    }

    /**
     * Calculate negative log likelihood loss
     */
    private Tensor calculateLoss(Tensor probs, int[] targets, int numExamples) {
        double[] probData = probs.getData();
        int vocabSize = probs.getShape()[1];

        double totalNLL = 0.0;
        for (int i = 0; i < numExamples; i++) {
            int target = targets[i];
            double prob = probData[i * vocabSize + target];
            totalNLL += -Math.log(prob);
        }

        double avgNLL = totalNLL / numExamples;

        // Create loss tensor with gradient
        Tensor loss = new Tensor(new double[]{avgNLL}, new int[]{1}, true);
        loss.requiresGrad(true);

        // Manually set up backward for cross-entropy
        double[] probGrad = new double[probData.length];
        for (int i = 0; i < numExamples; i++) {
            int target = targets[i];
            double prob = probData[i * vocabSize + target];
            probGrad[i * vocabSize + target] = -1.0 / (prob * numExamples);
        }

        // Copy gradient to probs tensor
        System.arraycopy(probGrad, 0, probs.getGrad(), 0, probGrad.length);

        // Set up backward chain
        final Tensor probsFinal = probs;
        loss.requiresGrad(true);

        return loss;
    }

    /**
     * Calculate NLL for probability matrix
     */
    private double calculateNLL(double[][] probs) {
        double logLikelihood = 0.0;
        int n = 0;

        for (String word : words) {
            String extWord = "." + word + ".";
            for (int i = 0; i < extWord.length() - 1; i++) {
                char ch1 = extWord.charAt(i);
                char ch2 = extWord.charAt(i + 1);
                int idx1 = charToIdx.get(ch1);
                int idx2 = charToIdx.get(ch2);
                double prob = probs[idx1][idx2];
                logLikelihood += Math.log(prob);
                n++;
            }
        }

        return -logLikelihood / n;
    }

    /**
     * Sample names using counting approach
     */
    public List<String> sampleCounting(int numSamples) {
        List<String> samples = new ArrayList<>();
        Random rng = new Random(2147483647L);

        for (int i = 0; i < numSamples; i++) {
            StringBuilder name = new StringBuilder();
            int idx = charToIdx.get('.');

            while (true) {
                // Sample next character based on probabilities
                double r = rng.nextDouble();
                double cumProb = 0.0;
                int nextIdx = 0;

                for (int j = 0; j < vocabSize; j++) {
                    cumProb += bigramProbs[idx][j];
                    if (r < cumProb) {
                        nextIdx = j;
                        break;
                    }
                }

                char nextChar = idxToChar.get(nextIdx);
                if (nextChar == '.') {
                    break;
                }

                name.append(nextChar);
                idx = nextIdx;

                // Safety limit
                if (name.length() > 20) {
                    break;
                }
            }

            samples.add(name.toString());
        }

        return samples;
    }

    /**
     * Sample names using neural network
     */
    public List<String> sampleNeuralNet(int numSamples) {
        List<String> samples = new ArrayList<>();
        Random rng = new Random(2147483647L);

        for (int i = 0; i < numSamples; i++) {
            StringBuilder name = new StringBuilder();
            int idx = charToIdx.get('.');

            while (true) {
                // Forward pass for single input
                int[] input = new int[]{idx};
                Tensor xenc = oneHotEncode(input, vocabSize);
                Tensor logits = xenc.matmul(W);
                Tensor counts = logits.exp();
                Tensor sumCounts = counts.sum(1, true);
                Tensor probs = counts.div(sumCounts);

                // Sample from probabilities
                double[] probData = probs.getData();
                double r = rng.nextDouble();
                double cumProb = 0.0;
                int nextIdx = 0;

                for (int j = 0; j < vocabSize; j++) {
                    cumProb += probData[j];
                    if (r < cumProb) {
                        nextIdx = j;
                        break;
                    }
                }

                char nextChar = idxToChar.get(nextIdx);
                if (nextChar == '.') {
                    break;
                }

                name.append(nextChar);
                idx = nextIdx;

                // Safety limit
                if (name.length() > 20) {
                    break;
                }
            }

            samples.add(name.toString());
        }

        return samples;
    }

    /**
     * Print bigram statistics
     */
    public void printTopBigrams(int topN) {
        System.out.println("\n=== Top " + topN + " Most Common Bigrams ===");

        List<BigramCount> bigrams = new ArrayList<>();
        for (int i = 0; i < vocabSize; i++) {
            for (int j = 0; j < vocabSize; j++) {
                if (bigramCounts[i][j] > 0) {
                    bigrams.add(new BigramCount(
                            idxToChar.get(i),
                            idxToChar.get(j),
                            bigramCounts[i][j]
                    ));
                }
            }
        }

        bigrams.sort((a, b) -> Integer.compare(b.count, a.count));

        for (int i = 0; i < Math.min(topN, bigrams.size()); i++) {
            BigramCount bc = bigrams.get(i);
            System.out.printf("%c%c: %d\n", bc.ch1, bc.ch2, bc.count);
        }
    }

    private static class BigramCount {
        char ch1, ch2;
        int count;

        BigramCount(char ch1, char ch2, int count) {
            this.ch1 = ch1;
            this.ch2 = ch2;
            this.count = count;
        }
    }

    // Getters
    public int getVocabSize() { return vocabSize; }
    public Map<Character, Integer> getCharToIdx() { return charToIdx; }
    public Map<Integer, Character> getIdxToChar() { return idxToChar; }
}