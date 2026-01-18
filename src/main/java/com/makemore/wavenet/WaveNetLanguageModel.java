package com.makemore.wavenet;

import com.makemore.Main;
import com.makemore.layers.*;
import com.makemore.mlp.Tensor;

import java.io.*;
import java.util.*;

/**
 * WaveNet-style Character-Level Language Model
 *
 * Hierarchical architecture that progressively fuses context information:
 * - Instead of flattening all characters at once
 * - Builds up context hierarchically using FlattenConsecutive layers
 *
 * Architecture example (block_size=8, emb_dim=24, hidden=128):
 *
 *   Input: 8 characters
 *   ↓
 *   Embedding(27, 24) → (batch, 8, 24)
 *   ↓
 *   FlattenConsecutive(2) → (batch, 4, 48)  // pairs of chars
 *   ↓
 *   Linear(48, 128) + BatchNorm + Tanh
 *   ↓
 *   FlattenConsecutive(2) → (batch, 2, 256)  // groups of 4
 *   ↓
 *   Linear(256, 128) + BatchNorm + Tanh
 *   ↓
 *   FlattenConsecutive(2) → (batch, 1, 256)  // groups of 8
 *   ↓
 *   Linear(256, 128) + BatchNorm + Tanh
 *   ↓
 *   Linear(128, 27) → logits
 *
 * Based on Karpathy's makemore Part 5 (WaveNet)
 * Paper: "WaveNet: A Generative Model for Raw Audio" (van den Oord et al., 2016)
 */
public class WaveNetLanguageModel {

    // Hyperparameters
    private int blockSize;      // Context length (must be power of 2)
    private int embeddingDim;   // Embedding dimension
    private int hiddenSize;     // Hidden layer size
    private int vocabSize;

    // Data
    private List<String> words;
    private Map<Character, Integer> charToIdx;
    private Map<Integer, Character> idxToChar;

    // Model
    private Sequential model;
    private List<Tensor> parameters;

    // Datasets
    private Tensor Xtr, Ytr;   // Training
    private Tensor Xdev, Ydev; // Validation
    private Tensor Xte, Yte;   // Test

    private Random rng;

    /**
     * Initialize WaveNet model
     *
     * @param blockSize Context length (MUST be power of 2, e.g., 8, 16, 32)
     * @param embeddingDim Embedding dimension
     * @param hiddenSize Hidden layer size
     */
    public WaveNetLanguageModel(int blockSize, int embeddingDim, int hiddenSize) {
        // Validate block_size is power of 2
        if (blockSize <= 0 || (blockSize & (blockSize - 1)) != 0) {
            throw new IllegalArgumentException(
                    "block_size must be a power of 2, got: " + blockSize);
        }

        this.blockSize = blockSize;
        this.embeddingDim = embeddingDim;
        this.hiddenSize = hiddenSize;
        this.charToIdx = new HashMap<>();
        this.idxToChar = new HashMap<>();
        this.rng = new Random(2147483647L);
        this.parameters = new ArrayList<>();
    }

    /**
     * Load words and build vocabulary
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
        for (String word : words) {
            for (char c : word.toCharArray()) {
                chars.add(c);
            }
        }

        // Create mappings (0 is reserved for '.')
        charToIdx.put('.', 0);
        idxToChar.put(0, '.');
        int idx = 1;
        for (Character c : chars) {
            charToIdx.put(c, idx);
            idxToChar.put(idx, c);
            idx++;
        }
        vocabSize = idx;

        System.out.println("Vocabulary size: " + vocabSize);
        System.out.println("Block size (context): " + blockSize);
    }

    /**
     * Build training dataset
     */
    public void buildDataset() {
        List<int[]> X = new ArrayList<>();
        List<Integer> Y = new ArrayList<>();

        for (String word : words) {
            int[] context = new int[blockSize];
            Arrays.fill(context, 0);  // Initialize with '.'

            for (char ch : word.toCharArray()) {
                int ix = charToIdx.get(ch);
                X.add(context.clone());
                Y.add(ix);

                // Shift context
                System.arraycopy(context, 1, context, 0, blockSize - 1);
                context[blockSize - 1] = ix;
            }

            // Predict end token
            X.add(context.clone());
            Y.add(0);
        }

        System.out.println("Dataset size: " + X.size() + " examples");

        // Convert to tensors
        double[] XData = new double[X.size() * blockSize];
        double[] YData = new double[Y.size()];

        for (int i = 0; i < X.size(); i++) {
            for (int j = 0; j < blockSize; j++) {
                XData[i * blockSize + j] = X.get(i)[j];
            }
            YData[i] = Y.get(i);
        }

        Tensor XAll = new Tensor(XData, new int[]{X.size(), blockSize});
        Tensor YAll = new Tensor(YData, new int[]{Y.size()});

        // Split dataset: 80% train, 10% dev, 10% test
        int n1 = (int) (0.8 * X.size());
        int n2 = (int) (0.9 * X.size());

        Xtr = sliceTensor(XAll, 0, n1);
        Ytr = sliceTensor(YAll, 0, n1);

        Xdev = sliceTensor(XAll, n1, n2);
        Ydev = sliceTensor(YAll, n1, n2);

        Xte = sliceTensor(XAll, n2, X.size());
        Yte = sliceTensor(YAll, n2, Y.size());

        System.out.println("Training: " + Xtr.getShape()[0] + " examples");
        System.out.println("Validation: " + Xdev.getShape()[0] + " examples");
        System.out.println("Test: " + Xte.getShape()[0] + " examples");
    }

    /**
     * Build hierarchical WaveNet model
     */
    public void buildModel() {
        List<Layer> layers = new ArrayList<>();

        // Step 1: Embedding layer
        // Input: (batch, block_size) - integers
        // Output: (batch, block_size, emb_dim)
        layers.add(new Embedding(vocabSize, embeddingDim, rng));

        // Calculate number of hierarchical levels
        int numLevels = (int) (Math.log(blockSize) / Math.log(2));
        int currentDim = embeddingDim;

        // Step 2: Hierarchical flattening + processing
        for (int level = 0; level < numLevels; level++) {
            layers.add(new FlattenConsecutive(2));
            currentDim *= 2;

            layers.add(new Linear(currentDim, hiddenSize, false, rng));
            // ❌ 移除這行
            // layers.add(new BatchNorm1d(hiddenSize));
            layers.add(new TanhLayer());

            currentDim = hiddenSize;
        }

        layers.add(new FlattenLayer());

        // Step 3: Final output layer
        layers.add(new Linear(hiddenSize, vocabSize, false, rng));

        // Create sequential model
        model = new Sequential(layers);

        // Collect all parameters
        parameters = model.parameters();

        System.out.println("\n=== Model Architecture ===");
        System.out.println(model);
        System.out.println("\nTotal parameters: " + parameters.size());
        System.out.println("Total parameter count: " +
                parameters.stream().mapToInt(Tensor::getSize).sum());
    }

    /**
     * Forward pass
     */
    public Tensor forward(Tensor X) {
        return model.forward(X);
    }

    /**
     * Compute loss
     */
    public double computeLoss(Tensor X, Tensor Y) {
        // Forward pass
        Tensor logits = forward(X);

        // Cross-entropy loss
        double loss = crossEntropyLoss(logits, Y);

        return loss;
    }

    /**
     * Cross-entropy loss
     */
    private double crossEntropyLoss(Tensor logits, Tensor Y) {
        // logits: (batch, vocab_size)
        // Y: (batch,) containing class indices

        int batchSize = logits.getShape()[0];

        // Get probabilities
        Tensor probs = logits.softmax(1);

        // Gather probabilities at correct indices
        Tensor selectedProbs = probs.gather(Y);

        // Negative log likelihood
        Tensor logProbs = selectedProbs.log();
        Tensor loss = logProbs.neg().mean();

        return loss.item();
    }

    /**
     * Generate a sample
     */
    public String sample(int maxLength) {
        model.setTraining(false);

        int[] context = new int[blockSize];
        Arrays.fill(context, 0);  // Start with '.'

        StringBuilder result = new StringBuilder();

        for (int i = 0; i < maxLength; i++) {
            // Convert context to tensor
            double[] contextData = new double[blockSize];
            for (int j = 0; j < blockSize; j++) {
                contextData[j] = context[j];
            }
            Tensor X = new Tensor(contextData, new int[]{1, blockSize});

            // Forward pass
            Tensor logits = forward(X);

            // Get probabilities
            Tensor probs = logits.softmax(1);

            // Sample from distribution
            int nextChar = sampleFromProbs(probs.getData());

            if (nextChar == 0) {
                break;  // End token
            }

            result.append(idxToChar.get(nextChar));

            // Update context
            System.arraycopy(context, 1, context, 0, blockSize - 1);
            context[blockSize - 1] = nextChar;
        }

        model.setTraining(true);
        return result.toString();
    }

    // ========================================================================
    // Helper Methods
    // ========================================================================

    private Tensor sliceTensor(Tensor t, int start, int end) {
        int[] shape = t.getShape();
        double[] data = t.getData();

        if (shape.length == 1) {
            // 1D tensor
            int newSize = end - start;
            double[] newData = new double[newSize];
            System.arraycopy(data, start, newData, 0, newSize);
            return new Tensor(newData, new int[]{newSize});

        } else if (shape.length == 2) {
            // 2D tensor
            int rows = end - start;
            int cols = shape[1];
            double[] newData = new double[rows * cols];
            System.arraycopy(data, start * cols, newData, 0, rows * cols);
            return new Tensor(newData, new int[]{rows, cols});
        }

        throw new UnsupportedOperationException("Slice only supports 1D and 2D tensors");
    }

    private int sampleFromProbs(double[] probs) {
        double r = rng.nextDouble();
        double cumulative = 0.0;

        for (int i = 0; i < probs.length; i++) {
            cumulative += probs[i];
            if (r < cumulative) {
                return i;
            }
        }

        return probs.length - 1;
    }

    // ========================================================================
    // Getters
    // ========================================================================

    public Sequential getModel() {
        return model;
    }

    public List<Tensor> getParameters() {
        return parameters;
    }

    public Tensor getXtr() {
        return Xtr;
    }

    public Tensor getYtr() {
        return Ytr;
    }

    public Tensor getXdev() {
        return Xdev;
    }

    public Tensor getYdev() {
        return Ydev;
    }

    public int getVocabSize() {
        return vocabSize;
    }

    public int getBlockSize() {
        return blockSize;
    }
}