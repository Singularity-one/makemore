package com.makemore.mlp;

import com.makemore.Main;

import java.io.*;
import java.nio.file.*;
import java.util.*;

/**
 * Multi-Layer Perceptron (MLP) Character-Level Language Model
 *
 * Based on Andrej Karpathy's makemore Lecture 3
 *
 * FIXED: Cross-entropy loss gradient propagation
 */
public class MLPLanguageModel {

    // Hyperparameters
    private int blockSize;      // Context length
    private int embeddingDim;   // Embedding dimension
    private int hiddenSize;     // Hidden layer size
    private int vocabSize;

    // Data
    private List<String> words;
    private Map<Character, Integer> charToIdx;
    private Map<Integer, Character> idxToChar;

    // Model parameters
    private Tensor C;   // Embedding table (vocabSize x embeddingDim)
    private Tensor W1;  // Hidden weights (blockSize*embeddingDim x hiddenSize)
    private Tensor b1;  // Hidden bias (hiddenSize)
    private Tensor W2;  // Output weights (hiddenSize x vocabSize)
    private Tensor b2;  // Output bias (vocabSize)

    private List<Tensor> parameters;

    // Datasets
    private Tensor Xtr, Ytr;  // Training
    private Tensor Xdev, Ydev; // Development (validation)
    private Tensor Xte, Yte;   // Test

    private Random rng;

    public MLPLanguageModel(int blockSize, int embeddingDim, int hiddenSize) {
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
            // Initialize with '.' (padding)
            Arrays.fill(context, 0);

            // Process word + ending '.'
            String extWord = word + ".";
            for (char ch : extWord.toCharArray()) {
                int ix = charToIdx.get(ch);
                X.add(Arrays.copyOf(context, blockSize));
                Y.add(ix);

                // Shift context window
                System.arraycopy(context, 1, context, 0, blockSize - 1);
                context[blockSize - 1] = ix;
            }
        }

        // Convert to tensors
        int n = X.size();
        System.out.println("Total examples: " + n);

        double[] xData = new double[n * blockSize];
        double[] yData = new double[n];

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < blockSize; j++) {
                xData[i * blockSize + j] = X.get(i)[j];
            }
            yData[i] = Y.get(i);
        }

        Tensor allX = new Tensor(xData, new int[]{n, blockSize});
        Tensor allY = new Tensor(yData, new int[]{n});

        // Split: 80% train, 10% dev, 10% test
        int n1 = (int)(0.8 * n);
        int n2 = (int)(0.9 * n);

        Xtr = sliceTensor(allX, 0, n1);
        Ytr = sliceTensor(allY, 0, n1);

        Xdev = sliceTensor(allX, n1, n2);
        Ydev = sliceTensor(allY, n1, n2);

        Xte = sliceTensor(allX, n2, n);
        Yte = sliceTensor(allY, n2, n);

        System.out.println("Training examples: " + Xtr.getShape()[0]);
        System.out.println("Dev examples: " + Xdev.getShape()[0]);
        System.out.println("Test examples: " + Xte.getShape()[0]);
    }

    private Tensor sliceTensor(Tensor t, int start, int end) {
        int[] shape = t.getShape();
        int dim = shape.length == 1 ? 1 : shape[1];
        int count = end - start;

        double[] sliceData = new double[count * dim];
        System.arraycopy(t.getData(), start * dim, sliceData, 0, count * dim);

        int[] sliceShape = shape.length == 1 ? new int[]{count} : new int[]{count, dim};
        return new Tensor(sliceData, sliceShape);
    }

    /**
     * Initialize model parameters
     */
    public void initializeParameters() {
        System.out.println("\n=== Initializing Parameters ===");

        // Embedding table
        C = Tensor.randn(rng, vocabSize, embeddingDim).requiresGrad(true);

        // Hidden layer
        int inputSize = blockSize * embeddingDim;
        W1 = Tensor.randn(rng, inputSize, hiddenSize).requiresGrad(true);
        b1 = Tensor.randn(rng, hiddenSize).requiresGrad(true);

        // Output layer
        W2 = Tensor.randn(rng, hiddenSize, vocabSize).requiresGrad(true);
        b2 = Tensor.randn(rng, vocabSize).requiresGrad(true);

        parameters.clear();
        parameters.add(C);
        parameters.add(W1);
        parameters.add(b1);
        parameters.add(W2);
        parameters.add(b2);

        int totalParams = 0;
        for (Tensor p : parameters) {
            totalParams += p.getSize();
        }

        System.out.println("Embedding: " + vocabSize + " x " + embeddingDim);
        System.out.println("W1: " + inputSize + " x " + hiddenSize);
        System.out.println("b1: " + hiddenSize);
        System.out.println("W2: " + hiddenSize + " x " + vocabSize);
        System.out.println("b2: " + vocabSize);
        System.out.println("Total parameters: " + totalParams);
    }

    /**
     * Forward pass
     */
    public Tensor forward(Tensor X) {
        // X shape: (batch, blockSize)

        // Embedding lookup: (batch, blockSize, embeddingDim)
        Tensor emb = C.index(X);

        // Flatten: (batch, blockSize * embeddingDim)
        int batchSize = X.getShape()[0];
        Tensor embFlat = emb.view(batchSize, blockSize * embeddingDim);

        // Hidden layer with tanh
        Tensor h = embFlat.matmul(W1).add(b1).tanh();

        // Output layer
        Tensor logits = h.matmul(W2).add(b2);

        return logits;
    }

    /**
     * FIXED: Calculate cross-entropy loss with proper gradient flow
     */
    public Tensor crossEntropyLoss(Tensor logits, Tensor targets) {
// Softmax
        Tensor exp = logits.exp();
        Tensor sumExp = exp.sum(1, true);
        Tensor probs = exp.div(sumExp);

        // ⭐ 使用 gather 保持梯度連接!
        Tensor selected = probs.gather(targets);

        // -log(p).mean()
        Tensor loss = selected.log().neg().mean();

        return loss;
    }

    /**
     * Train the model
     */
    public void train(int numIterations, double learningRate, int batchSize) {
        System.out.println("\n=== Training ===");
        System.out.println("Iterations: " + numIterations);
        System.out.println("Learning rate: " + learningRate);
        System.out.println("Batch size: " + batchSize);

        int nTrain = Xtr.getShape()[0];

        for (int iter = 0; iter < numIterations; iter++) {
            // Mini-batch
            int[] batchIndices = randomBatch(nTrain, batchSize);
            Tensor Xb = selectRows(Xtr, batchIndices);
            Tensor Yb = selectRows(Ytr, batchIndices);

            // Forward pass
            Tensor logits = forward(Xb);
            Tensor loss = crossEntropyLoss(logits, Yb);

            // Backward pass
            zeroGrad();
            loss.backward();

            // Debug: Check gradient norm on first iteration
            if (iter == 0) {
                double gradNorm = 0.0;
                for (Tensor p : parameters) {
                    for (double g : p.getGrad()) {
                        gradNorm += g * g;
                    }
                }
                System.out.printf("Initial gradient norm: %.6f\n", Math.sqrt(gradNorm));
            }

            // Update parameters
            for (Tensor p : parameters) {
                double[] data = p.getData();
                double[] grad = p.getGrad();
                for (int i = 0; i < data.length; i++) {
                    data[i] -= learningRate * grad[i];
                }
            }

            // Evaluate periodically
            if (iter % 10000 == 0 || iter == numIterations - 1) {
                double trainLoss = evaluate(Xtr, Ytr);
                double devLoss = evaluate(Xdev, Ydev);
                System.out.printf("Iter %d: loss=%.4f, train=%.4f, dev=%.4f\n",
                        iter, loss.item(), trainLoss, devLoss);
            }

            // Learning rate decay at 100k iterations
            if (iter == 100000) {
                learningRate = 0.01;
                System.out.println("Learning rate decayed to: " + learningRate);
            }
        }

        System.out.println("\n=== Final Evaluation ===");
        double trainLoss = evaluate(Xtr, Ytr);
        double devLoss = evaluate(Xdev, Ydev);
        double testLoss = evaluate(Xte, Yte);
        System.out.printf("Train loss: %.4f\n", trainLoss);
        System.out.printf("Dev loss: %.4f\n", devLoss);
        System.out.printf("Test loss: %.4f\n", testLoss);
    }

    /**
     * Evaluate loss on dataset without gradients
     */
    public double evaluate(Tensor X, Tensor Y) {
        // Forward pass
        Tensor logits = forward(X);

        // Calculate loss manually
        Tensor counts = logits.exp();
        Tensor sumCounts = counts.sum(1, true);
        Tensor probs = counts.div(sumCounts);

        int batchSize = X.getShape()[0];
        double[] probData = probs.getData();
        double[] targetData = Y.getData();

        double totalLoss = 0.0;
        for (int i = 0; i < batchSize; i++) {
            int target = (int) targetData[i];
            double prob = probData[i * vocabSize + target];
            totalLoss += -Math.log(Math.max(prob, 1e-10));
        }

        return totalLoss / batchSize;
    }

    /**
     * Sample names from the model
     */
    public List<String> sample(int numSamples) {
        List<String> samples = new ArrayList<>();
        Random sampleRng = new Random(2147483647L + 10);

        for (int s = 0; s < numSamples; s++) {
            List<Integer> out = new ArrayList<>();
            int[] context = new int[blockSize];

            while (true) {
                // Create tensor for context
                double[] contextData = new double[blockSize];
                for (int i = 0; i < blockSize; i++) {
                    contextData[i] = context[i];
                }
                Tensor contextTensor = new Tensor(contextData, new int[]{1, blockSize});

                // Forward pass
                Tensor logits = forward(contextTensor);

                // Softmax
                Tensor counts = logits.exp();
                Tensor sumCounts = counts.sum(1, true);
                Tensor probs = counts.div(sumCounts);

                // Sample from distribution
                double[] probData = probs.getData();
                int nextIdx = sampleMultinomial(probData, sampleRng);

                // Shift context
                System.arraycopy(context, 1, context, 0, blockSize - 1);
                context[blockSize - 1] = nextIdx;

                out.add(nextIdx);

                if (nextIdx == 0 || out.size() > 20) {
                    break;
                }
            }

            // Convert to string
            StringBuilder name = new StringBuilder();
            for (int idx : out) {
                if (idx == 0) break;
                name.append(idxToChar.get(idx));
            }
            samples.add(name.toString());
        }

        return samples;
    }

    private int sampleMultinomial(double[] probs, Random rng) {
        double r = rng.nextDouble();
        double cumProb = 0.0;
        for (int i = 0; i < probs.length; i++) {
            cumProb += probs[i];
            if (r < cumProb) {
                return i;
            }
        }
        return probs.length - 1;
    }

    private void zeroGrad() {
        for (Tensor p : parameters) {
            p.zeroGrad();
        }
    }

    private int[] randomBatch(int n, int batchSize) {
        int[] indices = new int[batchSize];
        for (int i = 0; i < batchSize; i++) {
            indices[i] = rng.nextInt(n);
        }
        return indices;
    }

    private Tensor selectRows(Tensor t, int[] indices) {
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

    public int getVocabSize() { return vocabSize; }
    public Map<Character, Integer> getCharToIdx() { return charToIdx; }
    public Map<Integer, Character> getIdxToChar() { return idxToChar; }


}