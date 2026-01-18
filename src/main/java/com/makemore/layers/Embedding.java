package com.makemore.layers;

import com.makemore.mlp.Tensor;
import java.util.*;

/**
 * Embedding Layer
 *
 * Maps discrete tokens (integers) to continuous vectors.
 *
 * Input: (batch, seq_len) - integers in range [0, num_embeddings)
 * Output: (batch, seq_len, embedding_dim)
 *
 * Equivalent to PyTorch's nn.Embedding
 *
 * Based on Karpathy's makemore Part 5 (WaveNet)
 */
public class Embedding implements Layer {

    private int numEmbeddings;   // Vocabulary size
    private int embeddingDim;    // Dimension of each embedding vector

    private Tensor weight;       // (num_embeddings, embedding_dim)
    private Tensor out;          // Last output

    /**
     * Initialize embedding layer
     *
     * @param numEmbeddings Size of the dictionary (vocabulary)
     * @param embeddingDim Dimension of embedding vectors
     * @param rng Random number generator
     */
    public Embedding(int numEmbeddings, int embeddingDim, Random rng) {
        this.numEmbeddings = numEmbeddings;
        this.embeddingDim = embeddingDim;

        // Initialize embedding table with random values
        // Using standard normal distribution
        this.weight = Tensor.randn(rng, numEmbeddings, embeddingDim);
        this.weight.requiresGrad(true);
    }

    @Override
    public Tensor forward(Tensor x) {
        // x shape: (batch, seq_len) containing integer indices
        // output shape: (batch, seq_len, embedding_dim)

        int[] xShape = x.getShape();

        if (xShape.length == 1) {
            // Input is 1D: (seq_len,)
            int seqLen = xShape[0];
            return lookupEmbeddings1D(x, seqLen);

        } else if (xShape.length == 2) {
            // Input is 2D: (batch, seq_len)
            int batchSize = xShape[0];
            int seqLen = xShape[1];
            return lookupEmbeddings2D(x, batchSize, seqLen);

        } else {
            throw new IllegalArgumentException(
                    "Embedding input must be 1D or 2D, got " + xShape.length + "D");
        }
    }

    /**
     * Lookup embeddings for 1D input
     */
    private Tensor lookupEmbeddings1D(Tensor x, int seqLen) {
        double[] xData = x.getData();
        double[] weightData = weight.getData();

        double[] result = new double[seqLen * embeddingDim];

        for (int i = 0; i < seqLen; i++) {
            int idx = (int) xData[i];

            if (idx < 0 || idx >= numEmbeddings) {
                throw new IllegalArgumentException(
                        "Index out of range: " + idx + " (vocab size: " + numEmbeddings + ")");
            }

            // Copy embedding vector
            System.arraycopy(weightData, idx * embeddingDim,
                    result, i * embeddingDim, embeddingDim);
        }

        out = new Tensor(result, new int[]{seqLen, embeddingDim}, weight.isRequiresGrad());
        setupBackward1D(x, seqLen);

        return out;
    }

    /**
     * Lookup embeddings for 2D input
     */
    private Tensor lookupEmbeddings2D(Tensor x, int batchSize, int seqLen) {
        double[] xData = x.getData();
        double[] weightData = weight.getData();

        double[] result = new double[batchSize * seqLen * embeddingDim];

        for (int b = 0; b < batchSize; b++) {
            for (int s = 0; s < seqLen; s++) {
                int idx = (int) xData[b * seqLen + s];

                if (idx < 0 || idx >= numEmbeddings) {
                    throw new IllegalArgumentException(
                            "Index out of range: " + idx + " (vocab size: " + numEmbeddings + ")");
                }

                // Copy embedding vector
                int outOffset = (b * seqLen + s) * embeddingDim;
                int weightOffset = idx * embeddingDim;

                System.arraycopy(weightData, weightOffset,
                        result, outOffset, embeddingDim);
            }
        }

        out = new Tensor(result, new int[]{batchSize, seqLen, embeddingDim},
                weight.isRequiresGrad());
        setupBackward2D(x, batchSize, seqLen);

        return out;
    }

    /**
     * Setup backward pass for 1D input
     */
    private void setupBackward1D(Tensor x, int seqLen) {
        if (!weight.isRequiresGrad()) {
            return;
        }

        out.prev.add(weight);
        out.prev.add(x);
        out.op = "embedding_1d";

        out.backward = (v) -> {
            if (weight.isRequiresGrad()) {
                double[] xData = x.getData();
                double[] outGrad = out.getGrad();
                double[] weightGrad = weight.getGrad();

                // Scatter gradients back to embedding table
                for (int i = 0; i < seqLen; i++) {
                    int idx = (int) xData[i];

                    for (int j = 0; j < embeddingDim; j++) {
                        weightGrad[idx * embeddingDim + j] +=
                                outGrad[i * embeddingDim + j];
                    }
                }
            }
            return null;
        };
    }

    /**
     * Setup backward pass for 2D input
     */
    private void setupBackward2D(Tensor x, int batchSize, int seqLen) {
        if (!weight.isRequiresGrad()) {
            return;
        }

        out.prev.add(weight);
        out.prev.add(x);
        out.op = "embedding_2d";

        out.backward = (v) -> {
            if (weight.isRequiresGrad()) {
                double[] xData = x.getData();
                double[] outGrad = out.getGrad();
                double[] weightGrad = weight.getGrad();

                // Scatter gradients back to embedding table
                for (int b = 0; b < batchSize; b++) {
                    for (int s = 0; s < seqLen; s++) {
                        int idx = (int) xData[b * seqLen + s];

                        for (int d = 0; d < embeddingDim; d++) {
                            int outIdx = (b * seqLen + s) * embeddingDim + d;
                            weightGrad[idx * embeddingDim + d] += outGrad[outIdx];
                        }
                    }
                }
            }
            return null;
        };
    }

    @Override
    public List<Tensor> parameters() {
        return Collections.singletonList(weight);
    }

    @Override
    public Tensor getOutput() {
        return out;
    }

    public Tensor getWeight() {
        return weight;
    }

    public int getNumEmbeddings() {
        return numEmbeddings;
    }

    public int getEmbeddingDim() {
        return embeddingDim;
    }
}