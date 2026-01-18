package com.makemore.layers;

import com.makemore.mlp.Tensor;
import java.util.*;

/**
 * FlattenConsecutive Layer
 *
 * Flattens consecutive elements along the sequence dimension.
 * This is the KEY innovation in WaveNet's hierarchical architecture!
 *
 * Input:  (batch, seq_len, embedding_dim)
 * Output: (batch, seq_len/n, embedding_dim * n)
 *
 * Example with n=2:
 *   Input:  (32, 8, 10)  - 8 tokens, each 10-dim
 *   Output: (32, 4, 20)  - 4 pairs, each 20-dim (2 tokens concatenated)
 *
 * This creates a hierarchical structure where:
 * - Layer 1: looks at individual characters
 * - Layer 2: looks at pairs of characters
 * - Layer 3: looks at groups of 4 characters
 * - Layer 4: looks at groups of 8 characters
 *
 * Based on Karpathy's makemore Part 5 (WaveNet)
 */
public class FlattenConsecutive implements Layer {

    private int n;  // Number of consecutive elements to flatten
    private Tensor out;

    /**
     * Initialize FlattenConsecutive layer
     *
     * @param n Number of consecutive elements to group together
     */
    public FlattenConsecutive(int n) {
        if (n <= 0) {
            throw new IllegalArgumentException("n must be positive, got: " + n);
        }
        this.n = n;
    }

    @Override
    public Tensor forward(Tensor x) {
        // x shape: (batch, seq_len, embedding_dim)
        int[] xShape = x.getShape();

        if (xShape.length != 3) {
            throw new IllegalArgumentException(
                    "FlattenConsecutive expects 3D input (batch, seq_len, emb_dim), got "
                            + xShape.length + "D");
        }

        int batchSize = xShape[0];
        int seqLen = xShape[1];
        int embDim = xShape[2];

        if (seqLen % n != 0) {
            throw new IllegalArgumentException(
                    "Sequence length " + seqLen + " must be divisible by n=" + n);
        }

        int newSeqLen = seqLen / n;
        int newEmbDim = embDim * n;

        // Reshape: (batch, seq_len, emb_dim) → (batch, new_seq_len, emb_dim * n)
        // This groups n consecutive tokens together

        double[] xData = x.getData();
        double[] result = new double[batchSize * newSeqLen * newEmbDim];

        for (int b = 0; b < batchSize; b++) {
            for (int newPos = 0; newPos < newSeqLen; newPos++) {
                // For each new position, concatenate n consecutive old positions
                for (int i = 0; i < n; i++) {
                    int oldPos = newPos * n + i;

                    // Copy embedding
                    int srcOffset = (b * seqLen + oldPos) * embDim;
                    int dstOffset = (b * newSeqLen + newPos) * newEmbDim + i * embDim;

                    System.arraycopy(xData, srcOffset, result, dstOffset, embDim);
                }
            }
        }

        out = new Tensor(result, new int[]{batchSize, newSeqLen, newEmbDim},
                x.isRequiresGrad());

        if (x.isRequiresGrad()) {
            setupBackward(x, batchSize, seqLen, embDim, newSeqLen, newEmbDim);
        }

        return out;
    }

    /**
     * Setup backward pass
     */
    private void setupBackward(Tensor x, int batchSize, int seqLen, int embDim,
                               int newSeqLen, int newEmbDim) {
        out.prev.add(x);
        out.op = "flatten_consecutive_" + n;

        out.backward = (v) -> {
            if (x.isRequiresGrad()) {
                double[] outGrad = out.getGrad();
                double[] xGrad = x.getGrad();

                // Scatter gradients back
                for (int b = 0; b < batchSize; b++) {
                    for (int newPos = 0; newPos < newSeqLen; newPos++) {
                        for (int i = 0; i < n; i++) {
                            int oldPos = newPos * n + i;

                            int srcOffset = (b * newSeqLen + newPos) * newEmbDim + i * embDim;
                            int dstOffset = (b * seqLen + oldPos) * embDim;

                            for (int d = 0; d < embDim; d++) {
                                xGrad[dstOffset + d] += outGrad[srcOffset + d];
                            }
                        }
                    }
                }
            }
            return null;
        };
    }

    @Override
    public List<Tensor> parameters() {
        return Collections.emptyList();  // No learnable parameters
    }

    @Override
    public Tensor getOutput() {
        return out;
    }

    public int getN() {
        return n;
    }

    @Override
    public String toString() {
        return "FlattenConsecutive(n=" + n + ")";
    }
}