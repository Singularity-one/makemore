package com.makemore.layers;

import com.makemore.mlp.Tensor;
import java.util.*;

/**
 * Linear (fully connected) layer: y = x @ W + b
 *
 * Equivalent to PyTorch's nn.Linear
 */
public class Linear implements Layer {

    private Tensor weight;  // (in_features, out_features)
    private Tensor bias;    // (out_features) or null
    private Tensor out;     // Last output (for diagnostics)

    private int inFeatures;
    private int outFeatures;
    private boolean useBias;

    /**
     * Initialize linear layer
     *
     * @param inFeatures Number of input features
     * @param outFeatures Number of output features
     * @param bias Whether to use bias (typically false with BatchNorm)
     * @param rng Random number generator
     */
    public Linear(int inFeatures, int outFeatures, boolean bias, Random rng) {
        this.inFeatures = inFeatures;
        this.outFeatures = outFeatures;
        this.useBias = bias;

        // Kaiming/He initialization for Tanh
        double scale = Math.sqrt(1.0 / inFeatures);

        weight = Tensor.randn(rng, inFeatures, outFeatures).mul(scale);
        weight.requiresGrad(true);

        if (bias) {
            this.bias = Tensor.zeros(outFeatures);
            this.bias.requiresGrad(true);
        } else {
            this.bias = null;
        }
    }

    /**
     *
     * 修正：支援任意維度輸入，只對最後一維做線性變換
     *
     * 支援的輸入形狀：
     * - 2D: (batch, in_features) → (batch, out_features)
     * - 3D: (batch, seq, in_features) → (batch, seq, out_features)
     * - 4D: (batch, d1, d2, in_features) → (batch, d1, d2, out_features)
     *
     * 這符合 PyTorch nn.Linear 的行為
     */
    @Override
    public Tensor forward(Tensor x) {
        int[] xShape = x.getShape();

        // Check last dimension matches in_features
        int lastDim = xShape[xShape.length - 1];
        if (lastDim != inFeatures) {
            throw new IllegalArgumentException(
                    "Expected last dimension to be " + inFeatures + ", got " + lastDim);
        }

        // Case 1: 2D input (batch, in_features) - original behavior
        if (xShape.length == 2) {
            out = x.matmul(weight);

            if (bias != null) {
                out = out.add(bias);
            }

            return out;
        }

        // Case 2: 3D or higher - reshape, matmul, reshape back
        // Example: (batch, seq, in_features) → (batch*seq, in_features) → matmul → (batch*seq, out_features) → (batch, seq, out_features)

        // Calculate total number of "rows" (everything except last dimension)
        int batchSize = 1;
        for (int i = 0; i < xShape.length - 1; i++) {
            batchSize *= xShape[i];
        }

        // Reshape to 2D: (batch_size, in_features)
        Tensor x2d = x.view(batchSize, inFeatures);

        // Apply linear transformation
        Tensor out2d = x2d.matmul(weight);

        if (bias != null) {
            out2d = out2d.add(bias);
        }

        // Reshape back to original shape (with out_features in last dimension)
        int[] outShape = new int[xShape.length];
        for (int i = 0; i < xShape.length - 1; i++) {
            outShape[i] = xShape[i];
        }
        outShape[xShape.length - 1] = outFeatures;

        out = out2d.view(outShape);

        return out;
    }

    @Override
    public List<Tensor> parameters() {
        List<Tensor> params = new ArrayList<>();
        params.add(weight);
        if (bias != null) {
            params.add(bias);
        }
        return params;
    }

    @Override
    public Tensor getOutput() {
        return out;
    }

    public Tensor getWeight() {
        return weight;
    }

    public Tensor getBias() {
        return bias;
    }
}