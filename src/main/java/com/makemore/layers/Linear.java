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

        // Initialize weights with proper scaling
        // Kaiming initialization: std = sqrt(2 / fan_in) for ReLU
        // For Tanh, we use std = sqrt(1 / fan_in) or just randn
        weight = Tensor.randn(rng, inFeatures, outFeatures);
        weight.requiresGrad(true);

        if (bias) {
            this.bias = Tensor.zeros(outFeatures);
            this.bias.requiresGrad(true);
        } else {
            this.bias = null;
        }
    }

    @Override
    public Tensor forward(Tensor x) {
        // x: (batch, inFeatures)
        // weight: (inFeatures, outFeatures)
        // out: (batch, outFeatures)

        out = x.matmul(weight);

        if (bias != null) {
            out = out.add(bias);
        }

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