package com.makemore.layers;

import com.makemore.mlp.Tensor;
import java.util.List;

/**
 * Base interface for all neural network layers
 * Inspired by PyTorch's nn.Module
 */
public interface Layer {

    /**
     * Forward pass through the layer
     * @param x Input tensor
     * @return Output tensor
     */
    Tensor forward(Tensor x);

    /**
     * Get all trainable parameters in this layer
     * @return List of parameter tensors
     */
    List<Tensor> parameters();

    /**
     * Set training mode
     * @param training true for training mode, false for evaluation
     */
    default void setTraining(boolean training) {
        // Default: do nothing (layers like Linear don't need this)
    }

    /**
     * Get the output tensor from last forward pass
     * Used for activation/gradient analysis
     */
    default Tensor getOutput() {
        return null;
    }
}