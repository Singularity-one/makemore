package com.makemore.layers;

import com.makemore.mlp.Tensor;
import java.util.*;

/**
 * Flatten Layer
 *
 * Flattens input tensor to 2D by keeping the first dimension (batch)
 * and flattening all other dimensions.
 *
 * Input:  (batch, d1, d2, ..., dn)
 * Output: (batch, d1*d2*...*dn)
 *
 * Common use case in WaveNet:
 * Input:  (batch, 1, hidden) after final FlattenConsecutive
 * Output: (batch, hidden)
 */
public class FlattenLayer implements Layer {

    private Tensor out;
    private int[] inputShape;

    @Override
    public Tensor forward(Tensor x) {
        inputShape = x.getShape();

        if (inputShape.length < 2) {
            throw new IllegalArgumentException(
                    "FlattenLayer expects at least 2D input, got " + inputShape.length + "D");
        }

        // Keep batch dimension, flatten everything else
        int batchSize = inputShape[0];
        int flattenedSize = 1;
        for (int i = 1; i < inputShape.length; i++) {
            flattenedSize *= inputShape[i];
        }

        // Use view to reshape (doesn't copy data)
        out = x.view(batchSize, flattenedSize);

        return out;
    }

    @Override
    public List<Tensor> parameters() {
        return Collections.emptyList();  // No learnable parameters
    }

    @Override
    public Tensor getOutput() {
        return out;
    }

    @Override
    public String toString() {
        return "FlattenLayer()";
    }
}