package com.makemore.layers;

import com.makemore.mlp.Tensor;
import java.util.*;

/**
 * Tanh activation layer
 *
 * Applies tanh(x) element-wise
 */
public class TanhLayer implements Layer {

    private Tensor out;

    @Override
    public Tensor forward(Tensor x) {
        out = x.tanh();
        return out;
    }

    @Override
    public List<Tensor> parameters() {
        return Collections.emptyList();  // No parameters
    }

    @Override
    public Tensor getOutput() {
        return out;
    }
}