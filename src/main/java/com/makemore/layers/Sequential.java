package com.makemore.layers;

import com.makemore.mlp.Tensor;
import java.util.*;

/**
 * Sequential Container
 *
 * Chains multiple layers together, applying them in sequence.
 *
 * Equivalent to PyTorch's nn.Sequential
 *
 * Example:
 *   Sequential seq = new Sequential(
 *       new Linear(10, 20, false, rng),
 *       new BatchNorm1d(20),
 *       new TanhLayer()
 *   );
 *
 *   Tensor out = seq.forward(input);
 *
 * Based on Karpathy's makemore Part 5 (WaveNet)
 */
public class Sequential implements Layer {

    private List<Layer> layers;
    private Tensor out;
    private boolean training;

    /**
     * Initialize Sequential with a list of layers
     */
    public Sequential(List<Layer> layers) {
        if (layers == null || layers.isEmpty()) {
            throw new IllegalArgumentException("Sequential requires at least one layer");
        }
        this.layers = new ArrayList<>(layers);
        this.training = true;
    }

    /**
     * Initialize Sequential with varargs
     */
    public Sequential(Layer... layers) {
        this(Arrays.asList(layers));
    }

    @Override
    public Tensor forward(Tensor x) {
        out = x;

        // Apply each layer in sequence
        for (Layer layer : layers) {
            out = layer.forward(out);
        }

        return out;
    }

    @Override
    public List<Tensor> parameters() {
        List<Tensor> allParams = new ArrayList<>();

        for (Layer layer : layers) {
            allParams.addAll(layer.parameters());
        }

        return allParams;
    }

    @Override
    public void setTraining(boolean training) {
        this.training = training;

        // Propagate to all layers
        for (Layer layer : layers) {
            layer.setTraining(training);
        }
    }

    @Override
    public Tensor getOutput() {
        return out;
    }

    /**
     * Get a specific layer by index
     */
    public Layer getLayer(int index) {
        return layers.get(index);
    }

    /**
     * Get all layers
     */
    public List<Layer> getLayers() {
        return Collections.unmodifiableList(layers);
    }

    /**
     * Get number of layers
     */
    public int size() {
        return layers.size();
    }

    /**
     * Add a layer to the end
     */
    public void add(Layer layer) {
        layers.add(layer);
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder("Sequential(\n");
        for (int i = 0; i < layers.size(); i++) {
            sb.append("  (").append(i).append("): ");
            sb.append(layers.get(i).getClass().getSimpleName());
            sb.append("\n");
        }
        sb.append(")");
        return sb.toString();
    }
}