package com.makemore;

import com.makemore.layers.*;
import com.makemore.mlp.Tensor;
import java.util.*;

/**
 * Deep Multi-Layer Perceptron with optional Batch Normalization
 *
 * Architecture (with BatchNorm):
 *   Embedding → Flatten
 *   → [Linear → BatchNorm → Tanh] × 5
 *   → Linear → BatchNorm
 *   → Softmax
 *
 * Architecture (without BatchNorm):
 *   Embedding → Flatten
 *   → [Linear → Tanh] × 5
 *   → Linear
 *   → Softmax
 */
public class DeepMLP {

    private Tensor C;  // Embedding table
    private List<Layer> layers;
    private List<Tensor> allParameters;

    private int vocabSize;
    private int embeddingDim;
    private int blockSize;
    private int hiddenSize;
    private boolean useBatchNorm;

    /**
     * Build deep MLP
     *
     * @param vocabSize Number of characters
     * @param embeddingDim Embedding dimension
     * @param blockSize Context length
     * @param hiddenSize Hidden layer size
     * @param useBatchNorm Whether to use Batch Normalization
     */
    public DeepMLP(int vocabSize, int embeddingDim, int blockSize,
                   int hiddenSize, boolean useBatchNorm) {

        this.vocabSize = vocabSize;
        this.embeddingDim = embeddingDim;
        this.blockSize = blockSize;
        this.hiddenSize = hiddenSize;
        this.useBatchNorm = useBatchNorm;

        Random rng = new Random(2147483647L);

        // Embedding table
        C = Tensor.randn(rng, vocabSize, embeddingDim);
        C.requiresGrad(true);

        // Build layers
        layers = new ArrayList<>();
        int inputSize = blockSize * embeddingDim;

        if (useBatchNorm) {
            // With BatchNorm: Linear (no bias) → BatchNorm → Tanh

            // First layer: input → hidden
            layers.add(new Linear(inputSize, hiddenSize, false, rng));
            layers.add(new BatchNorm1d(hiddenSize));
            layers.add(new TanhLayer());

            // Hidden layers: hidden → hidden (repeat 4 times)
            for (int i = 0; i < 4; i++) {
                layers.add(new Linear(hiddenSize, hiddenSize, false, rng));
                layers.add(new BatchNorm1d(hiddenSize));
                layers.add(new TanhLayer());
            }

            // Output layer: hidden → vocab
            layers.add(new Linear(hiddenSize, vocabSize, false, rng));
            layers.add(new BatchNorm1d(vocabSize));

        } else {
            // Without BatchNorm: Linear (with bias) → Tanh

            // First layer
            layers.add(new Linear(inputSize, hiddenSize, true, rng));
            layers.add(new TanhLayer());

            // Hidden layers (repeat 4 times)
            for (int i = 0; i < 4; i++) {
                layers.add(new Linear(hiddenSize, hiddenSize, true, rng));
                layers.add(new TanhLayer());
            }

            // Output layer
            layers.add(new Linear(hiddenSize, vocabSize, true, rng));
        }

        // Collect all parameters
        allParameters = new ArrayList<>();
        allParameters.add(C);
        for (Layer layer : layers) {
            allParameters.addAll(layer.parameters());
        }

        System.out.println("Built " + (useBatchNorm ? "WITH" : "WITHOUT") + " BatchNorm");
        System.out.println("Layers: " + layers.size());
    }

    /**
     * Forward pass
     */
    public Tensor forward(Tensor X) {
        // X: (batch, blockSize)

        // Embedding
        Tensor emb = C.index(X);  // (batch, blockSize, embeddingDim)

        // Flatten
        int batchSize = X.getShape()[0];
        Tensor x = emb.view(batchSize, blockSize * embeddingDim);

        // Through all layers
        for (Layer layer : layers) {
            x = layer.forward(x);
        }

        return x;  // logits
    }

    /**
     * Initialize last layer to be less confident
     * Multiply last BatchNorm's gamma by factor
     */
    public void initializeLastLayer(double factor) {
        if (useBatchNorm) {
            // Find last BatchNorm layer
            for (int i = layers.size() - 1; i >= 0; i--) {
                if (layers.get(i) instanceof BatchNorm1d) {
                    BatchNorm1d bn = (BatchNorm1d) layers.get(i);
                    Tensor gamma = bn.getGamma();
                    double[] gammaData = gamma.getData();
                    for (int j = 0; j < gammaData.length; j++) {
                        gammaData[j] *= factor;
                    }
                    System.out.println("✓ Initialized last layer gamma *= " + factor);
                    break;
                }
            }
        } else {
            // Find last Linear layer and scale weights
            for (int i = layers.size() - 1; i >= 0; i--) {
                if (layers.get(i) instanceof Linear) {
                    Linear linear = (Linear) layers.get(i);
                    Tensor weight = linear.getWeight();
                    double[] weightData = weight.getData();
                    for (int j = 0; j < weightData.length; j++) {
                        weightData[j] *= factor;
                    }
                    System.out.println("✓ Initialized last layer weight *= " + factor);
                    break;
                }
            }
        }
    }

    /**
     * Set training mode for all layers
     */
    public void setTrainMode() {
        for (Layer layer : layers) {
            layer.setTraining(true);
        }
    }

    /**
     * Set evaluation mode for all layers
     */
    public void setEvalMode() {
        for (Layer layer : layers) {
            layer.setTraining(false);
        }
    }

    /**
     * Zero all gradients
     */
    public void zeroGrad() {
        for (Tensor p : allParameters) {
            p.zeroGrad();
        }
    }

    /**
     * Update parameters with SGD
     */
    public void updateParameters(double learningRate) {
        for (Tensor p : allParameters) {
            double[] data = p.getData();
            double[] grad = p.getGrad();
            for (int i = 0; i < data.length; i++) {
                data[i] -= learningRate * grad[i];
            }
        }
    }

    /**
     * Get number of parameters
     */
    public int numParameters() {
        int total = 0;
        for (Tensor p : allParameters) {
            total += p.getSize();
        }
        return total;
    }

    /**
     * Get all layers (for diagnostics)
     */
    public List<Layer> getLayers() {
        return layers;
    }

    public List<Tensor> getAllParameters() {
        return allParameters;
    }
}