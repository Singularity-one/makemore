package com.makemore.layers;

import com.makemore.mlp.Tensor;
import java.util.*;

/**
 * Batch Normalization layer (1D version)
 *
 * Normalizes activations to have mean=0, std=1, then applies learnable
 * scale (gamma) and shift (beta).
 *
 * During training: uses batch statistics
 * During inference: uses running statistics
 *
 * Based on: "Batch Normalization: Accelerating Deep Network Training
 * by Reducing Internal Covariate Shift" (Ioffe & Szegedy, 2015)
 */
public class BatchNorm1d implements Layer {

    private int numFeatures;
    private double eps;
    private double momentum;
    private boolean training;

    // Learnable parameters
    private Tensor gamma;  // Scale (initialized to 1)
    private Tensor beta;   // Shift (initialized to 0)

    // Running statistics (buffers, not trained by backprop)
    private double[] runningMean;
    private double[] runningVar;

    // Last output (for diagnostics)
    private Tensor out;

    /**
     * Initialize BatchNorm layer
     *
     * @param numFeatures Number of features (neurons) to normalize
     * @param eps Small constant for numerical stability (default: 1e-5)
     * @param momentum Momentum for running statistics update (default: 0.1)
     */
    public BatchNorm1d(int numFeatures, double eps, double momentum) {
        this.numFeatures = numFeatures;
        this.eps = eps;
        this.momentum = momentum;
        this.training = true;

        // Initialize gamma (scale) to 1
        double[] gammaData = new double[numFeatures];
        Arrays.fill(gammaData, 1.0);
        this.gamma = new Tensor(gammaData, new int[]{numFeatures}, true);
        this.gamma.requiresGrad(true);

        // Initialize beta (shift) to 0
        this.beta = Tensor.zeros(numFeatures);
        this.beta.requiresGrad(true);

        // Initialize running statistics
        this.runningMean = new double[numFeatures];
        this.runningVar = new double[numFeatures];
        Arrays.fill(runningVar, 1.0);  // Start with variance = 1
    }

    public BatchNorm1d(int numFeatures) {
        this(numFeatures, 1e-5, 0.1);
    }

    @Override
    public Tensor forward(Tensor x) {
        // x shape: (batch, numFeatures)

        if (training) {
            // Training mode: use batch statistics

            // Calculate batch mean and variance
            Tensor mean = x.mean(0);  // Shape: (numFeatures,)
            Tensor variance = x.variance(0);  // Shape: (numFeatures,)

            // Normalize: (x - mean) / sqrt(var + eps)
            Tensor xCentered = x.subtract(mean);
            Tensor varPlusEps = variance.add(eps);
            Tensor std = varPlusEps.sqrt();
            Tensor xNorm = xCentered.div(std);

            // Scale and shift: gamma * xNorm + beta
            out = xNorm.mul(gamma).add(beta);

            // Update running statistics (no gradient!)
            double[] meanData = mean.getData();
            double[] varData = variance.getData();

            for (int i = 0; i < numFeatures; i++) {
                runningMean[i] = (1 - momentum) * runningMean[i] + momentum * meanData[i];
                runningVar[i] = (1 - momentum) * runningVar[i] + momentum * varData[i];
            }

        } else {
            // Inference mode: use running statistics

            // Create tensors from running stats
            Tensor mean = new Tensor(runningMean.clone(), new int[]{numFeatures});
            Tensor variance = new Tensor(runningVar.clone(), new int[]{numFeatures});

            // Normalize
            Tensor xCentered = x.subtract(mean);
            Tensor varPlusEps = variance.add(eps);
            Tensor std = varPlusEps.sqrt();
            Tensor xNorm = xCentered.div(std);

            // Scale and shift
            out = xNorm.mul(gamma).add(beta);
        }

        return out;
    }

    @Override
    public List<Tensor> parameters() {
        return Arrays.asList(gamma, beta);
    }

    @Override
    public void setTraining(boolean training) {
        this.training = training;
    }

    @Override
    public Tensor getOutput() {
        return out;
    }

    public Tensor getGamma() {
        return gamma;
    }

    public Tensor getBeta() {
        return beta;
    }

    public double[] getRunningMean() {
        return runningMean;
    }

    public double[] getRunningVar() {
        return runningVar;
    }
}