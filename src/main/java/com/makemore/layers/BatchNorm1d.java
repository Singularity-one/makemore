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

    /**
     *
     * 修正：支援 3D 輸入 (batch, seq, features)
     *
     * PyTorch BatchNorm1d 行為：
     * - 2D input (batch, features): normalize across batch
     * - 3D input (batch, seq, features): normalize across batch AND seq
     *
     * 我們採用 3D 輸入時的處理方式：
     * - 將 (batch, seq, features) reshape 成 (batch*seq, features)
     * - 當作 2D 處理
     * - reshape 回 (batch, seq, features)
     */
    @Override
    public Tensor forward(Tensor x) {
        int[] xShape = x.getShape();
        boolean is3D = (xShape.length == 3);

        Tensor x2d;
        int originalBatch = 0;
        int originalSeq = 0;

        // If 3D, reshape to 2D first
        if (is3D) {
            // (batch, seq, features) → (batch*seq, features)
            originalBatch = xShape[0];
            originalSeq = xShape[1];
            int features = xShape[2];

            if (features != numFeatures) {
                throw new IllegalArgumentException(
                        "Expected " + numFeatures + " features, got " + features);
            }

            x2d = x.view(originalBatch * originalSeq, features);
        } else if (xShape.length == 2) {
            // Already 2D: (batch, features)
            if (xShape[1] != numFeatures) {
                throw new IllegalArgumentException(
                        "Expected " + numFeatures + " features, got " + xShape[1]);
            }
            x2d = x;
        } else {
            throw new IllegalArgumentException(
                    "BatchNorm1d expects 2D or 3D input, got " + xShape.length + "D");
        }

        // Now process as 2D
        Tensor out2d;

        if (training) {
            // Training mode: use batch statistics

            // Calculate batch mean and variance
            Tensor mean = x2d.mean(0);  // Shape: (numFeatures,)
            Tensor variance = x2d.variance(0);  // Shape: (numFeatures,)

            // Normalize: (x - mean) / sqrt(var + eps)
            Tensor xCentered = x2d.subtract(mean);
            Tensor varPlusEps = variance.add(eps);
            Tensor std = varPlusEps.sqrt();
            Tensor xNorm = xCentered.div(std);

            // Scale and shift: gamma * xNorm + beta
            out2d = xNorm.mul(gamma).add(beta);

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
            Tensor xCentered = x2d.subtract(mean);
            Tensor varPlusEps = variance.add(eps);
            Tensor std = varPlusEps.sqrt();
            Tensor xNorm = xCentered.div(std);

            // Scale and shift
            out2d = xNorm.mul(gamma).add(beta);
        }

        // If input was 3D, reshape back
        if (is3D) {
            out = out2d.view(originalBatch, originalSeq, numFeatures);
        } else {
            out = out2d;
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