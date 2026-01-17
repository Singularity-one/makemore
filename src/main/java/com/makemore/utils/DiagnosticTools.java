package com.makemore.utils;

import com.makemore.layers.*;
import com.makemore.mlp.Tensor;
import java.util.List;

/**
 * Diagnostic tools for analyzing neural network training
 *
 * Key metrics:
 * - Activation statistics (mean, std, saturation)
 * - Gradient statistics (mean, std, dead neurons)
 * - Update ratios (gradient_std / param_std)
 */
public class DiagnosticTools {

    /**
     * Analyze activation statistics for all layers
     */
    public static void analyzeActivations(List<Layer> layers, String phase) {
        System.out.println("Activation Analysis - " + phase);
        System.out.println("-".repeat(70));

        int layerIdx = 0;
        for (Layer layer : layers) {
            Tensor out = layer.getOutput();

            if (out != null && layer instanceof TanhLayer) {
                double[] data = out.getData();

                // Calculate statistics
                double mean = calculateMean(data);
                double std = calculateStd(data, mean);
                double saturation = calculateSaturation(data);

                System.out.printf("Layer %2d (%12s): mean=%+.2f, std=%.2f, saturated=%.2f%%\n",
                        layerIdx, layer.getClass().getSimpleName(), mean, std, saturation * 100);
            }

            layerIdx++;
        }

        System.out.println();
    }

    /**
     * Calculate saturation for Tanh activations
     * Saturation = percentage of values with |tanh(x)| > 0.99
     */
    private static double calculateSaturation(double[] data) {
        int saturated = 0;
        for (double val : data) {
            if (Math.abs(val) > 0.99) {
                saturated++;
            }
        }
        return (double) saturated / data.length;
    }

    /**
     * Calculate mean
     */
    private static double calculateMean(double[] data) {
        double sum = 0.0;
        for (double val : data) {
            sum += val;
        }
        return sum / data.length;
    }

    /**
     * Calculate standard deviation
     */
    private static double calculateStd(double[] data, double mean) {
        double sumSq = 0.0;
        for (double val : data) {
            double diff = val - mean;
            sumSq += diff * diff;
        }
        return Math.sqrt(sumSq / data.length);
    }

    /**
     * Analyze gradient statistics
     */
    public static void analyzeGradients(List<Layer> layers) {
        System.out.println("Gradient Analysis");
        System.out.println("-".repeat(70));

        int layerIdx = 0;
        for (Layer layer : layers) {
            if (layer instanceof Linear) {
                Linear linear = (Linear) layer;
                Tensor weight = linear.getWeight();

                double[] grad = weight.getGrad();
                if (grad != null) {
                    double gradMean = calculateMean(grad);
                    double gradStd = calculateStd(grad, gradMean);

                    double[] data = weight.getData();
                    double dataMean = calculateMean(data);
                    double dataStd = calculateStd(data, dataMean);

                    double ratio = gradStd / dataStd;

                    System.out.printf("Layer %2d (Linear): grad_std=%.6f, data_std=%.6f, ratio=%.1e\n",
                            layerIdx, gradStd, dataStd, ratio);
                }
            }
            layerIdx++;
        }

        System.out.println();
    }

    /**
     * Check for dead neurons (neurons that never activate)
     */
    public static void checkDeadNeurons(List<Layer> layers) {
        System.out.println("Dead Neuron Check");
        System.out.println("-".repeat(70));

        int layerIdx = 0;
        for (Layer layer : layers) {
            if (layer instanceof TanhLayer) {
                Tensor out = layer.getOutput();
                if (out != null) {
                    double[] data = out.getData();
                    int[] shape = out.getShape();

                    if (shape.length == 2) {
                        int batchSize = shape[0];
                        int features = shape[1];

                        int deadCount = 0;
                        for (int j = 0; j < features; j++) {
                            boolean allSaturated = true;
                            for (int i = 0; i < batchSize; i++) {
                                if (Math.abs(data[i * features + j]) < 0.99) {
                                    allSaturated = false;
                                    break;
                                }
                            }
                            if (allSaturated) {
                                deadCount++;
                            }
                        }

                        if (deadCount > 0) {
                            System.out.printf("Layer %2d: %d / %d neurons saturated (%.1f%%)\n",
                                    layerIdx, deadCount, features, 100.0 * deadCount / features);
                        }
                    }
                }
            }
            layerIdx++;
        }

        System.out.println();
    }

    /**
     * Print activation distribution summary
     */
    public static void printActivationSummary(Tensor activations, String layerName) {
        double[] data = activations.getData();
        double mean = calculateMean(data);
        double std = calculateStd(data, mean);

        // Find min and max
        double min = Double.POSITIVE_INFINITY;
        double max = Double.NEGATIVE_INFINITY;
        for (double val : data) {
            if (val < min) min = val;
            if (val > max) max = val;
        }

        System.out.printf("%s: mean=%.4f, std=%.4f, min=%.4f, max=%.4f\n",
                layerName, mean, std, min, max);
    }
}