package com.makemore.backprop;

import com.makemore.mlp.Tensor;

import java.util.Arrays;

/**
 * Manual Backpropagation Implementations
 *
 * This class contains optimized backward pass implementations for common
 * neural network operations, demonstrating how gradients flow through
 * the computation graph WITHOUT using automatic differentiation.
 *
 * Based on Karpathy's "Becoming a Backprop Ninja" lecture.
 */
public class ManualBackprop {

    /**
     * Optimized Cross-Entropy Loss Backward Pass
     *
     * Combines the backward pass through:
     * - Softmax
     * - Negative log-likelihood
     * - Mean reduction
     *
     * Mathematical derivation:
     * ```
     * loss = -log(softmax(logits)[targets])
     *
     * For each logit[i,j]:
     *   if j == targets[i]:
     *     ∂loss/∂logits[i,j] = (softmax[i,j] - 1) / batch_size
     *   else:
     *     ∂loss/∂logits[i,j] = softmax[i,j] / batch_size
     * ```
     *
     * This is MUCH more efficient than backpropagating through each
     * operation separately!
     *
     * @param logits Pre-activation outputs (batch, vocab_size)
     * @param targets Target indices (batch,)
     * @return Gradient w.r.t. logits (batch, vocab_size)
     */
    public static Tensor crossEntropyBackward(Tensor logits, Tensor targets) {
        int[] shape = logits.getShape();
        int batchSize = shape[0];
        int vocabSize = shape[1];

        // Compute softmax probabilities
        Tensor probs = logits.softmax(1);

        // Initialize gradient as probs
        Tensor dlogits = probs.copy();

        double[] dlogitsData = dlogits.getData();
        double[] targetData = targets.getData();

        // Subtract 1 from the correct class probabilities
        for (int i = 0; i < batchSize; i++) {
            int target = (int) targetData[i];
            dlogitsData[i * vocabSize + target] -= 1.0;
        }

        // Divide by batch size (from mean reduction)
        for (int i = 0; i < dlogitsData.length; i++) {
            dlogitsData[i] /= batchSize;
        }

        return dlogits;
    }

    /**
     * Optimized Batch Normalization Backward Pass
     *
     * Now using Tensor.mul() with broadcasting support!
     */
    public static BatchNormGradients batchNormBackward(
            Tensor dout, Tensor x, Tensor gamma, double eps) {

        int[] shape = x.getShape();
        int batchSize = shape[0];
        int features = shape[1];
        double n = (double) batchSize;

        // Recompute forward pass statistics
        Tensor mean = x.mean(0);
        Tensor variance = x.variance(0);

        // Compute intermediate values
        Tensor xmu = x.subtract(mean);  // x - mean
        Tensor std = variance.add(eps).sqrt();  // sqrt(var + eps)
        Tensor xhat = xmu.div(std);  // normalized x

        // Gradient w.r.t. gamma and beta (easy part)
        Tensor dgamma = xhat.mul(dout).sum(0);
        Tensor dbeta = dout.sum(0);


//        System.out.println("dout shape: " + Arrays.toString(dout.getShape()));
//        System.out.println("gamma shape: " + Arrays.toString(gamma.getShape()));
        // Gradient w.r.t. input (hard part!)
        // dxhat = dout * gamma (現在支持廣播了!)
        Tensor dxhat = dout.mul(gamma);  // ✅ (batch, features) * (features,)

        // Gradient w.r.t. variance
        // dvar = sum(dxhat * xhat) * -0.5 * (var + eps)^(-1.5)
        Tensor dvar = dxhat.mul(xhat)
                .sum(0)
                .mul(-0.5)
                .mul(variance.add(eps).pow(-1.5));

        // Gradient w.r.t. input (優化版公式 - Exercise 3)
        // dx = (1/N) * gamma * (1/std) * (N*dout - dout.sum(0) - xhat*dgamma)
        Tensor dout_sum = dout.sum(0);           // (features,)
        Tensor xhat_dgamma = xhat.mul(dgamma);   // (batch, features)
        Tensor rstd = std.reciprocal();          // (features,)

        Tensor dx = dout.mul(batchSize)          // N * dout
                .subtract(dout_sum)       // - sum(dout)
                .subtract(xhat_dgamma)    // - xhat * dgamma
                .mul(gamma)               // * gamma
                .mul(rstd)                // * (1/std)
                .mul(1.0 / batchSize);    // * (1/N)

        return new BatchNormGradients(dx, dgamma, dbeta);
    }

    /**
     * Tanh Backward Pass
     *
     * Given y = tanh(x), we have:
     * dy/dx = 1 - tanh²(x) = 1 - y²
     *
     * @param dout Gradient from next layer
     * @param out Output from forward pass (tanh activation)
     * @return Gradient w.r.t. input
     */
    public static Tensor tanhBackward(Tensor dout, Tensor out) {
        // d/dx tanh(x) = 1 - tanh²(x)
        // Since we have tanh(x) as 'out', we use: 1 - out²

        double[] outData = out.getData();
        double[] doutData = dout.getData();
        double[] result = new double[outData.length];

        for (int i = 0; i < outData.length; i++) {
            double tanhVal = outData[i];
            result[i] = (1.0 - tanhVal * tanhVal) * doutData[i];
        }

        return new Tensor(result, out.getShape());
    }

    /**
     * Linear Layer Backward Pass
     *
     * Given y = x @ W + b, we have:
     * dy/dW = x^T @ dout
     * dy/db = sum(dout, axis=0)
     * dy/dx = dout @ W^T
     *
     * @param dout Gradient from next layer (batch, out_features)
     * @param x Input to linear layer (batch, in_features)
     * @param W Weight matrix (in_features, out_features)
     * @return LinearGradients containing dx, dW, db
     */
    public static LinearGradients linearBackward(
            Tensor dout, Tensor x, Tensor W) {

        // Gradient w.r.t. input: dout @ W^T
        Tensor dx = dout.matmul(W.transpose());

        // Gradient w.r.t. weight: x^T @ dout
        Tensor dW = x.transpose().matmul(dout);

        // Gradient w.r.t. bias: sum over batch
        Tensor db = dout.sum(0);

        return new LinearGradients(dx, dW, db);
    }

    /**
     * Embedding Layer Backward Pass
     *
     * Scatter gradients back to embedding table based on indices.
     * Only the embeddings that were actually used get gradients.
     *
     * @param dout Gradient from next layer (batch, block_size, emb_dim)
     * @param indices Input indices (batch, block_size)
     * @param vocabSize Size of vocabulary
     * @param embDim Embedding dimension
     * @return Gradient w.r.t. embedding table (vocab_size, emb_dim)
     */
    public static Tensor embeddingBackward(
            Tensor dout, Tensor indices, int vocabSize, int embDim) {

        int[] doutShape = dout.getShape();
        int batchSize = doutShape[0];
        int blockSize = doutShape[1];

        double[] dC = new double[vocabSize * embDim];
        double[] doutData = dout.getData();
        double[] indicesData = indices.getData();

        // Scatter gradients to embedding table
        for (int i = 0; i < batchSize; i++) {
            for (int j = 0; j < blockSize; j++) {
                int idx = (int) indicesData[i * blockSize + j];

                // Add gradient to this embedding
                for (int k = 0; k < embDim; k++) {
                    int doutIdx = (i * blockSize + j) * embDim + k;
                    int embIdx = idx * embDim + k;
                    dC[embIdx] += doutData[doutIdx];
                }
            }
        }

        return new Tensor(dC, new int[]{vocabSize, embDim});
    }

    /**
     * Helper: Multiply with broadcasting (batch, features) * (features,)
     */
    private static Tensor multiplyBroadcast(Tensor a, Tensor b) {
        // a: (batch, features)
        // b: (features,)
        // result: (batch, features)

        int[] aShape = a.getShape();
        int batchSize = aShape[0];
        int features = aShape[1];

        double[] aData = a.getData();
        double[] bData = b.getData();
        double[] result = new double[batchSize * features];

        for (int i = 0; i < batchSize; i++) {
            for (int j = 0; j < features; j++) {
                result[i * features + j] = aData[i * features + j] * bData[j];
            }
        }

        return new Tensor(result, aShape);
    }

    /**
     * Helper: Add with broadcasting (features,) → (batch, features)
     */
    private static Tensor addBroadcast(Tensor a, int batchSize) {
        // a: (features,)
        // result: (batch, features) - each row is a copy of a

        int features = a.getShape()[0];
        double[] aData = a.getData();
        double[] result = new double[batchSize * features];

        for (int i = 0; i < batchSize; i++) {
            System.arraycopy(aData, 0, result, i * features, features);
        }

        return new Tensor(result, new int[]{batchSize, features});
    }

    /**
     * Container for BatchNorm gradients
     */
    public static class BatchNormGradients {
        public final Tensor dx;      // Gradient w.r.t. input
        public final Tensor dgamma;  // Gradient w.r.t. scale
        public final Tensor dbeta;   // Gradient w.r.t. shift

        public BatchNormGradients(Tensor dx, Tensor dgamma, Tensor dbeta) {
            this.dx = dx;
            this.dgamma = dgamma;
            this.dbeta = dbeta;
        }
    }

    /**
     * Container for Linear layer gradients
     */
    public static class LinearGradients {
        public final Tensor dx;  // Gradient w.r.t. input
        public final Tensor dW;  // Gradient w.r.t. weight
        public final Tensor db;  // Gradient w.r.t. bias

        public LinearGradients(Tensor dx, Tensor dW, Tensor db) {
            this.dx = dx;
            this.dW = dW;
            this.db = db;
        }
    }
}