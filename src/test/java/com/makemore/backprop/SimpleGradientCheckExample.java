package com.makemore.backprop;

import com.makemore.mlp.Tensor;
import com.makemore.backprop.ManualBackprop;

/**
 * 簡單的梯度檢查示例
 * 展示如何使用 GradientChecker 驗證手動梯度
 *
 * 不使用 JUnit，可以直接運行
 */
public class SimpleGradientCheckExample {

    public static void main(String[] args) {
        System.out.println("=".repeat(80));
        System.out.println("Gradient Checking Examples");
        System.out.println("=".repeat(80) + "\n");

        // Example 1: Cross-Entropy Backward
        example1_CrossEntropy();

        // Example 2: BatchNorm Backward
        example2_BatchNorm();

        // Example 3: Numerical Gradient
        example3_NumericalGradient();

        System.out.println("\n" + "=".repeat(80));
        System.out.println("All examples completed!");
        System.out.println("=".repeat(80));
    }

    /**
     * Example 1: 檢查 Cross-Entropy Backward
     */
    private static void example1_CrossEntropy() {
        System.out.println("\n--- Example 1: Cross-Entropy Backward ---\n");

        // 準備數據
        double[] logitsData = {
                1.0, 2.0, 0.5,  // sample 0
                0.3, 0.7, 2.1,  // sample 1
                1.8, 0.2, 0.9   // sample 2
        };
        Tensor logits = new Tensor(logitsData, new int[]{3, 3});
        Tensor Y = new Tensor(new double[]{1, 2, 0}, new int[]{3});  // 真實標籤

        // 手動梯度 (你的實現)
        Tensor dlogits_manual = ManualBackprop.crossEntropyBackward(logits, Y);

        // 參考梯度 (用 softmax 公式)
        Tensor dlogits_reference = computeReferenceGrad(logits, Y);

        // 比較
        boolean passed = GradientChecker.check("dlogits", dlogits_manual, dlogits_reference);

        if (passed) {
            System.out.println("✅ Cross-Entropy gradient is CORRECT!");
        } else {
            System.out.println("❌ Cross-Entropy gradient has ERRORS!");
            System.out.println("   Check your implementation in ManualBackprop.crossEntropyBackward()");
        }
    }

    /**
     * Example 2: 檢查 BatchNorm Backward
     */
    private static void example2_BatchNorm() {
        System.out.println("\n--- Example 2: BatchNorm Backward ---\n");

        // 準備數據
        Tensor x = randomTensor(8, 10);
        Tensor gamma = onesLike(10);
        Tensor dout = randomTensor(8, 10);

        // 手動梯度
        ManualBackprop.BatchNormGradients grads_manual = ManualBackprop.batchNormBackward(
                dout, x, gamma, 1e-5);  // 改這行：只需要 4 個參數

        // 參考梯度 (簡化版 - 只檢查 dgamma 和 dbeta)
        Tensor mean = x.mean(0);
        Tensor var = x.variance(0);
        Tensor std = var.add(1e-5).sqrt();
        Tensor xhat = x.subtract(mean).div(std);
        Tensor dgamma_ref = dout.mul(xhat).sum(0);
        Tensor dbeta_ref = dout.sum(0);

        // 比較
        boolean gamma_ok = GradientChecker.check("dgamma", grads_manual.dgamma, dgamma_ref);
        boolean beta_ok = GradientChecker.check("dbeta", grads_manual.dbeta, dbeta_ref);

        if (gamma_ok && beta_ok) {
            System.out.println("✅ BatchNorm gradients (gamma, beta) are CORRECT!");
            System.out.println("   Note: dx gradient is more complex, see full tests for verification");
        } else {
            System.out.println("❌ BatchNorm gradients have ERRORS!");
        }
    }

    /**
     * Example 3: 使用數值梯度檢查
     */
    private static void example3_NumericalGradient() {
        System.out.println("\n--- Example 3: Numerical Gradient Check ---\n");

        // 簡單的函數: f(x) = sum(x^2)
        Tensor x = randomTensor(5);

        // 解析梯度: df/dx = 2*x
        double[] grad_analytic = new double[5];
        double[] x_data = x.getData();
        for (int i = 0; i < 5; i++) {
            grad_analytic[i] = 2.0 * x_data[i];
        }
        Tensor dlogits_analytic = new Tensor(grad_analytic, new int[]{5});

        // 數值梯度
        Tensor dlogits_numerical = GradientChecker.numericalGradient(
                () -> {
                    double sum = 0.0;
                    for (double val : x.getData()) {
                        sum += val * val;
                    }
                    return sum;
                },
                x,
                1e-5
        );

        // 比較
        boolean passed = GradientChecker.compare("df/dx", dlogits_analytic, dlogits_numerical);

        if (passed) {
            System.out.println("✅ Numerical gradient check PASSED!");
            System.out.println("   This validates the numerical gradient method itself.");
        }
    }

    // ========================================================================
    // Helper Methods
    // ========================================================================

    /**
     * 計算 Cross-Entropy 的參考梯度
     */
// 修改方法簽名：
    private static Tensor computeReferenceGrad(Tensor logits, Tensor Y) {
        int batchSize = logits.getShape()[0];
        int numClasses = logits.getShape()[1];

        // softmax
        Tensor probs = logits.softmax(1);
        double[] probs_data = probs.getData();
        double[] Y_data = Y.getData();  // 新增這行

        // 減去 one-hot
        for (int i = 0; i < batchSize; i++) {
            int label = (int) Y_data[i];  // 修改這行
            probs_data[i * numClasses + label] -= 1.0;
        }

        // 除以 batch_size
        for (int i = 0; i < probs_data.length; i++) {
            probs_data[i] /= batchSize;
        }

        return probs;
    }

    /**
     * 創建隨機張量
     */
    private static Tensor randomTensor(int... shape) {
        int size = 1;
        for (int dim : shape) {
            size *= dim;
        }

        double[] data = new double[size];
        java.util.Random rng = new java.util.Random(42);
        for (int i = 0; i < size; i++) {
            data[i] = rng.nextGaussian() * 0.1;
        }

        return new Tensor(data, shape);
    }

    /**
     * 創建全 1 張量
     */
    private static Tensor onesLike(int size) {
        double[] data = new double[size];
        java.util.Arrays.fill(data, 1.0);
        return new Tensor(data, new int[]{size});
    }

    /**
     * 創建全 0 張量
     */
    private static Tensor zerosLike(int size) {
        return new Tensor(new double[size], new int[]{size});
    }
}