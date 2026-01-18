package com.makemore.backprop;

import com.makemore.mlp.Tensor;
import java.util.Arrays;
import java.util.Random;

/**
 * 梯度檢查工具類
 * 用於驗證手動計算的梯度是否正確
 *
 * 基於 Andrej Karpathy 的 makemore 課程
 * Lecture 5: Becoming a Backprop Ninja
 */
public class GradientChecker {

    private static final double DEFAULT_H = 1e-5;  // 數值梯度的微小擾動
    private static final double TOLERANCE = 1e-5;   // 誤差容忍度

    /**
     * 損失函數介面
     * 用於計算數值梯度時重新計算 forward pass
     */
    @FunctionalInterface
    public interface LossFunction {
        double compute();
    }

    /**
     * 計算數值梯度 (中心差分法)
     * f'(x) ≈ [f(x+h) - f(x-h)] / (2h)
     *
     * @param lossFunc 損失函數
     * @param param 要檢查的參數
     * @param h 微小擾動 (通常 1e-5)
     * @return 數值梯度
     */
    public static Tensor numericalGradient(LossFunction lossFunc, Tensor param, double h) {
        double[] numGrad = new double[param.getSize()];
        double[] paramData = param.getData();

        // 對每個參數元素計算數值梯度
        for (int i = 0; i < param.getSize(); i++) {
            double original = paramData[i];

            // f(x + h)
            paramData[i] = original + h;
            double lossPlus = lossFunc.compute();

            // f(x - h)
            paramData[i] = original - h;
            double lossMinus = lossFunc.compute();

            // 中心差分
            numGrad[i] = (lossPlus - lossMinus) / (2.0 * h);

            // 恢復原值
            paramData[i] = original;
        }

        return new Tensor(numGrad, param.getShape());
    }

    /**
     * 數值梯度 (使用默認 h=1e-5)
     */
    public static Tensor numericalGradient(LossFunction lossFunc, Tensor param) {
        return numericalGradient(lossFunc, param, DEFAULT_H);
    }

    /**
     * 比較兩個梯度張量
     *
     * @param name 參數名稱
     * @param analyticGrad 解析梯度 (手動計算的)
     * @param referenceGrad 參考梯度 (autograd 或數值梯度)
     * @return 是否通過檢查
     */
    public static boolean compare(String name, Tensor analyticGrad, Tensor referenceGrad) {
        return compare(name, analyticGrad, referenceGrad, TOLERANCE);
    }

    /**
     * 比較兩個梯度張量 (自定義容忍度)
     *
     * @param name 參數名稱
     * @param analyticGrad 解析梯度
     * @param referenceGrad 參考梯度
     * @param tolerance 容忍度
     * @return 是否通過檢查
     */
    public static boolean compare(String name, Tensor analyticGrad,
                                  Tensor referenceGrad, double tolerance) {
        // 檢查形狀
        if (!Arrays.equals(analyticGrad.getShape(), referenceGrad.getShape())) {
            System.out.printf("%-20s | ❌ SHAPE MISMATCH: %s vs %s\n",
                    name,
                    Arrays.toString(analyticGrad.getShape()),
                    Arrays.toString(referenceGrad.getShape()));
            return false;
        }

        double[] ag = analyticGrad.getData();
        double[] rg = referenceGrad.getData();

        // 計算統計數據
        double maxDiff = 0.0;
        double sumDiff = 0.0;
        double sumRelDiff = 0.0;
        int exactCount = 0;

        for (int i = 0; i < ag.length; i++) {
            double diff = Math.abs(ag[i] - rg[i]);
            maxDiff = Math.max(maxDiff, diff);
            sumDiff += diff;

            // 相對誤差
            double denominator = Math.max(Math.abs(ag[i]), Math.abs(rg[i]));
            if (denominator > 1e-10) {
                sumRelDiff += diff / denominator;
            }

            // 完全相等的數量
            if (diff < 1e-10) {
                exactCount++;
            }
        }

        double avgDiff = sumDiff / ag.length;
        double avgRelDiff = sumRelDiff / ag.length;
        boolean isClose = maxDiff < tolerance;

        // 輸出格式類似 Karpathy 的 cmp() 函數
        System.out.printf("%-20s | %s | max_diff: %.2e | avg_diff: %.2e | exact: %d/%d\n",
                name,
                isClose ? "✅ PASS" : "❌ FAIL",
                maxDiff,
                avgDiff,
                exactCount,
                ag.length);

        return isClose;
    }

    /**
     * 簡化版檢查 (只顯示通過/失敗)
     */
    public static boolean check(String name, Tensor manual, Tensor reference) {
        double[] m = manual.getData();
        double[] r = reference.getData();

        if (!Arrays.equals(manual.getShape(), reference.getShape())) {
            System.out.printf("%s: ❌ SHAPE MISMATCH\n", name);
            return false;
        }

        double maxDiff = 0.0;
        for (int i = 0; i < m.length; i++) {
            maxDiff = Math.max(maxDiff, Math.abs(m[i] - r[i]));
        }

        boolean ok = maxDiff < TOLERANCE;
        System.out.printf("%s: %s (max_diff=%.2e)\n",
                name, ok ? "✅" : "❌", maxDiff);

        return ok;
    }

    /**
     * 批量檢查多個梯度
     */
    public static boolean compareAll(String[] names, Tensor[] analyticsGrads,
                                     Tensor[] referenceGrads) {
        boolean allPassed = true;

        System.out.println("\n" + "=".repeat(80));
        System.out.println("Gradient Checking Results:");
        System.out.println("=".repeat(80));

        for (int i = 0; i < names.length; i++) {
            boolean passed = compare(names[i], analyticsGrads[i], referenceGrads[i]);
            allPassed = allPassed && passed;
        }

        System.out.println("=".repeat(80));
        System.out.println(allPassed ? "✅ ALL TESTS PASSED" : "❌ SOME TESTS FAILED");
        System.out.println("=".repeat(80) + "\n");

        return allPassed;
    }

    /**
     * 隨機抽樣檢查 (用於大型張量)
     * 只檢查隨機選取的 sampleSize 個元素
     */
    public static boolean compareSampled(String name, Tensor analyticGrad,
                                         Tensor referenceGrad, int sampleSize) {
        double[] ag = analyticGrad.getData();
        double[] rg = referenceGrad.getData();

        Random rng = new Random(42);
        double maxDiff = 0.0;

        for (int i = 0; i < sampleSize; i++) {
            int idx = rng.nextInt(ag.length);
            double diff = Math.abs(ag[idx] - rg[idx]);
            maxDiff = Math.max(maxDiff, diff);
        }

        boolean ok = maxDiff < TOLERANCE;
        System.out.printf("%s (sampled %d/%d): %s (max_diff=%.2e)\n",
                name, sampleSize, ag.length, ok ? "✅" : "❌", maxDiff);

        return ok;
    }

    /**
     * 打印梯度統計信息 (用於 debug)
     */
    public static void printStats(String name, Tensor grad) {
        double[] data = grad.getData();

        double min = Double.POSITIVE_INFINITY;
        double max = Double.NEGATIVE_INFINITY;
        double sum = 0.0;
        double sumAbs = 0.0;

        for (double val : data) {
            min = Math.min(min, val);
            max = Math.max(max, val);
            sum += val;
            sumAbs += Math.abs(val);
        }

        double mean = sum / data.length;
        double meanAbs = sumAbs / data.length;

        System.out.printf("%s stats: min=%.2e, max=%.2e, mean=%.2e, mean_abs=%.2e\n",
                name, min, max, mean, meanAbs);
    }

    /**
     * 檢查梯度是否包含 NaN 或 Inf
     */
    public static boolean checkNaN(String name, Tensor grad) {
        double[] data = grad.getData();

        for (double val : data) {
            if (Double.isNaN(val) || Double.isInfinite(val)) {
                System.out.printf("%s: ❌ Contains NaN or Inf!\n", name);
                return false;
            }
        }

        System.out.printf("%s: ✅ No NaN or Inf\n", name);
        return true;
    }
}