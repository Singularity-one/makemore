package com.makemore.bigram;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.BeforeEach;
import static org.junit.jupiter.api.Assertions.*;

import java.util.Arrays;

/**
 * Unit tests for Tensor operations and BigramLanguageModel
 */
public class TensorTest {

    private static final double EPSILON = 1e-6;

    @Test
    public void testTensorCreation() {
        Tensor t = Tensor.zeros(3, 4);
        assertArrayEquals(new int[]{3, 4}, t.getShape());
        assertEquals(12, t.getSize());

        for (int i = 0; i < 12; i++) {
            assertEquals(0.0, t.getData()[i], EPSILON);
        }
    }

    @Test
    public void testTensorOnes() {
        Tensor t = Tensor.ones(2, 3);
        for (double val : t.getData()) {
            assertEquals(1.0, val, EPSILON);
        }
    }

    @Test
    public void testMatMul() {
        // Create 2x3 matrix
        double[] data1 = {1, 2, 3, 4, 5, 6};
        Tensor a = new Tensor(data1, new int[]{2, 3});

        // Create 3x2 matrix
        double[] data2 = {1, 2, 3, 4, 5, 6};
        Tensor b = new Tensor(data2, new int[]{3, 2});

        // Result should be 2x2
        Tensor c = a.matmul(b);
        assertArrayEquals(new int[]{2, 2}, c.getShape());

        // Check values
        // [1,2,3] @ [1,2]   = 1*1+2*3+3*5 = 22
        //           [3,4]
        //           [5,6]
        assertEquals(22.0, c.get(0, 0), EPSILON);
        assertEquals(28.0, c.get(0, 1), EPSILON);
        assertEquals(49.0, c.get(1, 0), EPSILON);
        assertEquals(64.0, c.get(1, 1), EPSILON);
    }

    @Test
    public void testMatMulGradient() {
        // Simple gradient test for matmul
        double[] data1 = {1, 2};
        Tensor a = new Tensor(data1, new int[]{1, 2}, true).requiresGrad(true);

        double[] data2 = {3, 4};
        Tensor b = new Tensor(data2, new int[]{2, 1}, true).requiresGrad(true);

        Tensor c = a.matmul(b);
        c.backward();

        // Gradient w.r.t a should be b.T
        assertArrayEquals(new double[]{3, 4}, a.getGrad(), EPSILON);

        // Gradient w.r.t b should be a.T
        assertArrayEquals(new double[]{1, 2}, b.getGrad(), EPSILON);
    }

    @Test
    public void testExp() {
        double[] data = {0, 1, 2};
        Tensor t = new Tensor(data, new int[]{3});
        Tensor result = t.exp();

        assertEquals(1.0, result.get(0), EPSILON);
        assertEquals(Math.E, result.get(1), EPSILON);
        assertEquals(Math.E * Math.E, result.get(2), EPSILON);
    }

    @Test
    public void testExpGradient() {
        double[] data = {1.0};
        Tensor t = new Tensor(data, new int[]{1}, true).requiresGrad(true);

        Tensor result = t.exp();
        result.backward();

        // d/dx exp(x) = exp(x)
        assertEquals(Math.E, t.getGrad()[0], EPSILON);
    }

    @Test
    public void testLog() {
        double[] data = {1, Math.E, Math.E * Math.E};
        Tensor t = new Tensor(data, new int[]{3});
        Tensor result = t.log();

        assertEquals(0.0, result.get(0), EPSILON);
        assertEquals(1.0, result.get(1), EPSILON);
        assertEquals(2.0, result.get(2), EPSILON);
    }

    @Test
    public void testLogGradient() {
        double[] data = {2.0};
        Tensor t = new Tensor(data, new int[]{1}, true).requiresGrad(true);

        Tensor result = t.log();
        result.backward();

        // d/dx log(x) = 1/x
        assertEquals(0.5, t.getGrad()[0], EPSILON);
    }

    @Test
    public void testSum() {
        // 2x3 matrix
        double[] data = {1, 2, 3, 4, 5, 6};
        Tensor t = new Tensor(data, new int[]{2, 3});

        // Sum along dimension 1 (columns)
        Tensor summed = t.sum(1, true);
        assertArrayEquals(new int[]{2, 1}, summed.getShape());
        assertEquals(6.0, summed.get(0), EPSILON);   // 1+2+3
        assertEquals(15.0, summed.get(1), EPSILON);  // 4+5+6
    }

    @Test
    public void testDivision() {
        double[] data1 = {6, 8, 10};
        Tensor a = new Tensor(data1, new int[]{3});

        double[] data2 = {2, 2, 2};
        Tensor b = new Tensor(data2, new int[]{3});

        Tensor c = a.div(b);
        assertEquals(3.0, c.get(0), EPSILON);
        assertEquals(4.0, c.get(1), EPSILON);
        assertEquals(5.0, c.get(2), EPSILON);
    }

    @Test
    public void testBroadcastDivision() {
        // 2x3 matrix
        double[] data1 = {6, 8, 10, 12, 14, 16};
        Tensor a = new Tensor(data1, new int[]{2, 3});

        // 2x1 matrix (will broadcast)
        double[] data2 = {2, 4};
        Tensor b = new Tensor(data2, new int[]{2, 1});

        Tensor c = a.div(b);

        // First row divided by 2
        assertEquals(3.0, c.get(0, 0), EPSILON);
        assertEquals(4.0, c.get(0, 1), EPSILON);
        assertEquals(5.0, c.get(0, 2), EPSILON);

        // Second row divided by 4
        assertEquals(3.0, c.get(1, 0), EPSILON);
        assertEquals(3.5, c.get(1, 1), EPSILON);
        assertEquals(4.0, c.get(1, 2), EPSILON);
    }

    @Test
    public void testMean() {
        double[] data = {1, 2, 3, 4, 5, 6};
        Tensor t = new Tensor(data, new int[]{6});

        Tensor mean = t.mean();
        assertEquals(3.5, mean.item(), EPSILON);
    }

    @Test
    public void testPower() {
        double[] data = {2, 3, 4};
        Tensor t = new Tensor(data, new int[]{3});

        Tensor squared = t.pow(2);
        assertEquals(4.0, squared.get(0), EPSILON);
        assertEquals(9.0, squared.get(1), EPSILON);
        assertEquals(16.0, squared.get(2), EPSILON);
    }

    @Test
    public void testNegation() {
        double[] data = {1, -2, 3};
        Tensor t = new Tensor(data, new int[]{3});

        Tensor negated = t.neg();
        assertEquals(-1.0, negated.get(0), EPSILON);
        assertEquals(2.0, negated.get(1), EPSILON);
        assertEquals(-3.0, negated.get(2), EPSILON);
    }

    @Test
    public void testScalarMultiplication() {
        double[] data = {1, 2, 3};
        Tensor t = new Tensor(data, new int[]{3});

        Tensor scaled = t.mul(2.0);
        assertEquals(2.0, scaled.get(0), EPSILON);
        assertEquals(4.0, scaled.get(1), EPSILON);
        assertEquals(6.0, scaled.get(2), EPSILON);
    }

    @Test
    public void testAddition() {
        double[] data1 = {1, 2, 3};
        Tensor a = new Tensor(data1, new int[]{3});

        double[] data2 = {4, 5, 6};
        Tensor b = new Tensor(data2, new int[]{3});

        Tensor c = a.add(b);
        assertEquals(5.0, c.get(0), EPSILON);
        assertEquals(7.0, c.get(1), EPSILON);
        assertEquals(9.0, c.get(2), EPSILON);
    }

    @Test
    public void testSoftmaxFlow() {
        // Test a simple softmax computation
        double[] logits = {1, 2, 3};
        Tensor t = new Tensor(logits, new int[]{1, 3});

        // Softmax: exp(x) / sum(exp(x))
        Tensor expT = t.exp();
        Tensor sumExp = expT.sum(1, true);
        Tensor probs = expT.div(sumExp);

        // Check that probabilities sum to 1
        double sum = 0;
        for (int i = 0; i < 3; i++) {
            sum += probs.get(0, i);
        }
        assertEquals(1.0, sum, EPSILON);

        // Check monotonicity (larger logits -> larger probs)
        assertTrue(probs.get(0, 0) < probs.get(0, 1));
        assertTrue(probs.get(0, 1) < probs.get(0, 2));
    }

    @Test
    public void testCompleteForwardBackward() {
        // Test a complete forward and backward pass
        // Simulating: loss = mean((X @ W)²)

        double[] xData = {1, 2};
        Tensor X = new Tensor(xData, new int[]{1, 2}, true).requiresGrad(true);

        double[] wData = {0.5, 0.3};
        Tensor W = new Tensor(wData, new int[]{2, 1}, true).requiresGrad(true);

        // Forward
        Tensor Y = X.matmul(W);
        Tensor loss = Y.pow(2).mean();

        // Backward
        loss.backward();

        // Check that gradients exist
        assertNotNull(X.getGrad());
        assertNotNull(W.getGrad());

        // Gradients should be non-zero
        assertTrue(Math.abs(X.getGrad()[0]) > EPSILON);
        assertTrue(Math.abs(W.getGrad()[0]) > EPSILON);
    }

    @Test
    public void testZeroGrad() {
        double[] data = {1, 2, 3};
        Tensor t = new Tensor(data, new int[]{3}, true);

        // Manually set gradients
        t.getGrad()[0] = 5.0;
        t.getGrad()[1] = 10.0;

        // Zero them
        t.zeroGrad();

        for (double grad : t.getGrad()) {
            assertEquals(0.0, grad, EPSILON);
        }
    }
}