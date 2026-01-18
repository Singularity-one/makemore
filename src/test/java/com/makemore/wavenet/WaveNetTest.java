package com.makemore.wavenet;

import com.makemore.mlp.Tensor;
import com.makemore.layers.*;
import org.junit.jupiter.api.*;

import java.util.*;

import static org.junit.jupiter.api.Assertions.*;

/**
 * WaveNet Model Tests (JUnit 5)
 */
@TestMethodOrder(MethodOrderer.OrderAnnotation.class)
public class WaveNetTest {

    // =========================================================================
    // Test 1: Embedding Layer
    // =========================================================================

    @Test
    @Order(1)
    @DisplayName("Embedding layer forward & parameters")
    void testEmbedding() {
        Random rng = new Random(42);
        Embedding emb = new Embedding(27, 10, rng);

        // 1D input
        Tensor x1D = new Tensor(new double[]{0, 5, 10, 15, 20}, new int[]{5});
        Tensor out1D = emb.forward(x1D);
        assertArrayEquals(new int[]{5, 10}, out1D.getShape());

        // 2D input
        Tensor x2D = new Tensor(new double[]{0, 5, 10, 1, 6, 11}, new int[]{2, 3});
        Tensor out2D = emb.forward(x2D);
        assertArrayEquals(new int[]{2, 3, 10}, out2D.getShape());

        // parameters
        List<Tensor> params = emb.parameters();
        assertEquals(1, params.size());
        assertArrayEquals(new int[]{27, 10}, params.get(0).getShape());
    }

    // =========================================================================
    // Test 2: FlattenConsecutive
    // =========================================================================

    @Test
    @Order(2)
    @DisplayName("FlattenConsecutive reshapes correctly")
    void testFlattenConsecutive() {
        FlattenConsecutive flatten = new FlattenConsecutive(2);

        Tensor x = new Tensor(new double[2 * 8 * 10], new int[]{2, 8, 10});
        Tensor out = flatten.forward(x);

        assertArrayEquals(new int[]{2, 4, 20}, out.getShape());
        assertEquals(0, flatten.parameters().size());
    }

    // =========================================================================
    // Test 3: FlattenLayer
    // =========================================================================

    @Test
    @Order(3)
    @DisplayName("FlattenLayer removes singleton sequence dimension")
    void testFlattenLayer() {
        FlattenLayer flatten = new FlattenLayer();

        Tensor x = new Tensor(new double[2 * 1 * 128], new int[]{2, 1, 128});
        Tensor out = flatten.forward(x);

        assertArrayEquals(new int[]{2, 128}, out.getShape());
    }

    // =========================================================================
    // Test 4: Sequential
    // =========================================================================

    @Test
    @Order(4)
    @DisplayName("Sequential container forwards and collects parameters")
    void testSequential() {
        Random rng = new Random(42);

        Sequential seq = new Sequential(
                new Linear(10, 20, false, rng),
                new TanhLayer(),
                new Linear(20, 5, false, rng)
        );

        Tensor x = new Tensor(new double[32 * 10], new int[]{32, 10});
        Tensor out = seq.forward(x);

        assertArrayEquals(new int[]{32, 5}, out.getShape());
        assertEquals(2, seq.parameters().size());
    }

    // =========================================================================
    // Test 5: Linear 3D Input
    // =========================================================================

    @Test
    @Order(5)
    @DisplayName("Linear layer supports 3D input")
    void testLinear3D() {
        Random rng = new Random(42);
        Linear linear = new Linear(10, 20, false, rng);

        Tensor x = new Tensor(new double[4 * 8 * 10], new int[]{4, 8, 10});
        Tensor out = linear.forward(x);

        assertArrayEquals(new int[]{4, 8, 20}, out.getShape());
    }

    // =========================================================================
    // Test 6: Cross-Entropy Gradient
    // =========================================================================

    @Test
    @Order(6)
    @DisplayName("Cross-entropy backward produces valid gradients")
    void testCrossEntropyGradient() {
        Tensor logits = new Tensor(new double[]{
                1.0, 2.0, 0.5,
                0.3, 0.7, 2.1
        }, new int[]{2, 3}, true);

        Tensor Y = new Tensor(new double[]{1, 2}, new int[]{2});

        Tensor loss = computeCrossEntropyLossTensor(logits, Y);
        logits.zeroGrad();
        loss.backward();

        double[] grad = logits.getGrad();

        double gradSum = Arrays.stream(grad).map(Math::abs).sum();
        assertTrue(gradSum > 1e-2);

        double s0 = grad[0] + grad[1] + grad[2];
        double s1 = grad[3] + grad[4] + grad[5];

        assertEquals(0.0, s0, 1e-6);
        assertEquals(0.0, s1, 1e-6);
    }

    // =========================================================================
    // Test 7: Full WaveNet Forward
    // =========================================================================

    @Test
    @Order(7)
    @DisplayName("Mini WaveNet forward pass produces valid logits")
    void testFullForward() {
        Random rng = new Random(42);

        int blockSize = 4;
        int embDim = 10;
        int hidden = 32;
        int vocab = 27;

        List<Layer> layers = new ArrayList<>();
        layers.add(new Embedding(vocab, embDim, rng));

        int dim = embDim;
        int levels = (int) (Math.log(blockSize) / Math.log(2));

        for (int i = 0; i < levels; i++) {
            layers.add(new FlattenConsecutive(2));
            dim *= 2;
            layers.add(new Linear(dim, hidden, false, rng));
            layers.add(new TanhLayer());
            dim = hidden;
        }

        layers.add(new FlattenLayer());
        layers.add(new Linear(hidden, vocab, false, rng));

        Sequential model = new Sequential(layers);

        Tensor X = new Tensor(new double[]{0,1,2,3,4,5,6,7}, new int[]{2, blockSize});
        Tensor logits = model.forward(X);

        assertArrayEquals(new int[]{2, vocab}, logits.getShape());

        for (double v : logits.getData()) {
            assertFalse(Double.isNaN(v));
            assertFalse(Double.isInfinite(v));
        }
    }

    // =========================================================================
    // Helper: Cross Entropy
    // =========================================================================

    private static Tensor computeCrossEntropyLossTensor(Tensor logits, Tensor Y) {
        int B = logits.getShape()[0];
        int V = logits.getShape()[1];

        Tensor probs = logits.softmax(1);
        double[] p = probs.getData();
        double[] y = Y.getData();

        double lossVal = 0;
        for (int i = 0; i < B; i++) {
            lossVal += -Math.log(p[i * V + (int) y[i]] + 1e-10);
        }
        lossVal /= B;

        Tensor loss = new Tensor(new double[]{lossVal}, new int[]{1}, true);
        loss.prev.add(logits);

        loss.backward = (v) -> {
            double[] g = logits.getGrad();
            for (int i = 0; i < B; i++) {
                int t = (int) y[i];
                for (int j = 0; j < V; j++) {
                    double grad = p[i * V + j] - (j == t ? 1.0 : 0.0);
                    g[i * V + j] += grad / B;
                }
            }
            return null;
        };

        return loss;
    }
}
