package com.makemore.mlp;

import java.util.*;
import java.util.function.Function;

/**
 * Enhanced Tensor with automatic differentiation
 * Supports embeddings, tanh, and all operations needed for MLP
 */
public class Tensor {
    private double[] data;
    private int[] shape;
    private int size;

    // For autograd
    private double[] grad;
    private Function<Void, Void> backward;
    private Set<Tensor> prev;
    private String op;
    private boolean requiresGrad;

    // Constructor
    public Tensor(double[] data, int[] shape, boolean requiresGrad) {
        this.data = Arrays.copyOf(data, data.length);
        this.shape = Arrays.copyOf(shape, shape.length);
        this.size = data.length;
        this.requiresGrad = requiresGrad;
        this.grad = new double[size];
        this.prev = new HashSet<>();
        this.op = "";
        this.backward = null;
    }

    public Tensor(double[] data, int[] shape) {
        this(data, shape, false);
    }

    // Static factory methods
    public static Tensor zeros(int... shape) {
        int size = calculateSize(shape);
        return new Tensor(new double[size], shape);
    }

    public static Tensor ones(int... shape) {
        int size = calculateSize(shape);
        double[] data = new double[size];
        Arrays.fill(data, 1.0);
        return new Tensor(data, shape);
    }

    public static Tensor randn(Random rng, int... shape) {
        int size = calculateSize(shape);
        double[] data = new double[size];
        for (int i = 0; i < size; i++) {
            data[i] = rng.nextGaussian();
        }
        return new Tensor(data, shape);
    }

    public static Tensor arange(int n) {
        double[] data = new double[n];
        for (int i = 0; i < n; i++) {
            data[i] = i;
        }
        return new Tensor(data, new int[]{n});
    }

    private static int calculateSize(int[] shape) {
        int size = 1;
        for (int dim : shape) {
            size *= dim;
        }
        return size;
    }

    // Enable gradient tracking
    public Tensor requiresGrad(boolean requires) {
        this.requiresGrad = requires;
        return this;
    }

    // Indexing operations
    public double get(int... indices) {
        int idx = 0;
        int stride = 1;
        for (int i = shape.length - 1; i >= 0; i--) {
            idx += indices[i] * stride;
            stride *= shape[i];
        }
        return data[idx];
    }

    public void set(int[] indices, double value) {
        int idx = 0;
        int stride = 1;
        for (int i = shape.length - 1; i >= 0; i--) {
            idx += indices[i] * stride;
            stride *= shape[i];
        }
        data[idx] = value;
    }

    // Index select for embeddings: C[X] where X contains indices
    public Tensor index(Tensor indices) {
        if (this.shape.length != 2) {
            throw new IllegalArgumentException("Index select only works on 2D tensors");
        }

        int[] indicesShape = indices.shape;
        int embDim = this.shape[1];

        // Result shape: indices.shape + (embDim,)
        int[] resultShape = new int[indicesShape.length + 1];
        System.arraycopy(indicesShape, 0, resultShape, 0, indicesShape.length);
        resultShape[indicesShape.length] = embDim;

        int resultSize = calculateSize(resultShape);
        double[] result = new double[resultSize];

        // For each index in indices, copy the corresponding row from this
        int numIndices = indices.size;
        for (int i = 0; i < numIndices; i++) {
            int idx = (int) indices.data[i];
            // Copy row idx from this to result
            System.arraycopy(this.data, idx * embDim, result, i * embDim, embDim);
        }

        Tensor out = new Tensor(result, resultShape, requiresGrad || indices.requiresGrad);
        out.prev.add(this);
        out.prev.add(indices);
        out.op = "index";

        if (out.requiresGrad) {
            out.backward = (v) -> {
                if (this.requiresGrad) {
                    // Scatter gradient back to embedding table
                    for (int i = 0; i < numIndices; i++) {
                        int idx = (int) indices.data[i];
                        for (int j = 0; j < embDim; j++) {
                            this.grad[idx * embDim + j] += out.grad[i * embDim + j];
                        }
                    }
                }
                return null;
            };
        }

        return out;
    }

    // Reshape/View operation
    public Tensor view(int... newShape) {
        int newSize = calculateSize(newShape);
        if (newSize != this.size) {
            throw new IllegalArgumentException("Cannot reshape tensor of size " + this.size +
                    " to size " + newSize);
        }

        Tensor out = new Tensor(this.data, newShape, requiresGrad);
        out.prev.add(this);
        out.op = "view";

        if (out.requiresGrad) {
            out.backward = (v) -> {
                if (this.requiresGrad) {
                    // Gradient flows directly through reshape
                    for (int i = 0; i < size; i++) {
                        this.grad[i] += out.grad[i];
                    }
                }
                return null;
            };
        }

        return out;
    }

    // Matrix multiplication
    public Tensor matmul(Tensor other) {
        if (shape.length != 2 || other.shape.length != 2) {
            throw new IllegalArgumentException("Both tensors must be 2D for matmul");
        }
        if (shape[1] != other.shape[0]) {
            throw new IllegalArgumentException("Incompatible shapes for matmul: " +
                    Arrays.toString(shape) + " @ " + Arrays.toString(other.shape));
        }

        int m = shape[0];
        int n = shape[1];
        int p = other.shape[1];

        double[] result = new double[m * p];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < p; j++) {
                double sum = 0.0;
                for (int k = 0; k < n; k++) {
                    sum += data[i * n + k] * other.data[k * p + j];
                }
                result[i * p + j] = sum;
            }
        }

        Tensor out = new Tensor(result, new int[]{m, p}, requiresGrad || other.requiresGrad);
        out.prev.add(this);
        out.prev.add(other);
        out.op = "@";

        if (out.requiresGrad) {
            final int fm = m, fn = n, fp = p;
            out.backward = (v) -> {
                if (this.requiresGrad) {
                    for (int i = 0; i < fm; i++) {
                        for (int k = 0; k < fn; k++) {
                            for (int j = 0; j < fp; j++) {
                                this.grad[i * fn + k] += out.grad[i * fp + j] * other.data[k * fp + j];
                            }
                        }
                    }
                }

                if (other.requiresGrad) {
                    for (int k = 0; k < fn; k++) {
                        for (int j = 0; j < fp; j++) {
                            for (int i = 0; i < fm; i++) {
                                other.grad[k * fp + j] += this.data[i * fn + k] * out.grad[i * fp + j];
                            }
                        }
                    }
                }
                return null;
            };
        }

        return out;
    }

    /**
     * Addition with broadcasting
     *
     * Supports:
     * 1. (batch, features) + (batch, features) - element-wise
     * 2. (batch, features) + (features,) - broadcast along batch
     * 3. (features,) + (batch, features) - broadcast (commutative)
     * 4. (batch, features) + (batch, 1) - broadcast along features
     *
     * @param other Tensor to add
     * @return Result of addition
     */
    public Tensor add(Tensor other) {
        // Case 1: Same shape - element-wise addition
        if (Arrays.equals(shape, other.shape)) {
            double[] result = new double[size];
            for (int i = 0; i < size; i++) {
                result[i] = data[i] + other.data[i];
            }

            Tensor out = new Tensor(result, shape, requiresGrad || other.requiresGrad);
            out.prev.add(this);
            out.prev.add(other);
            out.op = "+";

            if (out.requiresGrad) {
                out.backward = (v) -> {
                    if (this.requiresGrad) {
                        for (int i = 0; i < size; i++) {
                            this.grad[i] += out.grad[i];
                        }
                    }
                    if (other.requiresGrad) {
                        for (int i = 0; i < size; i++) {
                            other.grad[i] += out.grad[i];
                        }
                    }
                    return null;
                };
            }

            return out;
        }

        // Case 2: (batch, features) + (features,) - broadcast along batch
        if (shape.length == 2 && other.shape.length == 1 && shape[1] == other.shape[0]) {
            int batchSize = shape[0];
            int features = shape[1];

            double[] result = new double[size];
            for (int i = 0; i < batchSize; i++) {
                for (int j = 0; j < features; j++) {
                    result[i * features + j] = data[i * features + j] + other.data[j];
                }
            }

            Tensor out = new Tensor(result, shape, requiresGrad || other.requiresGrad);
            out.prev.add(this);
            out.prev.add(other);
            out.op = "+broadcast";

            if (out.requiresGrad) {
                final int fBatch = batchSize;
                final int fFeat = features;
                out.backward = (v) -> {
                    if (this.requiresGrad) {
                        // Gradient for (batch, features) tensor
                        for (int i = 0; i < size; i++) {
                            this.grad[i] += out.grad[i];
                        }
                    }
                    if (other.requiresGrad) {
                        // Gradient for (features,) tensor - sum over batch dimension
                        for (int i = 0; i < fBatch; i++) {
                            for (int j = 0; j < fFeat; j++) {
                                other.grad[j] += out.grad[i * fFeat + j];
                            }
                        }
                    }
                    return null;
                };
            }

            return out;
        }

        // Case 3: (features,) + (batch, features) - swap and use Case 2
        if (shape.length == 1 && other.shape.length == 2 && other.shape[1] == shape[0]) {
            return other.add(this);  // Commutative, so swap
        }

        // Case 4: (batch, features) + (batch, 1) - broadcast along features
        if (shape.length == 2 && other.shape.length == 2 &&
                shape[0] == other.shape[0] && other.shape[1] == 1) {

            int batchSize = shape[0];
            int features = shape[1];

            double[] result = new double[size];
            for (int i = 0; i < batchSize; i++) {
                for (int j = 0; j < features; j++) {
                    result[i * features + j] = data[i * features + j] + other.data[i];
                }
            }

            Tensor out = new Tensor(result, shape, requiresGrad || other.requiresGrad);
            out.prev.add(this);
            out.prev.add(other);
            out.op = "+broadcast_col";

            if (out.requiresGrad) {
                final int fBatch = batchSize;
                final int fFeat = features;
                out.backward = (v) -> {
                    if (this.requiresGrad) {
                        for (int i = 0; i < size; i++) {
                            this.grad[i] += out.grad[i];
                        }
                    }
                    if (other.requiresGrad) {
                        // Sum over features dimension
                        for (int i = 0; i < fBatch; i++) {
                            for (int j = 0; j < fFeat; j++) {
                                other.grad[i] += out.grad[i * fFeat + j];
                            }
                        }
                    }
                    return null;
                };
            }

            return out;
        }

        throw new IllegalArgumentException(
                "Unsupported shapes for add: " + Arrays.toString(shape) +
                        " + " + Arrays.toString(other.shape));
    }

    // Tanh activation
    public Tensor tanh() {
        double[] result = new double[size];
        for (int i = 0; i < size; i++) {
            result[i] = Math.tanh(data[i]);
        }

        Tensor out = new Tensor(result, shape, requiresGrad);
        out.prev.add(this);
        out.op = "tanh";

        if (out.requiresGrad) {
            out.backward = (v) -> {
                for (int i = 0; i < size; i++) {
                    // d/dx tanh(x) = 1 - tanh²(x)
                    double t = out.data[i];
                    this.grad[i] += (1.0 - t * t) * out.grad[i];
                }
                return null;
            };
        }

        return out;
    }

    // Exp
    public Tensor exp() {
        double[] result = new double[size];
        for (int i = 0; i < size; i++) {
            result[i] = Math.exp(data[i]);
        }

        Tensor out = new Tensor(result, shape, requiresGrad);
        out.prev.add(this);
        out.op = "exp";

        if (out.requiresGrad) {
            out.backward = (v) -> {
                for (int i = 0; i < size; i++) {
                    this.grad[i] += out.data[i] * out.grad[i];
                }
                return null;
            };
        }

        return out;
    }

    // Log
    public Tensor log() {
        double[] result = new double[size];
        for (int i = 0; i < size; i++) {
            result[i] = Math.log(data[i]);
        }

        Tensor out = new Tensor(result, shape, requiresGrad);
        out.prev.add(this);
        out.op = "log";

        if (out.requiresGrad) {
            out.backward = (v) -> {
                for (int i = 0; i < size; i++) {
                    this.grad[i] += out.grad[i] / this.data[i];
                }
                return null;
            };
        }

        return out;
    }

    // Sum along dimension
    public Tensor sum(int dim, boolean keepdim) {
        if (shape.length == 1) {
            // 1D tensor, sum all elements
            double sum = 0.0;
            for (double v : data) {
                sum += v;
            }
            int[] outShape = keepdim ? new int[]{1} : new int[]{};
            Tensor out = new Tensor(new double[]{sum}, outShape, requiresGrad);
            out.prev.add(this);
            out.op = "sum";

            if (out.requiresGrad) {
                out.backward = (v) -> {
                    for (int i = 0; i < size; i++) {
                        this.grad[i] += out.grad[0];
                    }
                    return null;
                };
            }
            return out;
        }

        if (shape.length != 2) {
            throw new IllegalArgumentException("Only 1D and 2D tensors supported for sum");
        }

        if (dim == 1) {
            // Sum along columns (result is rows x 1 or rows)
            double[] result = new double[shape[0]];
            for (int i = 0; i < shape[0]; i++) {
                for (int j = 0; j < shape[1]; j++) {
                    result[i] += data[i * shape[1] + j];
                }
            }
            int[] outShape = keepdim ? new int[]{shape[0], 1} : new int[]{shape[0]};

            Tensor out = new Tensor(result, outShape, requiresGrad);
            out.prev.add(this);
            out.op = "sum1";

            if (out.requiresGrad) {
                out.backward = (v) -> {
                    for (int i = 0; i < shape[0]; i++) {
                        for (int j = 0; j < shape[1]; j++) {
                            this.grad[i * shape[1] + j] += out.grad[i];
                        }
                    }
                    return null;
                };
            }

            return out;
        }

        throw new IllegalArgumentException("Invalid dimension for sum: " + dim);
    }

    // Division (element-wise or with broadcasting)
    /**
     * Division with broadcasting
     * Supports:
     * - (batch, features) / (batch, 1)  [原本支持]
     * - (batch, features) / (features,)  [新增! BatchNorm需要]
     * - Element-wise division
     */
    public Tensor div(Tensor other) {
        double[] result = new double[size];

        // Case 1: (batch, features) / (features,) - BatchNorm case!
        if (shape.length == 2 && other.shape.length == 1 && shape[1] == other.shape[0]) {
            int batchSize = shape[0];
            int features = shape[1];

            for (int i = 0; i < batchSize; i++) {
                for (int j = 0; j < features; j++) {
                    result[i * features + j] = data[i * features + j] / other.data[j];
                }
            }

            Tensor out = new Tensor(result, shape, requiresGrad || other.requiresGrad);
            out.prev.add(this);
            out.prev.add(other);
            out.op = "/";

            if (out.requiresGrad) {
                out.backward = (v) -> {
                    if (this.requiresGrad) {
                        for (int i = 0; i < batchSize; i++) {
                            for (int j = 0; j < features; j++) {
                                this.grad[i * features + j] += out.grad[i * features + j] / other.data[j];
                            }
                        }
                    }
                    if (other.requiresGrad) {
                        // ∂L/∂other[j] = sum_i ∂L/∂out[i,j] * (-x[i,j] / other[j]²)
                        for (int i = 0; i < batchSize; i++) {
                            for (int j = 0; j < features; j++) {
                                other.grad[j] += -out.grad[i * features + j] *
                                        this.data[i * features + j] / (other.data[j] * other.data[j]);
                            }
                        }
                    }
                    return null;
                };
            }

            return out;
        }

        // Case 2: (batch, features) / (batch, 1) - 原本的廣播
        if (shape.length == 2 && other.shape.length == 2 &&
                shape[0] == other.shape[0] && other.shape[1] == 1) {

            int batchSize = shape[0];
            int features = shape[1];

            for (int i = 0; i < batchSize; i++) {
                for (int j = 0; j < features; j++) {
                    result[i * features + j] = data[i * features + j] / other.data[i];
                }
            }

            Tensor out = new Tensor(result, shape, requiresGrad || other.requiresGrad);
            out.prev.add(this);
            out.prev.add(other);
            out.op = "/";

            if (out.requiresGrad) {
                out.backward = (v) -> {
                    if (this.requiresGrad) {
                        for (int i = 0; i < batchSize; i++) {
                            for (int j = 0; j < features; j++) {
                                this.grad[i * features + j] += out.grad[i * features + j] / other.data[i];
                            }
                        }
                    }
                    if (other.requiresGrad) {
                        for (int i = 0; i < batchSize; i++) {
                            for (int j = 0; j < features; j++) {
                                other.grad[i] += -out.grad[i * features + j] *
                                        data[i * features + j] / (other.data[i] * other.data[i]);
                            }
                        }
                    }
                    return null;
                };
            }

            return out;
        }

        // Case 3: Element-wise division (same shapes)
        if (Arrays.equals(shape, other.shape)) {
            for (int i = 0; i < size; i++) {
                result[i] = data[i] / other.data[i];
            }

            Tensor out = new Tensor(result, shape, requiresGrad || other.requiresGrad);
            out.prev.add(this);
            out.prev.add(other);
            out.op = "/";

            if (out.requiresGrad) {
                out.backward = (v) -> {
                    if (this.requiresGrad) {
                        for (int i = 0; i < size; i++) {
                            this.grad[i] += out.grad[i] / other.data[i];
                        }
                    }
                    if (other.requiresGrad) {
                        for (int i = 0; i < size; i++) {
                            other.grad[i] += -out.grad[i] * data[i] / (other.data[i] * other.data[i]);
                        }
                    }
                    return null;
                };
            }

            return out;
        }

        throw new IllegalArgumentException("Incompatible shapes for division: " +
                Arrays.toString(shape) + " / " + Arrays.toString(other.shape));
    }

    // Negation
    public Tensor neg() {
        double[] result = new double[size];
        for (int i = 0; i < size; i++) {
            result[i] = -data[i];
        }

        Tensor out = new Tensor(result, shape, requiresGrad);
        out.prev.add(this);
        out.op = "neg";

        if (out.requiresGrad) {
            out.backward = (v) -> {
                for (int i = 0; i < size; i++) {
                    this.grad[i] -= out.grad[i];
                }
                return null;
            };
        }

        return out;
    }

    // Mean
    public Tensor mean() {
        double sum = 0.0;
        for (double val : data) {
            sum += val;
        }
        double meanVal = sum / size;

        Tensor out = new Tensor(new double[]{meanVal}, new int[]{1}, requiresGrad);
        out.prev.add(this);
        out.op = "mean";

        if (out.requiresGrad) {
            out.backward = (v) -> {
                for (int i = 0; i < size; i++) {
                    this.grad[i] += out.grad[0] / size;
                }
                return null;
            };
        }

        return out;
    }

    // Backpropagation
    public void backward() {
        Arrays.fill(grad, 1.0);

        List<Tensor> topo = new ArrayList<>();
        Set<Tensor> visited = new HashSet<>();
        buildTopo(this, visited, topo);
        Collections.reverse(topo);

        for (Tensor t : topo) {
            if (t.backward != null) {
                t.backward.apply(null);
            }
        }
    }

    private void buildTopo(Tensor tensor, Set<Tensor> visited, List<Tensor> topo) {
        if (!visited.contains(tensor)) {
            visited.add(tensor);
            for (Tensor child : tensor.prev) {
                buildTopo(child, visited, topo);
            }
            topo.add(tensor);
        }
    }

    // Zero gradients
    public void zeroGrad() {
        Arrays.fill(grad, 0.0);
    }

    // Get item
    public double item() {
        if (size != 1) {
            throw new IllegalArgumentException("Can only call item() on single-element tensors");
        }
        return data[0];
    }

    // Getters
    public double[] getData() { return data; }
    public int[] getShape() { return shape; }
    public double[] getGrad() { return grad; }
    public int getSize() { return size; }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder("Tensor(shape=" + Arrays.toString(shape) + ", [");
        int limit = Math.min(10, size);
        for (int i = 0; i < limit; i++) {
            sb.append(String.format("%.4f", data[i]));
            if (i < limit - 1) sb.append(", ");
        }
        if (size > limit) sb.append(", ...");
        sb.append("])");
        return sb.toString();
    }

    /**
     * GATHER operation - Selects elements maintaining gradient flow
     *
     * Input: this is (batch, vocab), indices is (batch,)
     * Output: (batch,) where result[i] = this[i, indices[i]]
     *
     * Critical for cross-entropy loss!
     */
    public Tensor gather(Tensor indices) {
        if (this.shape.length != 2) {
            throw new IllegalArgumentException("Gather requires 2D input tensor");
        }
        if (indices.shape.length != 1) {
            throw new IllegalArgumentException("Gather requires 1D indices");
        }

        int batchSize = this.shape[0];
        int numClasses = this.shape[1];

        if (indices.shape[0] != batchSize) {
            throw new IllegalArgumentException("Batch size mismatch");
        }

        double[] result = new double[batchSize];
        double[] indicesData = indices.getData();

        // Forward: result[i] = this[i, indices[i]]
        for (int i = 0; i < batchSize; i++) {
            int classIdx = (int) indicesData[i];
            result[i] = this.data[i * numClasses + classIdx];
        }

        Tensor out = new Tensor(result, new int[]{batchSize}, requiresGrad);
        out.prev.add(this);
        out.prev.add(indices);
        out.op = "gather";

        if (out.requiresGrad) {
            final int fNumClasses = numClasses;
            out.backward = (v) -> {
                if (this.requiresGrad) {
                    // Backward: this.grad[i, indices[i]] += out.grad[i]
                    for (int i = 0; i < batchSize; i++) {
                        int classIdx = (int) indicesData[i];
                        this.grad[i * fNumClasses + classIdx] += out.grad[i];
                    }
                }
                return null;
            };
        }

        return out;
    }

    /**
     * Mean along dimension 0 (over batch)
     * Input: (batch, features)
     * Output: (features,)
     */
    public Tensor mean(int dim) {
        if (dim != 0 || shape.length != 2) {
            throw new IllegalArgumentException("Only supports mean over dim=0 for 2D tensors");
        }

        int batchSize = shape[0];
        int features = shape[1];

        double[] result = new double[features];
        for (int j = 0; j < features; j++) {
            double sum = 0.0;
            for (int i = 0; i < batchSize; i++) {
                sum += data[i * features + j];
            }
            result[j] = sum / batchSize;
        }

        Tensor out = new Tensor(result, new int[]{features}, requiresGrad);
        out.prev.add(this);
        out.op = "mean0";

        if (out.requiresGrad) {
            final int fBatch = batchSize;
            final int fFeat = features;
            out.backward = (v) -> {
                for (int i = 0; i < fBatch; i++) {
                    for (int j = 0; j < fFeat; j++) {
                        this.grad[i * fFeat + j] += out.grad[j] / fBatch;
                    }
                }
                return null;
            };
        }

        return out;
    }

    /**
     * Variance along dimension 0 (over batch)
     * Input: (batch, features)
     * Output: (features,)
     */
    public Tensor variance(int dim) {
        if (dim != 0 || shape.length != 2) {
            throw new IllegalArgumentException("Only supports variance over dim=0 for 2D tensors");
        }

        int batchSize = shape[0];
        int features = shape[1];

        // First calculate mean
        double[] means = new double[features];
        for (int j = 0; j < features; j++) {
            double sum = 0.0;
            for (int i = 0; i < batchSize; i++) {
                sum += data[i * features + j];
            }
            means[j] = sum / batchSize;
        }

        // Then calculate variance
        double[] result = new double[features];
        for (int j = 0; j < features; j++) {
            double sumSq = 0.0;
            for (int i = 0; i < batchSize; i++) {
                double diff = data[i * features + j] - means[j];
                sumSq += diff * diff;
            }
            result[j] = sumSq / batchSize;
        }

        Tensor out = new Tensor(result, new int[]{features}, requiresGrad);
        out.prev.add(this);
        out.op = "var0";

        if (out.requiresGrad) {
            final int fBatch = batchSize;
            final int fFeat = features;
            final double[] fMeans = means;
            out.backward = (v) -> {
                // ∂var/∂x = 2(x - mean) / N
                for (int i = 0; i < fBatch; i++) {
                    for (int j = 0; j < fFeat; j++) {
                        double diff = this.data[i * fFeat + j] - fMeans[j];
                        this.grad[i * fFeat + j] += 2.0 * diff / fBatch * out.grad[j];
                    }
                }
                return null;
            };
        }

        return out;
    }

    /**
     * Subtraction with broadcasting
     *
     * Supports:
     * 1. (batch, features) - (batch, features) - element-wise
     * 2. (batch, features) - (features,) - broadcast along batch
     * 3. (batch, features) - (batch, 1) - broadcast along features
     *
     * @param other Tensor to subtract
     * @return Result of subtraction
     */
    public Tensor subtract(Tensor other) {
        // Case 1: Same shape - element-wise subtraction
        if (Arrays.equals(shape, other.shape)) {
            double[] result = new double[size];
            for (int i = 0; i < size; i++) {
                result[i] = data[i] - other.data[i];
            }

            Tensor out = new Tensor(result, shape, requiresGrad || other.requiresGrad);
            out.prev.add(this);
            out.prev.add(other);
            out.op = "-";

            if (out.requiresGrad) {
                out.backward = (v) -> {
                    if (this.requiresGrad) {
                        for (int i = 0; i < size; i++) {
                            this.grad[i] += out.grad[i];
                        }
                    }
                    if (other.requiresGrad) {
                        for (int i = 0; i < size; i++) {
                            other.grad[i] -= out.grad[i];  // Note: minus!
                        }
                    }
                    return null;
                };
            }

            return out;
        }

        // Case 2: (batch, features) - (features,) - broadcast along batch
        if (shape.length == 2 && other.shape.length == 1 && shape[1] == other.shape[0]) {
            int batchSize = shape[0];
            int features = shape[1];

            double[] result = new double[size];
            for (int i = 0; i < batchSize; i++) {
                for (int j = 0; j < features; j++) {
                    result[i * features + j] = data[i * features + j] - other.data[j];
                }
            }

            Tensor out = new Tensor(result, shape, requiresGrad || other.requiresGrad);
            out.prev.add(this);
            out.prev.add(other);
            out.op = "-broadcast";

            if (out.requiresGrad) {
                final int fBatch = batchSize;
                final int fFeat = features;
                out.backward = (v) -> {
                    if (this.requiresGrad) {
                        // Gradient for (batch, features) tensor
                        for (int i = 0; i < size; i++) {
                            this.grad[i] += out.grad[i];
                        }
                    }
                    if (other.requiresGrad) {
                        // Gradient for (features,) tensor - sum over batch dimension
                        // Note: MINUS because d/dx(a - b) = 1 for a, -1 for b
                        for (int i = 0; i < fBatch; i++) {
                            for (int j = 0; j < fFeat; j++) {
                                other.grad[j] -= out.grad[i * fFeat + j];
                            }
                        }
                    }
                    return null;
                };
            }

            return out;
        }

        // Case 3: (batch, features) - (batch, 1) - broadcast along features
        if (shape.length == 2 && other.shape.length == 2 &&
                shape[0] == other.shape[0] && other.shape[1] == 1) {

            int batchSize = shape[0];
            int features = shape[1];

            double[] result = new double[size];
            for (int i = 0; i < batchSize; i++) {
                for (int j = 0; j < features; j++) {
                    result[i * features + j] = data[i * features + j] - other.data[i];
                }
            }

            Tensor out = new Tensor(result, shape, requiresGrad || other.requiresGrad);
            out.prev.add(this);
            out.prev.add(other);
            out.op = "-broadcast_col";

            if (out.requiresGrad) {
                final int fBatch = batchSize;
                final int fFeat = features;
                out.backward = (v) -> {
                    if (this.requiresGrad) {
                        for (int i = 0; i < size; i++) {
                            this.grad[i] += out.grad[i];
                        }
                    }
                    if (other.requiresGrad) {
                        // Sum over features dimension with minus sign
                        for (int i = 0; i < fBatch; i++) {
                            for (int j = 0; j < fFeat; j++) {
                                other.grad[i] -= out.grad[i * fFeat + j];
                            }
                        }
                    }
                    return null;
                };
            }

            return out;
        }

        throw new IllegalArgumentException(
                "Unsupported shapes for subtract: " + Arrays.toString(shape) +
                        " - " + Arrays.toString(other.shape));
    }

    /**
     * Square root element-wise
     */
    public Tensor sqrt() {
        double[] result = new double[size];
        for (int i = 0; i < size; i++) {
            result[i] = Math.sqrt(data[i]);
        }

        Tensor out = new Tensor(result, shape, requiresGrad);
        out.prev.add(this);
        out.op = "sqrt";

        if (out.requiresGrad) {
            out.backward = (v) -> {
                // ∂sqrt(x)/∂x = 1 / (2*sqrt(x))
                for (int i = 0; i < size; i++) {
                    this.grad[i] += out.grad[i] / (2.0 * out.data[i]);
                }
                return null;
            };
        }

        return out;
    }

    /**
     * Element-wise multiplication with broadcasting
     *
     * Supports:
     * 1. (batch, features) * (batch, features) - element-wise
     * 2. (batch, features) * (features,) - broadcast along batch
     * 3. (features,) * (batch, features) - broadcast (commutative)
     *
     * @param other Tensor to multiply with
     * @return Result of multiplication
     */
    public Tensor mul(Tensor other) {
        // Case 1: Same shape - element-wise multiplication
        if (Arrays.equals(shape, other.shape)) {
            double[] result = new double[size];
            for (int i = 0; i < size; i++) {
                result[i] = data[i] * other.data[i];
            }

            Tensor out = new Tensor(result, shape, requiresGrad || other.requiresGrad);
            out.prev.add(this);
            out.prev.add(other);
            out.op = "*";

            if (out.requiresGrad) {
                out.backward = (v) -> {
                    if (this.requiresGrad) {
                        for (int i = 0; i < size; i++) {
                            this.grad[i] += other.data[i] * out.grad[i];
                        }
                    }
                    if (other.requiresGrad) {
                        for (int i = 0; i < size; i++) {
                            other.grad[i] += this.data[i] * out.grad[i];
                        }
                    }
                    return null;
                };
            }

            return out;
        }

        // Case 2: (batch, features) * (features,) - broadcast along batch
        if (shape.length == 2 && other.shape.length == 1 && shape[1] == other.shape[0]) {
            int batchSize = shape[0];
            int features = shape[1];

            double[] result = new double[size];
            for (int i = 0; i < batchSize; i++) {
                for (int j = 0; j < features; j++) {
                    result[i * features + j] = data[i * features + j] * other.data[j];
                }
            }

            Tensor out = new Tensor(result, shape, requiresGrad || other.requiresGrad);
            out.prev.add(this);
            out.prev.add(other);
            out.op = "*broadcast";

            if (out.requiresGrad) {
                final int fBatch = batchSize;
                final int fFeat = features;
                out.backward = (v) -> {
                    if (this.requiresGrad) {
                        // Gradient for (batch, features) tensor
                        for (int i = 0; i < fBatch; i++) {
                            for (int j = 0; j < fFeat; j++) {
                                this.grad[i * fFeat + j] += other.data[j] * out.grad[i * fFeat + j];
                            }
                        }
                    }
                    if (other.requiresGrad) {
                        // Gradient for (features,) tensor - sum over batch dimension
                        for (int i = 0; i < fBatch; i++) {
                            for (int j = 0; j < fFeat; j++) {
                                other.grad[j] += this.data[i * fFeat + j] * out.grad[i * fFeat + j];
                            }
                        }
                    }
                    return null;
                };
            }

            return out;
        }

        // Case 3: (features,) * (batch, features) - swap and use Case 2
        if (shape.length == 1 && other.shape.length == 2 && other.shape[1] == shape[0]) {
            return other.mul(this);  // Commutative, so swap
        }

        throw new IllegalArgumentException(
                "Unsupported shapes for mul: " + Arrays.toString(shape) +
                        " * " + Arrays.toString(other.shape));
    }

    /**
     * Add a scalar to all elements
     *
     * @param scalar Value to add
     * @return New tensor with scalar added
     */
    public Tensor add(double scalar) {
        double[] result = new double[size];
        for (int i = 0; i < size; i++) {
            result[i] = data[i] + scalar;
        }

        Tensor out = new Tensor(result, shape, requiresGrad);
        out.prev.add(this);
        out.op = "+scalar";

        if (out.requiresGrad) {
            out.backward = (v) -> {
                for (int i = 0; i < size; i++) {
                    this.grad[i] += out.grad[i];
                }
                return null;
            };
        }

        return out;
    }

    /**
     * Softmax activation along a dimension
     *
     * softmax(x)[i] = exp(x[i]) / sum(exp(x))
     *
     * @param dim Dimension to apply softmax (typically 1 for batch processing)
     * @return Softmax probabilities
     */
    public Tensor softmax(int dim) {
        if (dim != 1 || shape.length != 2) {
            throw new IllegalArgumentException("Only supports softmax along dim=1 for 2D tensors");
        }

        int batchSize = shape[0];
        int features = shape[1];

        double[] result = new double[size];

        for (int i = 0; i < batchSize; i++) {
            // Find max for numerical stability
            double max = Double.NEGATIVE_INFINITY;
            for (int j = 0; j < features; j++) {
                double val = data[i * features + j];
                if (val > max) max = val;
            }

            // Compute exp(x - max)
            double sum = 0.0;
            for (int j = 0; j < features; j++) {
                double val = Math.exp(data[i * features + j] - max);
                result[i * features + j] = val;
                sum += val;
            }

            // Normalize
            for (int j = 0; j < features; j++) {
                result[i * features + j] /= sum;
            }
        }

        // Note: softmax for backprop doesn't need gradients
        return new Tensor(result, shape, false);
    }

    /**
     * Create a copy of this tensor
     *
     * @return New tensor with copied data
     */
    public Tensor copy() {
        return new Tensor(data.clone(), shape.clone(), requiresGrad);
    }

    /**
     * Element-wise power
     *
     * @param exponent Power to raise elements to
     * @return New tensor with each element raised to exponent
     */
    public Tensor pow(double exponent) {
        double[] result = new double[size];
        for (int i = 0; i < size; i++) {
            result[i] = Math.pow(data[i], exponent);
        }

        Tensor out = new Tensor(result, shape, requiresGrad);
        out.prev.add(this);
        out.op = "pow";

        if (out.requiresGrad) {
            out.backward = (v) -> {
                // d/dx x^n = n * x^(n-1)
                for (int i = 0; i < size; i++) {
                    this.grad[i] += exponent * Math.pow(this.data[i], exponent - 1) * out.grad[i];
                }
                return null;
            };
        }

        return out;
    }

    /**
     * Transpose a 2D tensor
     *
     * @return Transposed tensor
     */
    public Tensor transpose() {
        if (shape.length != 2) {
            throw new IllegalArgumentException("Transpose only works on 2D tensors");
        }

        int rows = shape[0];
        int cols = shape[1];

        double[] result = new double[size];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result[j * rows + i] = data[i * cols + j];
            }
        }

        Tensor out = new Tensor(result, new int[]{cols, rows}, requiresGrad);
        out.prev.add(this);
        out.op = "transpose";

        if (out.requiresGrad) {
            out.backward = (v) -> {
                // Gradient of transpose is transpose of gradient
                for (int i = 0; i < rows; i++) {
                    for (int j = 0; j < cols; j++) {
                        this.grad[i * cols + j] += out.grad[j * rows + i];
                    }
                }
                return null;
            };
        }

        return out;
    }

    /**
     * Sum along dimension 0 (over batch)
     * Input: (batch, features)
     * Output: (features,)
     *
     * Already implemented in previous lectures, but ensuring it's here
     */
    public Tensor sum(int dim) {
        if (dim != 0 || shape.length != 2) {
            throw new IllegalArgumentException("Only supports sum over dim=0 for 2D tensors");
        }

        int batchSize = shape[0];
        int features = shape[1];

        double[] result = new double[features];
        for (int j = 0; j < features; j++) {
            double sum = 0.0;
            for (int i = 0; i < batchSize; i++) {
                sum += data[i * features + j];
            }
            result[j] = sum;
        }

        Tensor out = new Tensor(result, new int[]{features}, requiresGrad);
        out.prev.add(this);
        out.op = "sum0";

        if (out.requiresGrad) {
            final int fBatch = batchSize;
            final int fFeat = features;
            out.backward = (v) -> {
                for (int i = 0; i < fBatch; i++) {
                    for (int j = 0; j < fFeat; j++) {
                        this.grad[i * fFeat + j] += out.grad[j];
                    }
                }
                return null;
            };
        }

        return out;
    }

    /**
     * Divide by a scalar (for mean)
     *
     * @param scalar Value to divide by
     * @return New tensor with all elements divided by scalar
     */
    public Tensor div(double scalar) {
        double[] result = new double[size];
        for (int i = 0; i < size; i++) {
            result[i] = data[i] / scalar;
        }

        Tensor out = new Tensor(result, shape, requiresGrad);
        out.prev.add(this);
        out.op = "/scalar";

        if (out.requiresGrad) {
            out.backward = (v) -> {
                for (int i = 0; i < size; i++) {
                    this.grad[i] += out.grad[i] / scalar;
                }
                return null;
            };
        }

        return out;
    }

    /**
     * Multiply by a scalar
     *
     * @param scalar Value to multiply by
     * @return New tensor with all elements multiplied by scalar
     */
    public Tensor mul(double scalar) {
        double[] result = new double[size];
        for (int i = 0; i < size; i++) {
            result[i] = data[i] * scalar;
        }

        Tensor out = new Tensor(result, shape, requiresGrad);
        out.prev.add(this);
        out.op = "*scalar";

        if (out.requiresGrad) {
            out.backward = (v) -> {
                for (int i = 0; i < size; i++) {
                    this.grad[i] += out.grad[i] * scalar;
                }
                return null;
            };
        }

        return out;
    }

    /**
     * Reciprocal (1/x) element-wise
     */
    public Tensor reciprocal() {
        double[] result = new double[size];
        for (int i = 0; i < size; i++) {
            result[i] = 1.0 / data[i];
        }

        Tensor out = new Tensor(result, shape, requiresGrad);
        out.prev.add(this);
        out.op = "reciprocal";

        if (out.requiresGrad) {
            out.backward = (v) -> {
                // d/dx (1/x) = -1/x²
                for (int i = 0; i < size; i++) {
                    this.grad[i] += -out.grad[i] / (this.data[i] * this.data[i]);
                }
                return null;
            };
        }

        return out;
    }

}