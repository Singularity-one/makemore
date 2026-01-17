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

    // Add with broadcasting (for bias)
    public Tensor add(Tensor other) {
        // Handle broadcasting: (m, n) + (n,) or (m, n) + (m, n)
        double[] result = new double[size];

        if (Arrays.equals(shape, other.shape)) {
            // Element-wise addition
            for (int i = 0; i < size; i++) {
                result[i] = data[i] + other.data[i];
            }
        } else if (shape.length == 2 && other.shape.length == 1 && shape[1] == other.shape[0]) {
            // Broadcasting: (m, n) + (n,)
            int m = shape[0];
            int n = shape[1];
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < n; j++) {
                    result[i * n + j] = data[i * n + j] + other.data[j];
                }
            }
        } else {
            throw new IllegalArgumentException("Incompatible shapes for add: " +
                    Arrays.toString(shape) + " + " + Arrays.toString(other.shape));
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
                    if (Arrays.equals(shape, other.shape)) {
                        for (int i = 0; i < other.size; i++) {
                            other.grad[i] += out.grad[i];
                        }
                    } else if (shape.length == 2 && other.shape.length == 1) {
                        // Sum over batch dimension
                        int m = shape[0];
                        int n = shape[1];
                        for (int i = 0; i < m; i++) {
                            for (int j = 0; j < n; j++) {
                                other.grad[j] += out.grad[i * n + j];
                            }
                        }
                    }
                }
                return null;
            };
        }

        return out;
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
    public Tensor div(Tensor other) {
        double[] result = new double[size];

        if (shape.length == 2 && other.shape.length == 2 &&
                shape[0] == other.shape[0] && other.shape[1] == 1) {
            // Broadcasting: (m, n) / (m, 1)
            for (int i = 0; i < shape[0]; i++) {
                for (int j = 0; j < shape[1]; j++) {
                    result[i * shape[1] + j] = data[i * shape[1] + j] / other.data[i];
                }
            }
        } else if (Arrays.equals(shape, other.shape)) {
            // Element-wise division
            for (int i = 0; i < size; i++) {
                result[i] = data[i] / other.data[i];
            }
        } else {
            throw new IllegalArgumentException("Incompatible shapes for division");
        }

        Tensor out = new Tensor(result, shape, requiresGrad || other.requiresGrad);
        out.prev.add(this);
        out.prev.add(other);
        out.op = "/";

        if (out.requiresGrad) {
            out.backward = (v) -> {
                if (this.requiresGrad) {
                    if (other.shape.length == 2 && other.shape[1] == 1 && shape.length == 2) {
                        for (int i = 0; i < shape[0]; i++) {
                            for (int j = 0; j < shape[1]; j++) {
                                this.grad[i * shape[1] + j] += out.grad[i * shape[1] + j] / other.data[i];
                            }
                        }
                    } else {
                        for (int i = 0; i < size; i++) {
                            this.grad[i] += out.grad[i] / other.data[i];
                        }
                    }
                }

                if (other.requiresGrad) {
                    if (other.shape.length == 2 && other.shape[1] == 1 && shape.length == 2) {
                        for (int i = 0; i < shape[0]; i++) {
                            for (int j = 0; j < shape[1]; j++) {
                                other.grad[i] += -out.grad[i * shape[1] + j] *
                                        data[i * shape[1] + j] / (other.data[i] * other.data[i]);
                            }
                        }
                    } else {
                        for (int i = 0; i < size; i++) {
                            other.grad[i] += -out.grad[i] * data[i] / (other.data[i] * other.data[i]);
                        }
                    }
                }
                return null;
            };
        }

        return out;
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
}