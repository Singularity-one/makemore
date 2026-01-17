package com.makemore.bigram;

import java.util.*;
import java.util.function.BiFunction;
import java.util.function.Function;

/**
 * Tensor with automatic differentiation support
 * Inspired by PyTorch tensors and the vectorized approach from java-micrograd
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

    public static Tensor randn(int... shape) {
        return randn(new Random(), shape);
    }

    public static Tensor randn(Random rng, int... shape) {
        int size = calculateSize(shape);
        double[] data = new double[size];
        for (int i = 0; i < size; i++) {
            data[i] = rng.nextGaussian();
        }
        return new Tensor(data, shape, false);
    }

    public static Tensor fromValue(double value, int... shape) {
        int size = calculateSize(shape);
        double[] data = new double[size];
        Arrays.fill(data, value);
        return new Tensor(data, shape);
    }

    // Helper to calculate total size
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

    // Get element (for 1D or 2D tensors)
    public double get(int i) {
        return data[i];
    }

    public double get(int i, int j) {
        if (shape.length != 2) {
            throw new IllegalArgumentException("Tensor is not 2D");
        }
        return data[i * shape[1] + j];
    }

    // Set element
    public void set(int i, double value) {
        data[i] = value;
    }

    public void set(int i, int j, double value) {
        if (shape.length != 2) {
            throw new IllegalArgumentException("Tensor is not 2D");
        }
        data[i * shape[1] + j] = value;
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
                    sum += get(i, k) * other.get(k, j);
                }
                result[i * p + j] = sum;
            }
        }

        Tensor out = new Tensor(result, new int[]{m, p}, requiresGrad || other.requiresGrad);
        out.prev.add(this);
        out.prev.add(other);
        out.op = "@";

        if (out.requiresGrad) {
            out.backward = (v) -> {
                // Gradient w.r.t. this: grad @ other.T
                if (this.requiresGrad) {
                    for (int i = 0; i < m; i++) {
                        for (int k = 0; k < n; k++) {
                            for (int j = 0; j < p; j++) {
                                this.grad[i * n + k] += out.grad[i * p + j] * other.data[k * p + j];
                            }
                        }
                    }
                }

                // Gradient w.r.t. other: this.T @ grad
                if (other.requiresGrad) {
                    for (int k = 0; k < n; k++) {
                        for (int j = 0; j < p; j++) {
                            for (int i = 0; i < m; i++) {
                                other.grad[k * p + j] += this.data[i * n + k] * out.grad[i * p + j];
                            }
                        }
                    }
                }
                return null;
            };
        }

        return out;
    }

    // Element-wise exponential
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

    // Element-wise logarithm
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

    // Sum along dimension with keepdim
    public Tensor sum(int dim, boolean keepdim) {
        if (shape.length != 2) {
            throw new IllegalArgumentException("Only 2D tensors supported for sum");
        }

        if (dim == 0) {
            // Sum along rows (result is 1 x cols)
            double[] result = new double[shape[1]];
            for (int j = 0; j < shape[1]; j++) {
                for (int i = 0; i < shape[0]; i++) {
                    result[j] += get(i, j);
                }
            }
            int[] outShape = keepdim ? new int[]{1, shape[1]} : new int[]{shape[1]};

            Tensor out = new Tensor(result, outShape, requiresGrad);
            out.prev.add(this);
            out.op = "sum0";

            if (out.requiresGrad) {
                out.backward = (v) -> {
                    for (int i = 0; i < shape[0]; i++) {
                        for (int j = 0; j < shape[1]; j++) {
                            this.grad[i * shape[1] + j] += out.grad[j];
                        }
                    }
                    return null;
                };
            }

            return out;
        } else if (dim == 1) {
            // Sum along columns (result is rows x 1)
            double[] result = new double[shape[0]];
            for (int i = 0; i < shape[0]; i++) {
                for (int j = 0; j < shape[1]; j++) {
                    result[i] += get(i, j);
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

        throw new IllegalArgumentException("Invalid dimension: " + dim);
    }

    // Division (element-wise or broadcasting)
    public Tensor div(Tensor other) {
        double[] result = new double[size];

        // Handle broadcasting for (m, n) / (m, 1)
        if (shape.length == 2 && other.shape.length == 2 &&
                shape[0] == other.shape[0] && other.shape[1] == 1) {

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
                    if (other.shape[1] == 1 && shape[1] > 1) {
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
                    if (other.shape[1] == 1 && shape[1] > 1) {
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

    // Addition
    public Tensor add(Tensor other) {
        if (!Arrays.equals(shape, other.shape)) {
            throw new IllegalArgumentException("Shapes must match for addition");
        }

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

    // Scalar multiplication
    public Tensor mul(double scalar) {
        double[] result = new double[size];
        for (int i = 0; i < size; i++) {
            result[i] = data[i] * scalar;
        }

        Tensor out = new Tensor(result, shape, requiresGrad);
        out.prev.add(this);
        out.op = "*" + scalar;

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

    // Power
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
                for (int i = 0; i < size; i++) {
                    this.grad[i] += out.grad[i] * exponent * Math.pow(data[i], exponent - 1);
                }
                return null;
            };
        }

        return out;
    }

    // Backpropagation
    public void backward() {
        // Initialize gradient to 1 for the output
        Arrays.fill(grad, 1.0);

        // Topological sort
        List<Tensor> topo = new ArrayList<>();
        Set<Tensor> visited = new HashSet<>();
        buildTopo(this, visited, topo);
        Collections.reverse(topo);

        // Backward pass
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

    // Get item (for indexing specific rows)
    public double item() {
        if (size != 1) {
            throw new IllegalArgumentException("Can only call item() on tensors with 1 element");
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
        if (shape.length == 1) {
            return "Tensor(" + Arrays.toString(data) + ")";
        } else if (shape.length == 2) {
            StringBuilder sb = new StringBuilder("Tensor([\n");
            for (int i = 0; i < shape[0]; i++) {
                sb.append("  [");
                for (int j = 0; j < shape[1]; j++) {
                    sb.append(String.format("%.4f", get(i, j)));
                    if (j < shape[1] - 1) sb.append(", ");
                }
                sb.append("]\n");
            }
            sb.append("])");
            return sb.toString();
        }
        return "Tensor(shape=" + Arrays.toString(shape) + ", size=" + size + ")";
    }
}