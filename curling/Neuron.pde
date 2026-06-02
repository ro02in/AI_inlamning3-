// Activation function choice for a Neuron / NeuronLayer.
// - TANH:   bounded [-1, 1], smooth, can saturate at the edges.
// - RELU:   leaky ReLU (see RELU_LEAK). Small negative slope keeps units recoverable.
// - LINEAR: identity. Used when an output should not be squashed.
enum ActivationKind { TANH, RELU, LINEAR }
enum NormKind { NONE, LAYERNORM }

class Neuron {
    static final float WEIGHT_CLIP = 1.0f;
    static final float RELU_LEAK = 0.01f;
    // Activations below this count as inactive in diagnostics / training penalty.
    static final float RELU_INACTIVE_THRESHOLD = 0.05f;

    float[] weights;
    float bias;
    ActivationKind activation;

    Neuron(int inputCount) {
        this(inputCount, ActivationKind.TANH);
    }

    // Backwards-compatible: false = TANH, true = LINEAR.
    Neuron(int inputCount, boolean linear) {
        this(inputCount, linear ? ActivationKind.LINEAR : ActivationKind.TANH);
    }

    Neuron(int inputCount, ActivationKind activation) {
        this.activation = activation;
        weights = new float[inputCount];
        // He init for ReLU (variance 2/n), Xavier-ish for symmetric activations (variance 1/n).
        float scale = (activation == ActivationKind.RELU)
                    ? sqrt(2.0f / inputCount)
                    : 1.0f / sqrt(inputCount);
        for (int i = 0; i < inputCount; i++) {
            weights[i] = random(-1, 1) * scale;
        }
        // ReLU: start slightly positive so most units fire on typical sparse state vectors.
        bias = (activation == ActivationKind.RELU)
            ? random(0.05f, 0.35f)
            : random(-0.3f, 0.3f);
    }

    float activate(float[] inputs) {
        float sum = bias;
        for (int i = 0; i < weights.length; i++) {
            sum += weights[i] * inputs[i];
        }
        switch (activation) {
            case TANH:   return (float) Math.tanh(sum);
            case RELU:   return sum > 0 ? sum : RELU_LEAK * sum;
            case LINEAR: return sum;
            default:     return sum;
        }
    }

    // Derivative of the activation function at a given pre-activation value z.
    // Used during backpropagation: dL/dz = dL/da * da/dz.
    float activationDerivative(float z) {
        switch (activation) {
            case TANH: {
                float t = (float) Math.tanh(z);
                return 1.0f - t * t;
            }
            case RELU:   return z > 0 ? 1.0f : RELU_LEAK;
            case LINEAR: return 1.0f;
            default:     return 1.0f;
        }
    }

    void clip() {
        for (int i = 0; i < weights.length; i++) {
            weights[i] = constrain(weights[i], -WEIGHT_CLIP, WEIGHT_CLIP);
        }
        bias = constrain(bias, -WEIGHT_CLIP, WEIGHT_CLIP);
    }

    void appendSaveLine(StringBuilder sb) {
        sb.append(bias);
        for (int i = 0; i < weights.length; i++) {
            sb.append(' ').append(weights[i]);
        }
        sb.append('\n');
    }

    Neuron createFromLine(String line) {
        String[] parts = split(trim(line), ' ');
        if (parts.length < 2) return null;
        int inputCount = parts.length - 1;
        Neuron n = new Neuron(inputCount, activation);
        n.bias = parseFloat(parts[0]);
        for (int i = 0; i < inputCount; i++) {
            n.weights[i] = parseFloat(parts[i + 1]);
        }
        return n;
    }
}


