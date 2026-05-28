// Activation function choice for a Neuron / NeuronLayer.
// - TANH:   bounded [-1, 1], smooth, can saturate at the edges.
// - RELU:   unbounded above, zero below. No upper saturation but can "die".
// - LINEAR: identity. Used when an output should not be squashed.
enum ActivationKind { TANH, RELU, LINEAR }

class Neuron {
    static final float WEIGHT_CLIP = 1.0f;

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
        bias = random(-0.3, 0.3);
    }

    float activate(float[] inputs) {
        float sum = bias;
        for (int i = 0; i < weights.length; i++) {
            sum += weights[i] * inputs[i];
        }
        switch (activation) {
            case TANH:   return (float) Math.tanh(sum);
            case RELU:   return max(0, sum);
            case LINEAR: return sum;
            default:     return sum;
        }
    }

    void mutate(float mutationRate, float mutationStrength) {
        for (int i = 0; i < weights.length; i++) {
            if (random(1) < mutationRate) {
                weights[i] += randomGaussian() * mutationStrength;
            }
        }
        if (random(1) < mutationRate) {
            bias += randomGaussian() * mutationStrength;
        }
        clip();
    }

    void clip() {
        for (int i = 0; i < weights.length; i++) {
            weights[i] = constrain(weights[i], -WEIGHT_CLIP, WEIGHT_CLIP);
        }
        bias = constrain(bias, -WEIGHT_CLIP, WEIGHT_CLIP);
    }

    Neuron copy() {
        Neuron clone = new Neuron(weights.length, activation);
        for (int i = 0; i < weights.length; i++) {
            clone.weights[i] = weights[i];
        }
        clone.bias = bias;
        return clone;
    }
}
