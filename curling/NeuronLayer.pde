class NeuronLayer {
    Neuron[] neurons;

    // Stored during feedForward so backward() can use them without re-running forward.
    float[] lastInput;
    float[] lastPreActivation; // pre-activation sums (z) for each neuron
    float[] lastOutput;        // post-activation values (a)

    // Adam optimizer moments: first (m) and second (v) for weights and biases.
    float[][] mW; // [neuron][weight]
    float[]   mB; // [neuron]
    float[][] vW;
    float[]   vB;
    int       adamStep = 0;

    static final float ADAM_BETA1   = 0.9f;
    static final float ADAM_BETA2   = 0.999f;
    static final float ADAM_EPSILON = 1e-8f;

    NeuronLayer(int neuronCount, int inputCountPerNeuron) {
        this(neuronCount, inputCountPerNeuron, ActivationKind.TANH);
    }

    // Backwards-compatible: false = TANH, true = LINEAR.
    NeuronLayer(int neuronCount, int inputCountPerNeuron, boolean linear) {
        this(neuronCount, inputCountPerNeuron,
             linear ? ActivationKind.LINEAR : ActivationKind.TANH);
    }

    NeuronLayer(int neuronCount, int inputCountPerNeuron, ActivationKind activation) {
        neurons = new Neuron[neuronCount];
        for (int i = 0; i < neuronCount; i++) {
            neurons[i] = new Neuron(inputCountPerNeuron, activation);
        }
        initAdam(neuronCount, inputCountPerNeuron);
    }

    void initAdam(int neuronCount, int inputCount) {
        mW = new float[neuronCount][inputCount];
        mB = new float[neuronCount];
        vW = new float[neuronCount][inputCount];
        vB = new float[neuronCount];
        adamStep = 0;
    }

    float[] feedForward(float[] inputs) {
        lastInput = inputs;
        lastPreActivation = new float[neurons.length];
        lastOutput        = new float[neurons.length];

        for (int i = 0; i < neurons.length; i++) {
            Neuron n = neurons[i];
            float z = n.bias;
            for (int j = 0; j < n.weights.length; j++) {
                z += n.weights[j] * inputs[j];
            }
            lastPreActivation[i] = z;
            switch (n.activation) {
                case TANH:   lastOutput[i] = (float) Math.tanh(z); break;
                case RELU:   lastOutput[i] = z > 0 ? z : Neuron.RELU_LEAK * z; break;
                case LINEAR: lastOutput[i] = z; break;
                default:     lastOutput[i] = z; break;
            }
        }
        return lastOutput;
    }

    // Backpropagate gradients through this layer.
    // gradOutput: dL/da for each output neuron (length = neurons.length).
    // lr: learning rate (used by Adam).
    // Returns: dL/dx (gradient w.r.t. layer input), for passing to the previous layer.
    float[] backward(float[] gradOutput, float lr) {
        if (lastInput == null) return new float[neurons[0].weights.length];

        adamStep++;
        float bc1 = 1.0f - pow(ADAM_BETA1, adamStep);
        float bc2 = 1.0f - pow(ADAM_BETA2, adamStep);

        int inputLen = neurons[0].weights.length;
        float[] gradInput = new float[inputLen];

        for (int i = 0; i < neurons.length; i++) {
            Neuron n   = neurons[i];
            float da   = gradOutput[i];
            float dz   = da * n.activationDerivative(lastPreActivation[i]);

            // Accumulate gradient w.r.t. input (for the previous layer).
            for (int j = 0; j < inputLen; j++) {
                gradInput[j] += dz * n.weights[j];
            }

            // Adam weight update.
            for (int j = 0; j < n.weights.length; j++) {
                float gw = dz * lastInput[j];
                mW[i][j] = ADAM_BETA1 * mW[i][j] + (1 - ADAM_BETA1) * gw;
                vW[i][j] = ADAM_BETA2 * vW[i][j] + (1 - ADAM_BETA2) * gw * gw;
                float mHat = mW[i][j] / bc1;
                float vHat = vW[i][j] / bc2;
                n.weights[j] += lr * mHat / (sqrt(vHat) + ADAM_EPSILON);
            }

            // Adam bias update.
            mB[i] = ADAM_BETA1 * mB[i] + (1 - ADAM_BETA1) * dz;
            vB[i] = ADAM_BETA2 * vB[i] + (1 - ADAM_BETA2) * dz * dz;
            float mBHat = mB[i] / bc1;
            float vBHat = vB[i] / bc2;
            n.bias += lr * mBHat / (sqrt(vBHat) + ADAM_EPSILON);
        }
        return gradInput;
    }

    void mutate(float mutationRate, float mutationStrength) {
        for (Neuron neuron : neurons) {
            neuron.mutate(mutationRate, mutationStrength);
        }
    }

    void clipAll() {
        for (Neuron neuron : neurons) {
            neuron.clip();
        }
    }

    NeuronLayer copy() {
        NeuronLayer clone = new NeuronLayer(neurons.length,
                                            neurons[0].weights.length,
                                            neurons[0].activation);
        for (int i = 0; i < neurons.length; i++) {
            clone.neurons[i] = neurons[i].copy();
        }
        return clone;
    }

    String activationName(ActivationKind kind) {
        switch (kind) {
            case RELU:   return "RELU";
            case LINEAR: return "LINEAR";
            default:     return "TANH";
        }
    }

    ActivationKind parseActivation(String name) {
        if (name == null) return ActivationKind.TANH;
        if (name.equals("RELU"))   return ActivationKind.RELU;
        if (name.equals("LINEAR")) return ActivationKind.LINEAR;
        return ActivationKind.TANH;
    }

    void appendSave(StringBuilder sb, String layerTag) {
        sb.append("@layer ").append(layerTag).append(' ')
          .append(activationName(neurons[0].activation)).append(' ')
          .append(neurons.length).append(' ')
          .append(neurons[0].weights.length).append('\n');
        for (Neuron n : neurons) {
            n.appendSaveLine(sb);
        }
    }

    // Load a @layer block into THIS layer object. Returns next index after layer.
    // Used by ShotTypeSelector and other non-policy owners.
    int loadSelf(String[] lines, int idx) {
        if (idx >= lines.length) return idx;
        String header = trim(lines[idx]);
        if (!header.startsWith("@layer ")) return idx;
        String[] parts = split(header, ' ');
        if (parts.length < 5) return idx + 1;
        ActivationKind act  = parseActivation(parts[2]);
        int neuronCount = parseInt(parts[3]);
        int inputCount  = parseInt(parts[4]);
        if (neuronCount != neurons.length || inputCount != neurons[0].weights.length) {
            // Rebuild if dimensions changed, also reinitialize Adam moments.
            NeuronLayer rebuilt = new NeuronLayer(neuronCount, inputCount, act);
            neurons = rebuilt.neurons;
            mW = rebuilt.mW; mB = rebuilt.mB;
            vW = rebuilt.vW; vB = rebuilt.vB;
            adamStep = 0;
        }
        idx++;
        Neuron parser = new Neuron(inputCount, act);
        for (int i = 0; i < neuronCount && idx < lines.length; i++, idx++) {
            Neuron n = parser.createFromLine(lines[idx]);
            if (n != null) neurons[i] = n;
        }
        return idx;
    }

    // Parse @layer line at lines[idx] and following neuron lines. Returns index after layer.
    int loadFromLines(String[] lines, int idx, NeuralPolicy policy,
                      String expectedTag) {
        if (idx >= lines.length) return idx;
        String header = trim(lines[idx]);
        if (!header.startsWith("@layer ")) return idx;

        String[] parts = split(header, ' ');
        if (parts.length < 5) return idx + 1;
        String tag = parts[1];
        if (!tag.equals(expectedTag)) return idx + 1;

        ActivationKind act = parseActivation(parts[2]);
        int neuronCount = parseInt(parts[3]);
        int inputCount  = parseInt(parts[4]);

        NeuronLayer layer = new NeuronLayer(neuronCount, inputCount, act);
        idx++;
        Neuron parser = new Neuron(inputCount, act);
        for (int i = 0; i < neuronCount && idx < lines.length; i++, idx++) {
            Neuron n = parser.createFromLine(lines[idx]);
            if (n != null) layer.neurons[i] = n;
        }

        if (expectedTag.equals("hidden"))  policy.hiddenLayer = layer;
        else if (expectedTag.equals("output")) policy.outputLayer = layer;
        else if (expectedTag.equals("logstd")) policy.outputLogStd = layer;
        return idx;
    }
}
