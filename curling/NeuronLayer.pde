class NeuronLayer {
    Neuron[] neurons;
    NormKind normKind = NormKind.NONE;

    // Stored during feedForward so backward() can use them without re-running forward.
    float[] lastInput;
    float[] lastRawPreActivation; // raw pre-activation sums (z) before optional norm
    float[] lastPreActivation; // activation input after optional norm
    float[] lastNormalizedPreActivation;
    float[] lastOutput;        // post-activation values (a)
    float lastNormMean = 0;
    float lastNormInvStd = 1;

    // Adam optimizer moments: first (m) and second (v) for weights and biases.
    float[][] mW; // [neuron][weight]
    float[]   mB; // [neuron]
    float[][] vW;
    float[]   vB;
    int       adamStep = 0;

    static final float ADAM_BETA1   = 0.9f;
    static final float ADAM_BETA2   = 0.999f;
    static final float ADAM_EPSILON = 1e-8f;
    static final float LAYER_NORM_EPSILON = 1e-5f;

    NeuronLayer(int neuronCount, int inputCountPerNeuron) {
        this(neuronCount, inputCountPerNeuron, ActivationKind.TANH);
    }

    // Backwards-compatible: false = TANH, true = LINEAR.
    NeuronLayer(int neuronCount, int inputCountPerNeuron, boolean linear) {
        this(neuronCount, inputCountPerNeuron,
             linear ? ActivationKind.LINEAR : ActivationKind.TANH);
    }

    NeuronLayer(int neuronCount, int inputCountPerNeuron, ActivationKind activation) {
        this(neuronCount, inputCountPerNeuron, activation, NormKind.NONE);
    }

    NeuronLayer(int neuronCount, int inputCountPerNeuron,
                ActivationKind activation, NormKind normKind) {
        this.normKind = normKind;
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
        lastRawPreActivation = new float[neurons.length];
        lastPreActivation = new float[neurons.length];
        lastNormalizedPreActivation = new float[neurons.length];
        lastOutput        = new float[neurons.length];

        for (int i = 0; i < neurons.length; i++) {
            Neuron n = neurons[i];
            float z = n.bias;
            for (int j = 0; j < n.weights.length; j++) {
                z += n.weights[j] * inputs[j];
            }
            lastRawPreActivation[i] = z;
        }

        if (normKind == NormKind.LAYERNORM) {
            float mean = 0;
            for (float z : lastRawPreActivation) mean += z;
            mean /= max(1, neurons.length);
            float variance = 0;
            for (float z : lastRawPreActivation) {
                float d = z - mean;
                variance += d * d;
            }
            variance /= max(1, neurons.length);
            lastNormMean = mean;
            lastNormInvStd = 1.0f / sqrt(variance + LAYER_NORM_EPSILON);
            for (int i = 0; i < neurons.length; i++) {
                float zn = (lastRawPreActivation[i] - mean) * lastNormInvStd;
                lastNormalizedPreActivation[i] = zn;
                lastPreActivation[i] = zn;
            }
        } else {
            lastNormMean = 0;
            lastNormInvStd = 1;
            for (int i = 0; i < neurons.length; i++) {
                lastNormalizedPreActivation[i] = lastRawPreActivation[i];
                lastPreActivation[i] = lastRawPreActivation[i];
            }
        }

        for (int i = 0; i < neurons.length; i++) {
            Neuron n = neurons[i];
            float z = lastPreActivation[i];
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
        float[] dzNorm = new float[neurons.length];
        float[] dzRaw  = new float[neurons.length];

        for (int i = 0; i < neurons.length; i++) {
            Neuron n   = neurons[i];
            float da   = gradOutput[i];
            dzNorm[i]  = da * n.activationDerivative(lastPreActivation[i]);
        }

        if (normKind == NormKind.LAYERNORM) {
            float meanGrad = 0;
            float meanGradNorm = 0;
            for (int i = 0; i < neurons.length; i++) {
                meanGrad += dzNorm[i];
                meanGradNorm += dzNorm[i] * lastNormalizedPreActivation[i];
            }
            meanGrad /= max(1, neurons.length);
            meanGradNorm /= max(1, neurons.length);
            for (int i = 0; i < neurons.length; i++) {
                dzRaw[i] = lastNormInvStd
                         * (dzNorm[i] - meanGrad
                            - lastNormalizedPreActivation[i] * meanGradNorm);
            }
        } else {
            for (int i = 0; i < neurons.length; i++) {
                dzRaw[i] = dzNorm[i];
            }
        }

        for (int i = 0; i < neurons.length; i++) {
            Neuron n   = neurons[i];
            float dz   = dzRaw[i];

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
                                            neurons[0].activation,
                                            normKind);
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

    String normName(NormKind kind) {
        return kind == NormKind.LAYERNORM ? "LAYERNORM" : "NONE";
    }

    NormKind parseNorm(String name) {
        if (name == null) return NormKind.NONE;
        if (name.equals("LAYERNORM")) return NormKind.LAYERNORM;
        return NormKind.NONE;
    }

    void appendSave(StringBuilder sb, String layerTag) {
        sb.append("@layer ").append(layerTag).append(' ')
          .append(activationName(neurons[0].activation)).append(' ')
          .append(neurons.length).append(' ')
          .append(neurons[0].weights.length).append(' ')
          .append(normName(normKind)).append('\n');
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
        NormKind norm = parts.length >= 6 ? parseNorm(parts[5]) : NormKind.NONE;
        if (neuronCount != neurons.length || inputCount != neurons[0].weights.length) {
            // Rebuild if dimensions changed, also reinitialize Adam moments.
            NeuronLayer rebuilt = new NeuronLayer(neuronCount, inputCount, act, norm);
            neurons = rebuilt.neurons;
            mW = rebuilt.mW; mB = rebuilt.mB;
            vW = rebuilt.vW; vB = rebuilt.vB;
            adamStep = 0;
        }
        normKind = norm;
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
        NormKind norm = parts.length >= 6 ? parseNorm(parts[5]) : NormKind.NONE;

        NeuronLayer layer = new NeuronLayer(neuronCount, inputCount, act, norm);
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
