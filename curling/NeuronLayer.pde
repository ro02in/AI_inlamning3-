class NeuronLayer {
    Neuron[] neurons;

    // Stored during feedForward so backward() can update weights from the same pass.
    float[] lastInput;
    float[] lastPreActivation;
    float[] lastOutput;

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
    }

    float[] feedForward(float[] inputs) {
        lastInput = inputs;
        lastPreActivation = new float[neurons.length];
        lastOutput = new float[neurons.length];

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

    // Backpropagates through this layer and applies a plain gradient update.
    // Positive gradients increase the matching output value.
    float[] backward(float[] gradOutput, float lr) {
        if (lastInput == null) return new float[neurons[0].weights.length];

        int inputLen = neurons[0].weights.length;
        float[] gradInput = new float[inputLen];

        for (int i = 0; i < neurons.length; i++) {
            Neuron n = neurons[i];
            float dz = gradOutput[i] * n.activationDerivative(lastPreActivation[i]);

            for (int j = 0; j < inputLen; j++) {
                gradInput[j] += dz * n.weights[j];
            }

            for (int j = 0; j < n.weights.length; j++) {
                n.weights[j] += lr * dz * lastInput[j];
            }
            n.bias += lr * dz;
        }
        return gradInput;
    }

    void clipAll() {
        for (Neuron neuron : neurons) {
            neuron.clip();
        }
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

    int loadSelf(String[] lines, int idx) {
        if (idx >= lines.length) return idx;
        String header = trim(lines[idx]);
        if (!header.startsWith("@layer ")) return idx;
        String[] parts = split(header, ' ');
        if (parts.length < 5) return idx + 1;

        ActivationKind act = parseActivation(parts[2]);
        int neuronCount = parseInt(parts[3]);
        int inputCount = parseInt(parts[4]);
        if (neuronCount != neurons.length || inputCount != neurons[0].weights.length) {
            NeuronLayer rebuilt = new NeuronLayer(neuronCount, inputCount, act);
            neurons = rebuilt.neurons;
        }

        idx++;
        Neuron parser = new Neuron(inputCount, act);
        for (int i = 0; i < neuronCount && idx < lines.length; i++, idx++) {
            Neuron n = parser.createFromLine(lines[idx]);
            if (n != null) neurons[i] = n;
        }
        return idx;
    }
}
