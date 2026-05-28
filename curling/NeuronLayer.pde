class NeuronLayer {
    Neuron[] neurons;
    float[] lastOutputs;

    NeuronLayer(int neuronCount, int inputCountPerNeuron) {
        this(neuronCount, inputCountPerNeuron, ActivationType.TANH);
    }

    NeuronLayer(int neuronCount, int inputCountPerNeuron, ActivationType activationType) {
        neurons = new Neuron[neuronCount];
        for (int i = 0; i < neuronCount; i++) {
            neurons[i] = new Neuron(inputCountPerNeuron, activationType);
        }
    }

    // Backward compat for evolution code paths.
    NeuronLayer(int neuronCount, int inputCountPerNeuron, boolean linear) {
        this(neuronCount, inputCountPerNeuron,
            linear ? ActivationType.LINEAR : ActivationType.TANH);
    }

    float[] forward(float[] inputs) {
        float[] outputs = new float[neurons.length];
        for (int i = 0; i < neurons.length; i++) {
            outputs[i] = neurons[i].forward(inputs);
        }
        lastOutputs = outputs;
        return outputs;
    }

    float[] feedForward(float[] inputs) {
        return forward(inputs);
    }

    float[] backward(float[] outputGrads) {
        int inputSize = neurons[0].lastInputs.length;
        float[] inputGrads = new float[inputSize];

        for (int j = 0; j < neurons.length; j++) {
            Neuron n = neurons[j];
            float grad = outputGrads[j] * n.activationDerivative();
            n.biasGrad += grad;
            for (int i = 0; i < n.weights.length; i++) {
                n.weightGrads[i] += grad * n.lastInputs[i];
                inputGrads[i] += grad * n.weights[i];
            }
        }
        return inputGrads;
    }

    void zeroGrads() {
        for (Neuron neuron : neurons) {
            neuron.zeroGrads();
        }
    }

    void applyGrads(float learningRate, float weightDecay) {
        applyGrads(learningRate, weightDecay, weightDecay);
    }

    void applyGrads(float learningRate, float weightDecay, float biasDecay) {
        for (Neuron neuron : neurons) {
            neuron.applyGrads(learningRate, weightDecay, biasDecay);
        }
    }

    void scaleGrads(float factor) {
        for (Neuron neuron : neurons) {
            neuron.scaleGrads(factor);
        }
    }

    void clipGrads(float maxAbs) {
        for (Neuron neuron : neurons) {
            neuron.clipGrads(maxAbs);
        }
    }

    boolean hasFiniteGrads() {
        for (Neuron neuron : neurons) {
            if (!neuron.hasFiniteGrads()) return false;
        }
        return true;
    }

    boolean hasFiniteWeights() {
        for (Neuron neuron : neurons) {
            if (!neuron.hasFiniteWeights()) return false;
        }
        return true;
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
        ActivationType act = neurons[0].activationType;
        NeuronLayer clone = new NeuronLayer(neurons.length, neurons[0].weights.length, act);
        for (int i = 0; i < neurons.length; i++) {
            clone.neurons[i] = neurons[i].copy();
        }
        return clone;
    }
}
