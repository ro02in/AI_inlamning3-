class NeuronLayer {
    Neuron[] neurons;

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
        float[] outputs = new float[neurons.length];
        for (int i = 0; i < neurons.length; i++) {
            outputs[i] = neurons[i].activate(inputs);
        }
        return outputs;
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
}
