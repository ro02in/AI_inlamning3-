class NeuronLayer {
    Neuron[] neurons;
    int      activation;
    float    weightClip;
    float    biasMin;
    float    biasMax;

    NeuronLayer(int neuronCount, int inputCountPerNeuron) {
        this(neuronCount, inputCountPerNeuron, Neuron.ACT_TANH, 0.4f, -0.4f, 0.4f);
    }

    NeuronLayer(int neuronCount, int inputCountPerNeuron, int activation) {
        this(neuronCount, inputCountPerNeuron, activation, 0.4f, -0.4f, 0.4f);
    }

    NeuronLayer(int neuronCount, int inputCountPerNeuron, int activation,
                float weightClip, float biasMin, float biasMax) {
        this.activation  = activation;
        this.weightClip  = weightClip;
        this.biasMin     = biasMin;
        this.biasMax     = biasMax;
        neurons = new Neuron[neuronCount];
        for (int i = 0; i < neuronCount; i++) {
            neurons[i] = new Neuron(inputCountPerNeuron, activation);
            neurons[i].weightClip = weightClip;
            neurons[i].biasMin    = biasMin;
            neurons[i].biasMax    = biasMax;
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
        NeuronLayer clone = new NeuronLayer(neurons.length, neurons[0].weights.length,
                                            activation, weightClip, biasMin, biasMax);
        for (int i = 0; i < neurons.length; i++) {
            clone.neurons[i] = neurons[i].copy();
        }
        return clone;
    }
}
