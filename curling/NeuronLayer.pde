class NeuronLayer {
    Neuron[] neurons;

    NeuronLayer(int neuronCount, int inputCountPerNeuron) {
        neurons = new Neuron[neuronCount];
        for (int i = 0; i < neuronCount; i++) {
            neurons[i] = new Neuron(inputCountPerNeuron);
        }
    }

    float[] feedForward(float[] inputs) {
        float[] outputs = new float[neurons.length];
        for (int i = 0; i < neurons.length; i++) {
            outputs[i] = neurons[i].activate(inputs);
        }
        return outputs;
    }

    void mutate(float mutationRate) {
        for (Neuron neuron : neurons) {
            neuron.mutate(mutationRate);
        }
    }

    NeuronLayer copy() {
        NeuronLayer clone = new NeuronLayer(neurons.length, neurons[0].weights.length);
        for (int i = 0; i < neurons.length; i++) {
            clone.neurons[i] = neurons[i].copy();
        }
        return clone;
    }
}