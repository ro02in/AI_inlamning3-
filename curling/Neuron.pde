class Neuron {
    float[] weights;
    float bias;

    Neuron(int inputCount) {
        weights = new float[inputCount];
        for (int i = 0; i < inputCount; i++) {
            weights[i] = random(-1, 1);
        }
        bias = random(-1, 1);
    }

    float activate(float[] inputs) {
        float sum = bias;
        for (int i = 0; i < weights.length; i++) {
            sum += weights[i] * inputs[i];
        }
        return (float) Math.tanh(sum);
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
    }

    Neuron copy() {
        Neuron clone = new Neuron(weights.length);
        for (int i = 0; i < weights.length; i++) {
            clone.weights[i] = weights[i];
        }
        clone.bias = bias;
        return clone;
    }
}