class Neuron {
    static final float WEIGHT_CLIP = 1.0f;

    float[] weights;
    float bias;
    boolean linear; // true = no tanh (used on output layer)

    Neuron(int inputCount) {
        this(inputCount, false);
    }

    Neuron(int inputCount, boolean linear) {
        this.linear = linear;
        weights = new float[inputCount];
        for (int i = 0; i < inputCount; i++) {
            weights[i] = random(-1, 1) / sqrt(inputCount);
        }
        bias = random(-0.3, 0.3);
    }

    float activate(float[] inputs) {
        float sum = bias;
        for (int i = 0; i < weights.length; i++) {
            sum += weights[i] * inputs[i];
        }
        return linear ? sum : (float) Math.tanh(sum);
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
        Neuron clone = new Neuron(weights.length, linear);
        for (int i = 0; i < weights.length; i++) {
            clone.weights[i] = weights[i];
        }
        clone.bias = bias;
        return clone;
    }
}
