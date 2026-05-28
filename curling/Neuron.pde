class Neuron {
    static final float LEAKY_SLOPE = 0.01f;

    static final int ACT_TANH       = 0;
    static final int ACT_RELU       = 1;
    static final int ACT_LEAKY_RELU = 2;
    static final int ACT_LINEAR     = 3;

    float[] weights;
    float bias;
    int   activation;
    float weightClip = 0.4f;
    float biasMin    = -0.4f;
    float biasMax    =  0.4f;

    Neuron(int inputCount) {
        this(inputCount, ACT_TANH);
    }

    Neuron(int inputCount, int activation) {
        this.activation = activation;
        weights = new float[inputCount];
        float scale = 1.0f / sqrt(inputCount);
        for (int i = 0; i < inputCount; i++) {
            weights[i] = random(-1, 1) * scale;
        }
        if (activation == ACT_RELU || activation == ACT_LEAKY_RELU) {
            bias = random(0.1f, 0.4f);
        } else {
            bias = random(-0.3f, 0.3f);
        }
    }

    float preActivation(float[] inputs) {
        float sum = bias;
        for (int i = 0; i < weights.length; i++) {
            sum += weights[i] * inputs[i];
        }
        return sum;
    }

    float activate(float[] inputs) {
        return applyActivation(preActivation(inputs));
    }

    float applyActivation(float sum) {
        switch (activation) {
            case ACT_LINEAR:
                return sum;
            case ACT_RELU:
                return max(0, sum);
            case ACT_LEAKY_RELU:
                return sum > 0 ? sum : LEAKY_SLOPE * sum;
            default:
                return (float) Math.tanh(sum);
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
            weights[i] = constrain(weights[i], -weightClip, weightClip);
        }
        bias = constrain(bias, biasMin, biasMax);
    }

    Neuron copy() {
        Neuron clone = new Neuron(weights.length, activation);
        clone.weightClip = weightClip;
        clone.biasMin    = biasMin;
        clone.biasMax    = biasMax;
        for (int i = 0; i < weights.length; i++) {
            clone.weights[i] = weights[i];
        }
        clone.bias = bias;
        return clone;
    }
}
