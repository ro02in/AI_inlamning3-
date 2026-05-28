enum ActivationType { RELU, TANH, LINEAR }

class Neuron {
    static final float WEIGHT_CLIP = 1.0f;

    float[] weights;
    float bias;
    ActivationType activationType;

    float preActivation;
    float activation;
    float[] lastInputs;
    float[] weightGrads;
    float biasGrad;

    Neuron(int inputCount) {
        this(inputCount, ActivationType.TANH);
    }

    Neuron(int inputCount, ActivationType activationType) {
        this.activationType = activationType;
        weights = new float[inputCount];
        weightGrads = new float[inputCount];
        for (int i = 0; i < inputCount; i++) {
            weights[i] = random(-1, 1) / sqrt(inputCount);
        }
        bias = random(-0.3, 0.3);
    }

    // Backward compat for evolution code paths.
    Neuron(int inputCount, boolean linear) {
        this(inputCount, linear ? ActivationType.LINEAR : ActivationType.TANH);
    }

    float forward(float[] inputs) {
        if (lastInputs == null || lastInputs.length != inputs.length) {
            lastInputs = new float[inputs.length];
        }
        arrayCopy(inputs, lastInputs);

        preActivation = bias;
        for (int i = 0; i < weights.length; i++) {
            preActivation += weights[i] * inputs[i];
        }
        activation = applyActivation(preActivation);
        return activation;
    }

    float activate(float[] inputs) {
        return forward(inputs);
    }

    float applyActivation(float z) {
        switch (activationType) {
            case RELU:   return z > 0 ? z : 0.01f * z;
            case TANH:   return (float) Math.tanh(z);
            case LINEAR: return z;
        }
        return z;
    }

    float activationDerivative() {
        switch (activationType) {
            case RELU:   return preActivation > 0 ? 1 : 0.01f;
            case TANH:   return 1 - activation * activation;
            case LINEAR: return 1;
        }
        return 1;
    }

    void zeroGrads() {
        for (int i = 0; i < weightGrads.length; i++) {
            weightGrads[i] = 0;
        }
        biasGrad = 0;
    }

    void applyGrads(float learningRate, float weightDecay) {
        applyGrads(learningRate, weightDecay, weightDecay);
    }

    void applyGrads(float learningRate, float weightDecay, float biasDecay) {
        for (int i = 0; i < weights.length; i++) {
            weights[i] -= learningRate * (weightGrads[i] + 2 * weightDecay * weights[i]);
        }
        bias -= learningRate * (biasGrad + 2 * biasDecay * bias);
        zeroGrads();
    }

    void scaleGrads(float factor) {
        for (int i = 0; i < weightGrads.length; i++) {
            weightGrads[i] *= factor;
        }
        biasGrad *= factor;
    }

    void clipGrads(float maxAbs) {
        for (int i = 0; i < weightGrads.length; i++) {
            weightGrads[i] = constrain(weightGrads[i], -maxAbs, maxAbs);
        }
        biasGrad = constrain(biasGrad, -maxAbs, maxAbs);
    }

    boolean hasFiniteGrads() {
        if (!Float.isFinite(biasGrad)) return false;
        for (float g : weightGrads) if (!Float.isFinite(g)) return false;
        return true;
    }

    boolean hasFiniteWeights() {
        if (!Float.isFinite(bias)) return false;
        for (float w : weights) if (!Float.isFinite(w)) return false;
        return true;
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
        Neuron clone = new Neuron(weights.length, activationType);
        for (int i = 0; i < weights.length; i++) {
            clone.weights[i] = weights[i];
        }
        clone.bias = bias;
        return clone;
    }
}
