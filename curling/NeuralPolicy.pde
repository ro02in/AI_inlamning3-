class NeuralPolicy {
    int inputSize = TOTAL_STONES * 4 + 2;
    int hiddenSize = 24;
    int outputSize = 3;

    NeuronLayer hiddenLayer;
    NeuronLayer outputLayer;

    float[] lastHidden;
    float[] lastOutput;

    NeuralPolicy() {
        hiddenLayer = new NeuronLayer(hiddenSize, inputSize, ActivationType.RELU);
        outputLayer = new NeuronLayer(outputSize, hiddenSize, ActivationType.LINEAR);
        initOutputWeights();
    }

    void initOutputWeights() {
        for (Neuron n : outputLayer.neurons) {
            for (int i = 0; i < n.weights.length; i++) {
                n.weights[i] *= 2.0f;
            }
        }
    }

    void forward(float[] state) {
        lastHidden = hiddenLayer.forward(state);
        lastOutput = outputLayer.forward(lastHidden);
    }

    Shot predictFromCache() {
        float minSpeed = minSpeed();
        float speedRange = UI.SPEED_MAX - minSpeed;

        float curl  = constrain(lastOutput[0], -1, 1);
        float speedNorm = constrain(lastOutput[1], -1, 1);
        float speed = minSpeed + ((speedNorm + 1) / 2.0) * speedRange;
        float angle = constrain(lastOutput[2], -1, 1) * PI / 18;
        return new Shot(curl, speed, angle);
    }

    Shot predict(float[] state) {
        forward(state);
        return predictFromCache();
    }

    void backwardFromShotGrad(float dCurl, float dSpeed, float dAngle) {
        float minSpeed = minSpeed();
        float speedRange = UI.SPEED_MAX - minSpeed;

        float[] outputGrads = new float[outputSize];
        outputGrads[0] = clipGrad(lastOutput[0], -1, 1, dCurl);
        outputGrads[1] = clipGrad(lastOutput[1], -1, 1, dSpeed * speedRange / 2.0f);
        outputGrads[2] = clipGrad(lastOutput[2], -1, 1, dAngle * (18.0f / PI));

        float[] hiddenGrads = outputLayer.backward(outputGrads);
        hiddenLayer.backward(hiddenGrads);
    }

    float clipGrad(float raw, float lo, float hi, float grad) {
        if (raw <= lo && grad < 0) return 0;
        if (raw >= hi && grad > 0) return 0;
        return grad;
    }

    void zeroGrads() {
        hiddenLayer.zeroGrads();
        outputLayer.zeroGrads();
    }

    void applyGrads(float learningRate, float weightDecay) {
        hiddenLayer.applyGrads(learningRate, weightDecay);
        outputLayer.applyGrads(learningRate, weightDecay);
    }

    void applyGrads(float learningRate, float hiddenDecay, float outputDecay) {
        hiddenLayer.applyGrads(learningRate, hiddenDecay, hiddenDecay);
        outputLayer.applyGrads(learningRate, 0, outputDecay);
    }

    void scaleGrads(float factor) {
        hiddenLayer.scaleGrads(factor);
        outputLayer.scaleGrads(factor);
    }

    void clipGrads(float maxAbs) {
        hiddenLayer.clipGrads(maxAbs);
        outputLayer.clipGrads(maxAbs);
    }

    boolean hasFiniteGrads() {
        return hiddenLayer.hasFiniteGrads() && outputLayer.hasFiniteGrads();
    }

    boolean hasFiniteWeights() {
        return hiddenLayer.hasFiniteWeights() && outputLayer.hasFiniteWeights();
    }

    float minSpeed() {
        return UI.SPEED_MAX * 0.2f;
    }

    void mutate(float mutationRate, float mutationStrength) {
        hiddenLayer.mutate(mutationRate, mutationStrength);
        outputLayer.mutate(mutationRate, mutationStrength);
    }

    void clipWeights() {
        hiddenLayer.clipAll();
        outputLayer.clipAll();
    }

    float weightL2() {
        float sum = 0;
        for (Neuron n : hiddenLayer.neurons) {
            for (float w : n.weights) sum += w * w;
            sum += n.bias * n.bias;
        }
        for (Neuron n : outputLayer.neurons) {
            for (float w : n.weights) sum += w * w;
            sum += n.bias * n.bias;
        }
        return sum;
    }

    NeuralPolicy copy() {
        NeuralPolicy clone = new NeuralPolicy();
        clone.hiddenLayer = hiddenLayer.copy();
        clone.outputLayer = outputLayer.copy();
        return clone;
    }

    float[] convertState(ArrayList<Stone> playedStones, int stonesLeft, int lastStoneTeam) {
        float[] state = new float[inputSize];
        int i = 0;
        for (int slot = 0; slot < TOTAL_STONES; slot++) {
            if (slot < playedStones.size()) {
                Stone s = playedStones.get(slot);
                state[i++] = (s.pos.x - sheet.centerX) / (sheet.SHEET_WIDTH_FT * 0.5);
                state[i++] = (s.pos.y - sheet.hogY) / (sheet.backFarY - sheet.hogY);
                state[i++] = s.team == TEAM_RED ? 1 : 0;
                state[i++] = 1;
            } else {
                state[i++] = 0;
                state[i++] = 0;
                int throwTeam = (slot % 2 == 0) ? TEAM_RED : TEAM_YELLOW;
                state[i++] = throwTeam == TEAM_RED ? 1 : 0;
                state[i++] = -1;
            }
        }
        state[i++] = stonesLeft / (float) STONES_PER_TEAM;
        state[i] = lastStoneTeam == TEAM_RED ? 0 : 1;
        return state;
    }
}
