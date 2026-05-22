class NeuralPolicy {
    int inputSize = TOTAL_STONES * 4 + 2;
    int hiddenSize = 12;
    int outputSize = 3;

    NeuronLayer hiddenLayer;
    NeuronLayer outputLayer;

    NeuralPolicy() {
        hiddenLayer = new NeuronLayer(hiddenSize, inputSize, false);
        outputLayer = new NeuronLayer(outputSize, hiddenSize, false); // tanh on both layers; output capped in predict
    }

    Shot predict(float[] state) {
        float[] hiddenOutputs = hiddenLayer.feedForward(state);
        float[] outputValues = outputLayer.feedForward(hiddenOutputs);

        // All three outputs are tanh → [-1, 1].
        // curl stays as-is; speed maps [-1,1] → [MIN,MAX]; angle maps to ±10°.
        float MIN_SPEED = UI.SPEED_MAX * 0.2f;

        float curl  = outputValues[0];
        float speed = MIN_SPEED + ((outputValues[1] + 1) / 2.0) * (UI.SPEED_MAX - MIN_SPEED);
        float angle = outputValues[2] * PI / 18;
        return new Shot(curl, speed, angle);
    }

    void mutate(float mutationRate, float mutationStrength) {
        hiddenLayer.mutate(mutationRate, mutationStrength);
        outputLayer.mutate(mutationRate, mutationStrength);
    }

    void clipWeights() {
        hiddenLayer.clipAll();
        outputLayer.clipAll();
    }

    // Sum of all squared weights and biases. Used as L2 regularisation penalty.
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
                state[i++] = 1; // 1 = existing stone
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
