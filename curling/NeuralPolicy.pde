class NeuralPolicy {
    int inputSize = TOTAL_STONES * 4 + 2;
    int hiddenSize = 8;
    int outputSize = 3;

    NeuronLayer hiddenLayer;
    NeuronLayer outputLayer;

    NeuralPolicy() {
        hiddenLayer = new NeuronLayer(hiddenSize, inputSize);
        outputLayer = new NeuronLayer(outputSize, hiddenSize);
    }

    Shot predict(float[] state) {
        float[] hiddenOutputs = hiddenLayer.feedForward(state);
        float[] outputValues = outputLayer.feedForward(hiddenOutputs);

        float MIN_SPEED = UI.SPEED_MAX * 0.3f; // Minimum speed to prevent all shots from being too soft

        float curl = outputValues[0];
        float speed = MIN_SPEED + ((outputValues[1] + 1) / 2) * (UI.SPEED_MAX - MIN_SPEED);
        float angle = outputValues[2] * PI / 18;
        return new Shot(curl, speed, angle);
    }

    void mutate(float mutationRate, float mutationStrength) {
        hiddenLayer.mutate(mutationRate, mutationStrength);
        outputLayer.mutate(mutationRate, mutationStrength);
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
                state[i++] = s.pos.x / sheet.SHEET_WIDTH_FT;
                state[i++] = s.pos.y / sheet.yMax;
                state[i++] = s.team == TEAM_RED ? 0 : 1;
                state[i++] = 1;
            } else {
                i += 4;
            }
        }
        state[i++] = stonesLeft / (float) STONES_PER_TEAM;
        state[i] = lastStoneTeam == TEAM_RED ? 0 : 1;
        return state;
    }
}
