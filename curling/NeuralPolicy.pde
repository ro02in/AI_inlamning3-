class NeuralPolicy {
    int inputSize = TOTAL_STONES * 4 + 2;
    int hiddenSize = 12;
    int outputSize = 3;

    NeuronLayer hiddenLayer;
    NeuronLayer outputLayer;

    NeuralPolicy() {
        // Hidden: leaky ReLU, smaller weights, bias floor at 0 keeps neurons alive.
        hiddenLayer = new NeuronLayer(hiddenSize, inputSize, Neuron.ACT_LEAKY_RELU,
                                      0.4f, 0.0f, 0.5f);
        outputLayer = new NeuronLayer(outputSize, hiddenSize, Neuron.ACT_LINEAR,
                                      0.6f, -0.6f, 0.6f);
    }

    // Fraction of hidden neurons with pre-activation <= 0 on this layout.
    float deadHiddenFraction(float[] state) {
        int dead = 0;
        for (Neuron n : hiddenLayer.neurons) {
            if (n.preActivation(state) <= 0) dead++;
        }
        return dead / (float) hiddenLayer.neurons.length;
    }

    float[] hiddenPreActivations(float[] state) {
        float[] pre = new float[hiddenLayer.neurons.length];
        for (int i = 0; i < hiddenLayer.neurons.length; i++) {
            pre[i] = hiddenLayer.neurons[i].preActivation(state);
        }
        return pre;
    }

    Shot predict(float[] state) {
        float[] hiddenOutputs = hiddenLayer.feedForward(state);
        float[] rawOutputs = outputLayer.feedForward(hiddenOutputs);

        float MIN_SPEED = UI.SPEED_MAX * 0.2f;

        float curl  = (float) Math.tanh(rawOutputs[0]);
        float speed = MIN_SPEED + (( (float) Math.tanh(rawOutputs[1]) + 1) / 2.0) * (UI.SPEED_MAX - MIN_SPEED);
        float angle = (float) Math.tanh(rawOutputs[2]) * PI / 18;
        return new Shot(curl, speed, angle);
    }

    float[] outputActivations(float[] state) {
        float[] hidden = hiddenLayer.feedForward(state);
        float[] raw = outputLayer.feedForward(hidden);
        return new float[] {
            (float) Math.tanh(raw[0]),
            (float) Math.tanh(raw[1]),
            (float) Math.tanh(raw[2])
        };
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
