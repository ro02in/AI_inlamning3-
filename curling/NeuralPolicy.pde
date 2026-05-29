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
                state[i++] = 0;
                state[i++] = 0;
                state[i++] = 0;
                state[i++] = 0;
            }
        }
        state[i++] = stonesLeft / (float) STONES_PER_TEAM;
        state[i] = lastStoneTeam == TEAM_RED ? 0 : 1;
        return state;
    }
    void saveToFile(String filename) {
        StringBuilder sb = new StringBuilder();

        sb.append("NEURAL POLICY\n");
        sb.append("====================\n\n");

        // -------------------------
        // Hidden layer
        // -------------------------
        sb.append("HIDDEN LAYER\n");
        sb.append("neurons=").append(hiddenLayer.neurons.length).append("\n");
        sb.append("inputsPerNeuron=").append(hiddenLayer.neurons[0].weights.length).append("\n\n");

        for (int i = 0; i < hiddenLayer.neurons.length; i++) {
            Neuron n = hiddenLayer.neurons[i];
            sb.append("Neuron ").append(i).append("\n");
            sb.append("weights: ");

            for (int j = 0; j < n.weights.length; j++) {
                sb.append(n.weights[j]);
                if (j < n.weights.length - 1) sb.append(", ");
            }

            sb.append("\n");
            sb.append("bias: ").append(n.bias).append("\n\n");
        }

        // -------------------------
        // Output layer
        // -------------------------
        sb.append("OUTPUT LAYER\n");
        sb.append("neurons=").append(outputLayer.neurons.length).append("\n");
        sb.append("inputsPerNeuron=").append(outputLayer.neurons[0].weights.length).append("\n\n");

        for (int i = 0; i < outputLayer.neurons.length; i++) {
            Neuron n = outputLayer.neurons[i];
            sb.append("Neuron ").append(i).append("\n");
            sb.append("weights: ");

            for (int j = 0; j < n.weights.length; j++) {
                sb.append(n.weights[j]);
                if (j < n.weights.length - 1) sb.append(", ");
            }

            sb.append("\n");
            sb.append("bias: ").append(n.bias).append("\n\n");
        }

        saveStrings(filename, sb.toString().split("\n"));
    }
}
