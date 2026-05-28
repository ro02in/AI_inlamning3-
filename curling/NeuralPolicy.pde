class NeuralPolicy {
    int inputSize = TOTAL_STONES * 4 + 2;
    int hiddenSize = 16;
    int outputSize = 3;

    final ActivationKind HIDDEN_ACTIVATION = ActivationKind.RELU;
    final ActivationKind OUTPUT_ACTIVATION = ActivationKind.TANH;

    // Default output ranges (matches original predict() behaviour).
    static final float DEFAULT_MIN_CURL      = -1f;
    static final float DEFAULT_MAX_CURL      =  1f;
    static final float DEFAULT_MIN_SPEED     = UI.SPEED_MAX * 0.3f; // 12 ft/s
    static final float DEFAULT_MAX_SPEED     = UI.SPEED_MAX;        // 40 ft/s
    static final float DEFAULT_MIN_ANGLE_DEG = -10f;
    static final float DEFAULT_MAX_ANGLE_DEG =  10f;

    float minCurl, maxCurl;
    float minSpeed, maxSpeed;
    float minAngleDeg, maxAngleDeg;

    NeuronLayer hiddenLayer;
    NeuronLayer outputLayer;

    NeuralPolicy() {
        this(DEFAULT_MIN_CURL, DEFAULT_MAX_CURL,
            DEFAULT_MIN_SPEED, DEFAULT_MAX_SPEED,
            DEFAULT_MIN_ANGLE_DEG, DEFAULT_MAX_ANGLE_DEG);
    }

    NeuralPolicy(float minCurl, float maxCurl,
                float minSpeed, float maxSpeed,
                float minAngleDeg, float maxAngleDeg) {
        this.minCurl      = minCurl;
        this.maxCurl      = maxCurl;
        this.minSpeed     = minSpeed;
        this.maxSpeed     = maxSpeed;
        this.minAngleDeg  = minAngleDeg;
        this.maxAngleDeg  = maxAngleDeg;
        hiddenLayer = new NeuronLayer(hiddenSize, inputSize, HIDDEN_ACTIVATION);
        outputLayer = new NeuronLayer(outputSize, hiddenSize, OUTPUT_ACTIVATION);
    }

    NeuralPolicy expertDraw() {
        return new NeuralPolicy(-0.2f, 0.2f, 18f, 26f, -3f, 3f);
    }

    NeuralPolicy expertCurlRight() {
        return new NeuralPolicy(0.35f, 1f, 18f, 36f, -10f, 3f);
    }

    NeuralPolicy expertCurlLeft() {
        return new NeuralPolicy(-1f, -0.35f, 18f, 36f, -3f, 10f);
    }

    NeuralPolicy expertTakeout() {
        return new NeuralPolicy(-0.5f, 0.5f, 26f, 40f, -8f, 8f);
    }

    NeuralPolicy expertGuard() {
        return new NeuralPolicy(-0.5f, 0.5f, 12f, 18f, -6f, 6f);
    }

    NeuralPolicy expertFreeze() {
        return new NeuralPolicy(-0.2f, 0.2f, 12f, 22f, -5f, 5f);
    }

float mapTanhToRange(float tanhOut, float lo, float hi) {
        return lo + (tanhOut + 1f) * 0.5f * (hi - lo);
    }

    float angleSpanDeg() {
        return maxAngleDeg - minAngleDeg;
    }

    float angleSpanRad() {
        return radians(angleSpanDeg());
    }

    Shot predict(float[] state) {
        float[] hiddenOutputs = hiddenLayer.feedForward(state);
        float[] outputValues = outputLayer.feedForward(hiddenOutputs);

        // tanh outputs in [-1, 1] mapped linearly to per-policy caps.
        float curl  = mapTanhToRange(outputValues[0], minCurl, maxCurl);
        float speed = mapTanhToRange(outputValues[1], minSpeed, maxSpeed);
        float angle = mapTanhToRange(outputValues[2], minAngleDeg, maxAngleDeg);
        angle = radians(angle);
        return new Shot(curl, speed, angle);
    }

    // Fraction of hidden units with activation below RELU_INACTIVE_THRESHOLD (0..1).
    float hiddenInactiveFrac(float[] state) {
        if (HIDDEN_ACTIVATION != ActivationKind.RELU) return 0;
        float[] hidden = hiddenLayer.feedForward(state);
        int inactive = 0;
        for (float v : hidden) {
            if (v < Neuron.RELU_INACTIVE_THRESHOLD) inactive++;
        }
        return inactive / (float) hidden.length;
    }

    // [inactiveFrac, meanSquared] for the hidden layer on this state.
    // Combined in one feedforward to avoid recomputation during training.
    float[] hiddenStats(float[] state) {
        float[] hidden = hiddenLayer.feedForward(state);
        int inactive = 0;
        float sq = 0;
        boolean isRelu = (HIDDEN_ACTIVATION == ActivationKind.RELU);
        for (float v : hidden) {
            if (isRelu && v < Neuron.RELU_INACTIVE_THRESHOLD) inactive++;
            sq += v * v;
        }
        int n = hidden.length;
        return new float[]{
            isRelu ? (inactive / (float) n) : 0,
            sq / n
        };
    }

    // Fraction of output units with |tanh-out| > 0.9 (0..1).
    // Only meaningful with tanh-output policies (the default).
    float outputSaturationFrac(float[] state) {
        if (OUTPUT_ACTIVATION != ActivationKind.TANH) return 0;
        float[] hidden = hiddenLayer.feedForward(state);
        float[] out = outputLayer.feedForward(hidden);
        int saturated = 0;
        for (float v : out) {
            if (abs(v) > 0.9f) saturated++;
        }
        return saturated / (float) out.length;
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
        NeuralPolicy clone = new NeuralPolicy(minCurl, maxCurl,
                                              minSpeed, maxSpeed,
                                              minAngleDeg, maxAngleDeg);
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
                state[i++] = s.team == TEAM_RED ? 1 : -1;
                state[i++] = 1;
            } else {
                state[i++] = 0;
                state[i++] = 0;
                int throwTeam = (slot % 2 == 0) ? TEAM_RED : TEAM_YELLOW;
                state[i++] = throwTeam == TEAM_RED ? 1 : -1;
                state[i++] = -1;
            }
        }
        state[i++] = stonesLeft / (float) STONES_PER_TEAM;
        state[i] = lastStoneTeam == TEAM_RED ? 1 : -1;
        return state;
    }
}
