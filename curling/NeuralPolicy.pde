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

    // Ranges used by predict() / predictMean() to map tanh outputs.
    float minCurl, maxCurl;
    float minSpeed, maxSpeed;
    float minAngleDeg, maxAngleDeg;

    // When true the policy outputs (μ, log σ) per action dimension and
    // supports sample() for stochastic training.
    boolean meanAndStd = false;

    // Log-std clamp to keep σ in a sensible range during gradient training.
    // σ = exp(logStd), so clamp logStd to [-3, 1] → σ ∈ [0.05, 2.72].
    static final float LOG_STD_MIN = -3.0f;
    static final float LOG_STD_MAX =  1.0f;

    NeuronLayer hiddenLayer;
    NeuronLayer outputLayer;    // 3 neurons for deterministic policy or 3 mean neurons
    NeuronLayer outputLogStd;   // 3 neurons (LINEAR) for log σ — only used when meanAndStd=true

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

    // Stochastic policy: outputs (μ, log σ) per action dimension.
    // Output ranges are the full defaults; the model learns where to point.
    NeuralPolicy(boolean meanAndStd) {
        this(DEFAULT_MIN_CURL, DEFAULT_MAX_CURL,
             DEFAULT_MIN_SPEED, DEFAULT_MAX_SPEED,
             DEFAULT_MIN_ANGLE_DEG, DEFAULT_MAX_ANGLE_DEG,
             meanAndStd);
    }

    NeuralPolicy(float minCurl, float maxCurl,
                 float minSpeed, float maxSpeed,
                 float minAngleDeg, float maxAngleDeg,
                 boolean meanAndStd) {
        this.minCurl      = minCurl;
        this.maxCurl      = maxCurl;
        this.minSpeed     = minSpeed;
        this.maxSpeed     = maxSpeed;
        this.minAngleDeg  = minAngleDeg;
        this.maxAngleDeg  = maxAngleDeg;
        this.meanAndStd   = meanAndStd;
        hiddenLayer  = new NeuronLayer(hiddenSize, inputSize,  HIDDEN_ACTIVATION);
        outputLayer  = new NeuronLayer(outputSize, hiddenSize, OUTPUT_ACTIVATION);
        if (meanAndStd) {
            outputLogStd = new NeuronLayer(outputSize, hiddenSize, ActivationKind.LINEAR);
        }
    }

    // ---- Expert factory constructors ----
    // Pass meanAndStd=true for policy-gradient ensemble seeds (same output caps).

    NeuralPolicy expertDraw() {
        return expertDraw(false);
    }

    NeuralPolicy expertDraw(boolean meanAndStd) {
        return new NeuralPolicy(-0.2f, 0.2f, 18f, 26f, -3f, 3f, meanAndStd);
    }

    NeuralPolicy expertCurlRight() {
        return expertCurlRight(false);
    }

    NeuralPolicy expertCurlRight(boolean meanAndStd) {
        return new NeuralPolicy(0.35f, 1f, 18f, 36f, -10f, 3f, meanAndStd);
    }

    NeuralPolicy expertCurlLeft() {
        return expertCurlLeft(false);
    }

    NeuralPolicy expertCurlLeft(boolean meanAndStd) {
        return new NeuralPolicy(-1f, -0.35f, 18f, 36f, -3f, 10f, meanAndStd);
    }

    NeuralPolicy expertTakeout() {
        return expertTakeout(false);
    }

    NeuralPolicy expertTakeout(boolean meanAndStd) {
        return new NeuralPolicy(-0.5f, 0.5f, 26f, 40f, -8f, 8f, meanAndStd);
    }

    NeuralPolicy expertGuard() {
        return expertGuard(false);
    }

    NeuralPolicy expertGuard(boolean meanAndStd) {
        return new NeuralPolicy(-0.5f, 0.5f, 12f, 18f, -6f, 6f, meanAndStd);
    }

    NeuralPolicy expertFreeze() {
        return expertFreeze(false);
    }

    NeuralPolicy expertFreeze(boolean meanAndStd) {
        return new NeuralPolicy(-0.2f, 0.2f, 12f, 22f, -5f, 5f, meanAndStd);
    }

    // ---- Forward pass helpers ----

    float mapTanhToRange(float tanhOut, float lo, float hi) {
        return lo + (tanhOut + 1f) * 0.5f * (hi - lo);
    }

    float angleSpanDeg() {
        return maxAngleDeg - minAngleDeg;
    }

    float angleSpanRad() {
        return radians(angleSpanDeg());
    }

    // Deterministic prediction. For a meanAndStd policy this returns the mean shot.
    Shot predict(float[] state) {
        float[] hiddenOutputs = hiddenLayer.feedForward(state);
        float[] outputValues  = outputLayer.feedForward(hiddenOutputs);

        float curl  = mapTanhToRange(outputValues[0], minCurl, maxCurl);
        float speed = mapTanhToRange(outputValues[1], minSpeed, maxSpeed);
        float angle = mapTanhToRange(outputValues[2], minAngleDeg, maxAngleDeg);
        angle = radians(angle);
        return new Shot(curl, speed, angle);
    }

    // Alias for clarity when using a meanAndStd policy in play mode.
    Shot predictMean(float[] state) {
        return predict(state);
    }

    // Stochastic sample for play/exploration (e.g. during AI test).
    // Runs a fresh forward pass, so it is NOT used inside PolicyGradientTraining.updateStep()
    // (which does its own single forward pass to preserve the stored layer state for backprop).
    Shot sample(float[] state) {
        if (!meanAndStd) return predict(state);

        float[] hidden  = hiddenLayer.feedForward(state);
        float[] means   = outputLayer.feedForward(hidden);
        float[] logStds = outputLogStd.feedForward(hidden);

        float[] actions = new float[3];
        for (int i = 0; i < 3; i++) {
            float sigma  = exp(constrain(logStds[i], LOG_STD_MIN, LOG_STD_MAX));
            actions[i]   = constrain(means[i] + sigma * randomGaussian(), -1f, 1f);
        }

        float curl  = mapTanhToRange(actions[0], minCurl, maxCurl);
        float speed = mapTanhToRange(actions[1], minSpeed, maxSpeed);
        float angle = mapTanhToRange(actions[2], minAngleDeg, maxAngleDeg);
        return new Shot(curl, speed, radians(angle));
    }

    // Compute gradients of log π(action|state) w.r.t. network outputs (means and log-stds)
    // for one sampled action. Returns float[6]: [dLogP/dMean0..2, dLogP/dLogStd0..2].
    // The caller multiplies by advantage and sums across samples before backpropping.
    float[] logProbGradients(float[] means, float[] logStds, float[] actionRaw) {
        float[] grad = new float[6];
        for (int i = 0; i < 3; i++) {
            float logStdClamped = constrain(logStds[i], LOG_STD_MIN, LOG_STD_MAX);
            float sigma  = exp(logStdClamped);
            float sigma2 = sigma * sigma;
            float diff   = actionRaw[i] - means[i];

            // d log N(a; μ, σ) / dμ  = (a - μ) / σ²
            grad[i]     = diff / sigma2;
            // d log N(a; μ, σ) / d(log σ) = (a - μ)² / σ² - 1
            grad[3 + i] = (diff * diff) / sigma2 - 1.0f;
        }
        return grad;
    }

    // Backpropagate a gradient vector through the stochastic policy.
    // gradMean[3]: gradient w.r.t. mean output neurons (post-tanh) — sign convention: positive = increase
    // gradLogStd[3]: gradient w.r.t. log-std output neurons
    // Caller has already scaled gradients by advantage and learning rate direction.
    void backwardMeanAndStd(float[] gradMean, float[] gradLogStd, float lr) {
        if (!meanAndStd) return;

        // Backward through log-std head → get gradient w.r.t. hidden layer.
        float[] gradHiddenFromLogStd = outputLogStd.backward(gradLogStd, lr);

        // Backward through mean head → get gradient w.r.t. hidden layer.
        float[] gradHiddenFromMean = outputLayer.backward(gradMean, lr);

        // Sum the two gradient signals into the hidden layer.
        float[] gradHidden = new float[hiddenSize];
        for (int i = 0; i < hiddenSize; i++) {
            gradHidden[i] = gradHiddenFromMean[i] + gradHiddenFromLogStd[i];
        }

        // Backward through hidden layer (updates its weights; input gradient discarded).
        hiddenLayer.backward(gradHidden, lr);
    }

    // ---- Diagnostics (used by ExpertEnsemble) ----

    float hiddenInactiveFrac(float[] state) {
        if (HIDDEN_ACTIVATION != ActivationKind.RELU) return 0;
        float[] hidden = hiddenLayer.feedForward(state);
        int inactive = 0;
        for (float v : hidden) {
            if (v < Neuron.RELU_INACTIVE_THRESHOLD) inactive++;
        }
        return inactive / (float) hidden.length;
    }

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

    // ---- Policy search helpers ----

    void mutate(float mutationRate, float mutationStrength) {
        hiddenLayer.mutate(mutationRate, mutationStrength);
        outputLayer.mutate(mutationRate, mutationStrength);
        if (meanAndStd && outputLogStd != null) {
            outputLogStd.mutate(mutationRate, mutationStrength);
        }
    }

    void clipWeights() {
        hiddenLayer.clipAll();
        outputLayer.clipAll();
        if (meanAndStd && outputLogStd != null) {
            outputLogStd.clipAll();
        }
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
        if (meanAndStd && outputLogStd != null) {
            for (Neuron n : outputLogStd.neurons) {
                for (float w : n.weights) sum += w * w;
                sum += n.bias * n.bias;
            }
        }
        return sum;
    }

    NeuralPolicy copy() {
        NeuralPolicy clone = new NeuralPolicy(minCurl, maxCurl,
                                              minSpeed, maxSpeed,
                                              minAngleDeg, maxAngleDeg,
                                              meanAndStd);
        clone.hiddenLayer = hiddenLayer.copy();
        clone.outputLayer = outputLayer.copy();
        if (meanAndStd && outputLogStd != null) {
            clone.outputLogStd = outputLogStd.copy();
        }
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
