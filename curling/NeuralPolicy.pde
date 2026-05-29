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
    int lastLoadedUpdateCount = -1; // set by loadFromLines()

    // Log-std clamp: σ = exp(logStd). Per-dimension minimum σ (in tanh space for curl/speed;
    // angle min is converted from degrees using this policy's angle span).
    static final float MIN_STD_CURL_TANH  = 0.15f;
    static final float MIN_STD_SPEED_TANH = 0.20f;
    static final float MIN_STD_ANGLE_DEG  = 0.50f;
    static final float MAX_STD_CURL_TANH  = 1.00f;
    static final float MAX_STD_SPEED_TANH = 0.80f;
    static final float MAX_STD_ANGLE_TANH = 0.80f;

    // Minimum σ in tanh space for action dimension i (0=curl, 1=speed, 2=angle).
    float minStdTanh(int dim) {
        if (dim == 0) return MIN_STD_CURL_TANH;
        if (dim == 1) return MIN_STD_SPEED_TANH;
        float span = max(1f, angleSpanDeg());
        return MIN_STD_ANGLE_DEG / (0.5f * span);
    }

    float logStdMin(int dim) {
        return log(minStdTanh(dim));
    }

    float maxStdTanh(int dim) {
        if (dim == 0) return MAX_STD_CURL_TANH;
        if (dim == 1) return MAX_STD_SPEED_TANH;
        return MAX_STD_ANGLE_TANH;
    }

    float logStdMax(int dim) {
        return log(maxStdTanh(dim));
    }

    float clampLogStd(int dim, float logStd) {
        return constrain(logStd, logStdMin(dim), logStdMax(dim));
    }

    float sigmaFromLogStd(int dim, float logStd) {
        return exp(clampLogStd(dim, logStd));
    }

    boolean logStdAtMin(int dim, float logStd) {
        return logStd <= logStdMin(dim) + 1e-4f;
    }

    boolean logStdAtMax(int dim, float logStd) {
        return logStd >= logStdMax(dim) - 1e-4f;
    }

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
        hiddenLayer = new NeuronLayer(hiddenSize, inputSize,
                                      HIDDEN_ACTIVATION, NormKind.LAYERNORM);
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
        hiddenLayer  = new NeuronLayer(hiddenSize, inputSize,
                                       HIDDEN_ACTIVATION, NormKind.LAYERNORM);
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
        return new NeuralPolicy(-0.1f, 0.1f, 18f, 26f, -3f, 3f, meanAndStd);
    }

    NeuralPolicy expertDrawCurlRight() {
        return expertDrawCurlRight(false);
    }

    NeuralPolicy expertDrawCurlRight(boolean meanAndStd) {
        return new NeuralPolicy(0.1f, 0.35f, 18f, 26f, -5f, 1f, meanAndStd);
    }

    NeuralPolicy expertDrawCurlLeft() {
        return expertDrawCurlLeft(false);
    }

    NeuralPolicy expertDrawCurlLeft(boolean meanAndStd) {
        return new NeuralPolicy(-0.35f, -0.1f, 18f, 26f, -1f, 5f, meanAndStd);
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
        return new NeuralPolicy(-0.5f, 0.5f, 12f, 22f, -6f, 6f, meanAndStd);
    }

    NeuralPolicy expertFreeze() {
        return expertFreeze(false);
    }

    NeuralPolicy expertFreeze(boolean meanAndStd) {
        return new NeuralPolicy(-0.2f, 0.2f, 14f, 27f, -5f, 5f, meanAndStd);
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
            float sigma  = sigmaFromLogStd(i, logStds[i]);
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
            float sigma  = sigmaFromLogStd(i, logStds[i]);
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
                state[i++] = 0;
                state[i++] = 0;
            }
        }
        state[i++] = stonesLeft / (float) STONES_PER_TEAM;
        state[i] = lastStoneTeam == TEAM_RED ? 1 : -1;
        return state;
    }

    // ---- File I/O ----

    void appendSave(StringBuilder sb, String expertName, int updateCount) {
        sb.append("@policy ").append(expertName).append('\n');
        sb.append("mean_and_std=").append(meanAndStd ? 1 : 0).append('\n');
        sb.append("min_curl=").append(minCurl).append('\n');
        sb.append("max_curl=").append(maxCurl).append('\n');
        sb.append("min_speed=").append(minSpeed).append('\n');
        sb.append("max_speed=").append(maxSpeed).append('\n');
        sb.append("min_angle_deg=").append(minAngleDeg).append('\n');
        sb.append("max_angle_deg=").append(maxAngleDeg).append('\n');
        if (updateCount >= 0) {
            sb.append("update_count=").append(updateCount).append('\n');
        }
        hiddenLayer.appendSave(sb, "hidden");
        outputLayer.appendSave(sb, "output");
        if (meanAndStd && outputLogStd != null) {
            outputLogStd.appendSave(sb, "logstd");
        }
        sb.append("@end\n");
    }

    // Load one @policy ... @end block starting at lines[startIdx]. Returns index after @end.
    int loadFromLines(String[] lines, int startIdx) {
        int idx = startIdx;
        if (idx >= lines.length || !trim(lines[idx]).startsWith("@policy ")) return startIdx;

        boolean meanStdFlag = meanAndStd;
        float mc = minCurl, xc = maxCurl;
        float ms = minSpeed, xs = maxSpeed;
        float ma = minAngleDeg, xa = maxAngleDeg;
        lastLoadedUpdateCount = -1;

        idx++;
        while (idx < lines.length) {
            String line = trim(lines[idx]);
            if (line.length() == 0 || line.startsWith("#")) { idx++; continue; }
            if (line.equals("@end")) return idx + 1;
            if (line.startsWith("@policy ")) return idx;
            if (line.startsWith("@layer ")) break;

            int eq = line.indexOf('=');
            if (eq <= 0) { idx++; continue; }
            String key = line.substring(0, eq);
            String val = line.substring(eq + 1);

            if      (key.equals("mean_and_std"))   meanStdFlag = parseInt(val) != 0;
            else if (key.equals("min_curl"))       mc = parseFloat(val);
            else if (key.equals("max_curl"))       xc = parseFloat(val);
            else if (key.equals("min_speed"))      ms = parseFloat(val);
            else if (key.equals("max_speed"))      xs = parseFloat(val);
            else if (key.equals("min_angle_deg"))  ma = parseFloat(val);
            else if (key.equals("max_angle_deg"))  xa = parseFloat(val);
            else if (key.equals("update_count"))   lastLoadedUpdateCount = parseInt(val);
            idx++;
        }

        minCurl = mc;
        maxCurl = xc;
        minSpeed = ms;
        maxSpeed = xs;
        minAngleDeg = ma;
        maxAngleDeg = xa;
        meanAndStd = meanStdFlag;
        if (meanAndStd && outputLogStd == null) {
            outputLogStd = new NeuronLayer(outputSize, hiddenSize, ActivationKind.LINEAR);
        } else if (!meanAndStd) {
            outputLogStd = null;
        }

        NeuronLayer layerLoader = new NeuronLayer(1, 1);
        while (idx < lines.length) {
            String line = trim(lines[idx]);
            if (line.equals("@end")) return idx + 1;
            if (line.startsWith("@policy ")) return idx;
            if (line.startsWith("@layer hidden")) {
                idx = layerLoader.loadFromLines(lines, idx, this, "hidden");
            } else if (line.startsWith("@layer output")) {
                idx = layerLoader.loadFromLines(lines, idx, this, "output");
            } else if (line.startsWith("@layer logstd")) {
                idx = layerLoader.loadFromLines(lines, idx, this, "logstd");
            } else {
                idx++;
            }
        }
        return idx;
    }
}
