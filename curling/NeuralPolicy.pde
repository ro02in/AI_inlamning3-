// Oliwer Carpman, Rafal Galinski, Robin Karpman
class NeuralPolicy {
    int inputSize = TOTAL_STONES * 4 + 2;
    int hiddenSize = 12;
    int defaultHiddenLayerCount = 5;
    int outputSize = 3;

    final ActivationKind HIDDEN_ACTIVATION = ActivationKind.RELU;
    final ActivationKind OUTPUT_ACTIVATION = ActivationKind.TANH;

    static final float DEFAULT_MIN_CURL      = -1f;
    static final float DEFAULT_MAX_CURL      =  1f;
    static final float DEFAULT_MIN_SPEED     = UI.SPEED_MAX * 0.3f;
    static final float DEFAULT_MAX_SPEED     = UI.SPEED_MAX;
    static final float DEFAULT_MIN_ANGLE_DEG = -10f;
    static final float DEFAULT_MAX_ANGLE_DEG =  10f;

    float minCurl, maxCurl;
    float minSpeed, maxSpeed;
    float minAngleDeg, maxAngleDeg;
    int lastLoadedUpdateCount = -1;

    NeuronLayer[] hiddenLayers;
    NeuronLayer outputLayer;

    NeuralPolicy() {
        this(DEFAULT_MIN_CURL, DEFAULT_MAX_CURL,
            DEFAULT_MIN_SPEED, DEFAULT_MAX_SPEED,
            DEFAULT_MIN_ANGLE_DEG, DEFAULT_MAX_ANGLE_DEG);
    }

    NeuralPolicy(float minCurl, float maxCurl,
                float minSpeed, float maxSpeed,
                float minAngleDeg, float maxAngleDeg) {
        this.minCurl = minCurl;
        this.maxCurl = maxCurl;
        this.minSpeed = minSpeed;
        this.maxSpeed = maxSpeed;
        this.minAngleDeg = minAngleDeg;
        this.maxAngleDeg = maxAngleDeg;
        initializeHiddenLayers(defaultHiddenLayerCount);
        outputLayer = new NeuronLayer(outputSize, hiddenSize, OUTPUT_ACTIVATION);
    }

    NeuralPolicy expertDraw() {
        return new NeuralPolicy(-0.1f, 0.1f, 18f, 26f, -3f, 3f);
    }

    NeuralPolicy expertDrawCurlRight() {
        return new NeuralPolicy(0.1f, 0.35f, 18f, 26f, -5f, 1f);
    }

    NeuralPolicy expertDrawCurlLeft() {
        return new NeuralPolicy(-0.35f, -0.1f, 18f, 26f, -1f, 5f);
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
        return new NeuralPolicy(-0.5f, 0.5f, 12f, 22f, -6f, 6f);
    }

    NeuralPolicy expertFreeze() {
        return new NeuralPolicy(-0.2f, 0.2f, 14f, 27f, -5f, 5f);
    }

    void initializeHiddenLayers(int count) {
        hiddenLayers = new NeuronLayer[max(1, count)];
        int previousSize = inputSize;
        for (int i = 0; i < hiddenLayers.length; i++) {
            hiddenLayers[i] = new NeuronLayer(hiddenSize, previousSize, HIDDEN_ACTIVATION);
            previousSize = hiddenSize;
        }
    }

    float[] feedForwardHidden(float[] state) {
        float[] values = state;
        for (NeuronLayer layer : hiddenLayers) {
            values = layer.feedForward(values);
        }
        return values;
    }

    float[] rawOutput(float[] state) {
        float[] hiddenOutputs = feedForwardHidden(state);
        return outputLayer.feedForward(hiddenOutputs);
    }

    void appendLoadedHiddenLayer(NeuronLayer layer) {
        if (layer == null) return;
        int oldLength = hiddenLayers == null ? 0 : hiddenLayers.length;
        NeuronLayer[] expanded = new NeuronLayer[oldLength + 1];
        for (int i = 0; i < oldLength; i++) expanded[i] = hiddenLayers[i];
        expanded[oldLength] = layer;
        hiddenLayers = expanded;
    }

    float mapTanhToRange(float tanhOut, float lo, float hi) {
        return lo + (tanhOut + 1f) * 0.5f * (hi - lo);
    }

    float mapRangeToTanh(float value, float lo, float hi) {
        if (abs(hi - lo) < 1e-6f) return 0;
        return constrain(((value - lo) / (hi - lo)) * 2f - 1f, -1f, 1f);
    }

    Shot shotFromRaw(float[] raw) {
        float curl = mapTanhToRange(constrain(raw[0], -1f, 1f), minCurl, maxCurl);
        float speed = mapTanhToRange(constrain(raw[1], -1f, 1f), minSpeed, maxSpeed);
        float angle = mapTanhToRange(constrain(raw[2], -1f, 1f), minAngleDeg, maxAngleDeg);
        return new Shot(curl, speed, radians(angle));
    }

    float[] rawFromShot(Shot shot) {
        return new float[]{
            mapRangeToTanh(shot.curl, minCurl, maxCurl),
            mapRangeToTanh(shot.speed, minSpeed, maxSpeed),
            mapRangeToTanh(degrees(shot.angle), minAngleDeg, maxAngleDeg)
        };
    }

    Shot predict(float[] state) {
        return shotFromRaw(rawOutput(state));
    }

    Shot predictMean(float[] state) {
        return predict(state);
    }

    void backwardOutput(float[] gradOutput, float lr) {
        float[] gradHidden = outputLayer.backward(gradOutput, lr);
        for (int i = hiddenLayers.length - 1; i >= 0; i--) {
            gradHidden = hiddenLayers[i].backward(gradHidden, lr);
        }
    }

    void clipWeights() {
        for (NeuronLayer layer : hiddenLayers) layer.clipAll();
        outputLayer.clipAll();
    }

    float[] convertState(ArrayList<Stone> playedStones, int stonesLeft, int lastStoneTeam) {
        float[] state = new float[inputSize];
        int i = 0;
        for (int slot = 0; slot < TOTAL_STONES; slot++) {
            if (slot < playedStones.size()) {
                Stone s = playedStones.get(slot);
                state[i++] = (s.pos.x - sheet.centerX) / (sheet.SHEET_WIDTH_FT * 0.5);
                state[i++] = (s.pos.y - sheet.teeY) / (sheet.backFarY - sheet.teeY);
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

    void appendSave(StringBuilder sb, String expertName, int updateCount) {
        sb.append("@policy ").append(expertName).append('\n');
        sb.append("min_curl=").append(minCurl).append('\n');
        sb.append("max_curl=").append(maxCurl).append('\n');
        sb.append("min_speed=").append(minSpeed).append('\n');
        sb.append("max_speed=").append(maxSpeed).append('\n');
        sb.append("min_angle_deg=").append(minAngleDeg).append('\n');
        sb.append("max_angle_deg=").append(maxAngleDeg).append('\n');
        if (updateCount >= 0) {
            sb.append("update_count=").append(updateCount).append('\n');
        }
        for (int i = 0; i < hiddenLayers.length; i++) {
            hiddenLayers[i].appendSave(sb, "hidden" + (i + 1));
        }
        outputLayer.appendSave(sb, "output");
        sb.append("@end\n");
    }

    int loadFromLines(String[] lines, int startIdx) {
        int idx = startIdx;
        if (idx >= lines.length || !trim(lines[idx]).startsWith("@policy ")) return startIdx;

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

            if      (key.equals("min_curl"))      mc = parseFloat(val);
            else if (key.equals("max_curl"))      xc = parseFloat(val);
            else if (key.equals("min_speed"))     ms = parseFloat(val);
            else if (key.equals("max_speed"))     xs = parseFloat(val);
            else if (key.equals("min_angle_deg")) ma = parseFloat(val);
            else if (key.equals("max_angle_deg")) xa = parseFloat(val);
            else if (key.equals("update_count"))  lastLoadedUpdateCount = parseInt(val);
            idx++;
        }

        minCurl = mc;
        maxCurl = xc;
        minSpeed = ms;
        maxSpeed = xs;
        minAngleDeg = ma;
        maxAngleDeg = xa;

        hiddenLayers = new NeuronLayer[0];
        while (idx < lines.length) {
            String line = trim(lines[idx]);
            if (line.equals("@end")) return idx + 1;
            if (line.startsWith("@policy ")) return idx;
            if (line.startsWith("@layer hidden")) {
                NeuronLayer loaded = new NeuronLayer(1, 1);
                idx = loaded.loadSelf(lines, idx);
                appendLoadedHiddenLayer(loaded);
            } else if (line.startsWith("@layer output")) {
                NeuronLayer loaded = new NeuronLayer(1, 1);
                idx = loaded.loadSelf(lines, idx);
                outputLayer = loaded;
            } else {
                idx++;
            }
        }
        return idx;
    }
}
