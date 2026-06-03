// Chooses which expert should be used for a board state.
class ShotTypeSelector {
    GradientEnsemble ensemble;
    int numTypes;
    int inputSize;
    int hiddenSize = 12;
    int defaultHiddenLayerCount = 3;

    NeuronLayer[] hiddenLayers;
    NeuronLayer outputLayer;

    RandomState randomState = new RandomState();
    ArrayList<Stone> stones = new ArrayList<Stone>();

    int updateCount = 0;
    int logEvery = 200;

    ShotTypeSelector(GradientEnsemble ensemble) {
        this.ensemble = ensemble;
        this.numTypes = ensemble.count;
        this.inputSize = TOTAL_STONES * 4 + 2;
        initializeHiddenLayers(defaultHiddenLayerCount);
        outputLayer = new NeuronLayer(numTypes, hiddenSize, ActivationKind.LINEAR);
    }

    void reset() {
        initializeHiddenLayers(defaultHiddenLayerCount);
        outputLayer = new NeuronLayer(numTypes, hiddenSize, ActivationKind.LINEAR);
        updateCount = 0;
    }

    float[] probs(float[] state) {
        float[] hidden = feedForwardHidden(state);
        float[] logits = outputLayer.feedForward(hidden);
        return softmax(logits);
    }

    void initializeHiddenLayers(int count) {
        hiddenLayers = new NeuronLayer[max(1, count)];
        int previousSize = inputSize;
        for (int i = 0; i < hiddenLayers.length; i++) {
            hiddenLayers[i] = new NeuronLayer(hiddenSize, previousSize, ActivationKind.RELU);
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

    void backwardHidden(float[] gradHidden, float lr) {
        for (int i = hiddenLayers.length - 1; i >= 0; i--) {
            gradHidden = hiddenLayers[i].backward(gradHidden, lr);
        }
    }

    void appendLoadedHiddenLayer(NeuronLayer layer) {
        if (layer == null) return;
        int oldLength = hiddenLayers == null ? 0 : hiddenLayers.length;
        NeuronLayer[] expanded = new NeuronLayer[oldLength + 1];
        for (int i = 0; i < oldLength; i++) expanded[i] = hiddenLayers[i];
        expanded[oldLength] = layer;
        hiddenLayers = expanded;
    }

    int argmax(float[] state) {
        return argmaxFromProbs(probs(state));
    }

    int argmaxFromProbs(float[] p) {
        int best = 0;
        for (int i = 1; i < p.length; i++) {
            if (p[i] > p[best]) best = i;
        }
        return best;
    }

    void updateStep(int depthCap) {
        int depth = max(1, (int) random(1, depthCap + 1));
        int stonesToPlace = TOTAL_STONES - depth;
        randomState.randomizeForDepth(stones, stonesToPlace);
        int stonesLeft = max(1, (int) ceil(depth / 2.0));

        NeuralPolicy refPolicy = ensemble.anyPolicy();
        float[] state = refPolicy.convertState(stones, stonesLeft, TEAM_RED);

        float[] rewards = new float[numTypes];
        FinalScoreHeuristic finalScore = ensemble.finalHeuristic;
        for (int t = 0; t < numTypes; t++) {
            NeuralPolicy p = ensemble.trainers[t].policy;
            float[] expertState = p.convertState(stones, stonesLeft, TEAM_RED);
            Shot shot = p.predictMean(expertState);
            ShotResult result = simulateShot(shot, stones);

            float typeReward = 0;
            for (Heuristic h : ensemble.typeHeuristics[t]) {
                if (h instanceof FinalScoreHeuristic) continue;
                typeReward += h.contribute(h.scoreResult(result));
            }
            float finalReward = finalScore.contribute(finalScore.scoreResult(result));
            rewards[t] = finalReward + SELECTOR_TYPE_REWARD_BLEND * typeReward;
        }

        float[] target = rewardSoftmax(rewards, SELECTOR_TEMP);
        float[] hidden = feedForwardHidden(state);
        float[] logits = outputLayer.feedForward(hidden);
        float[] pred = softmax(logits);

        float[] gradLogits = new float[numTypes];
        for (int i = 0; i < numTypes; i++) {
            gradLogits[i] = target[i] - pred[i];
        }

        float[] gradHidden = outputLayer.backward(gradLogits, SELECTOR_LR);
        backwardHidden(gradHidden, SELECTOR_LR);

        updateCount++;
        if (updateCount == 1 || updateCount % logEvery == 0) {
            println("Selector step " + updateCount
                + " depth=" + depth
                + " best=" + ensemble.names[argmaxFromProbs(target)]);
        }
    }

    private float[] softmax(float[] x) {
        float maxX = Float.NEGATIVE_INFINITY;
        for (float v : x) maxX = max(maxX, v);
        float[] out = new float[x.length];
        float sum = 0;
        for (int i = 0; i < x.length; i++) {
            out[i] = exp(x[i] - maxX);
            sum += out[i];
        }
        if (sum > 0) {
            for (int i = 0; i < out.length; i++) out[i] /= sum;
        }
        return out;
    }

    private float[] rewardSoftmax(float[] rewards, float temp) {
        float[] scaled = new float[rewards.length];
        for (int i = 0; i < rewards.length; i++) {
            scaled[i] = rewards[i] / max(temp, 1e-6f);
        }
        return softmax(scaled);
    }

    private ShotResult simulateShot(Shot shot, ArrayList<Stone> layout) {
        ArrayList<Stone> before = copyLayout(layout);
        ArrayList<Stone> simStones = copyLayout(layout);
        ScoreResult scoreBefore = house.scoreEnd(simStones);

        PVector h = sheet.hackWorld();
        Stone fired = new Stone(h.x, h.y, TEAM_YELLOW);
        fired.curl = constrain(shot.curl, -1, 1);
        fired.vel.set(sin(shot.angle) * shot.speed, cos(shot.angle) * shot.speed);
        simStones.add(fired);

        for (int step = 0; step < 100000; step++) {
            physics.step(simStones, DT);
            boolean anyMoving = false;
            for (Stone s : simStones) {
                if (s.isMoving()) { anyMoving = true; break; }
            }
            if (!anyMoving) break;
        }

        return new ShotResult(simStones, before, fired, scoreBefore, shot);
    }

    private ArrayList<Stone> copyLayout(ArrayList<Stone> layout) {
        ArrayList<Stone> copy = new ArrayList<Stone>();
        for (Stone s : layout) {
            Stone c = new Stone(s.pos.x, s.pos.y, s.team);
            c.hogPassed = s.hogPassed;
            copy.add(c);
        }
        return copy;
    }

    void appendSave(StringBuilder sb) {
        sb.append("@selector\n");
        sb.append("update_count=").append(updateCount).append('\n');
        for (int i = 0; i < hiddenLayers.length; i++) {
            hiddenLayers[i].appendSave(sb, "sel_hidden" + (i + 1));
        }
        outputLayer.appendSave(sb, "sel_output");
        sb.append("@end\n");
    }

    int loadFromLines(String[] lines, int startIdx) {
        int idx = startIdx + 1;
        hiddenLayers = new NeuronLayer[0];
        while (idx < lines.length) {
            String line = trim(lines[idx]);
            if (line.equals("@end")) return idx + 1;
            if (line.startsWith("update_count=")) {
                updateCount = parseInt(line.substring(13));
                idx++;
                continue;
            }
            if (line.startsWith("@layer sel_hidden")) {
                NeuronLayer loaded = new NeuronLayer(1, 1);
                idx = loaded.loadSelf(lines, idx);
                appendLoadedHiddenLayer(loaded);
                continue;
            }
            if (line.startsWith("@layer sel_output")) {
                NeuronLayer loaded = new NeuronLayer(1, 1);
                idx = loaded.loadSelf(lines, idx);
                outputLayer = loaded;
                continue;
            }
            idx++;
        }
        return idx;
    }
}
