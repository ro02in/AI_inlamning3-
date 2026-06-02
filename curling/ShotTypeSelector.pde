// Shot-type selector network.
//
// A small MLP: inputSize -> 3 hidden layers (RELU) -> numTypes (LINEAR logits).
// Training: for a given board state, evaluate each expert's shot via FinalScoreHeuristic
// (full self-play rollout if depth > 1). Compute reward-weighted softmax target and
// add an entropy bonus to prevent locking onto one shot type.
//
// Inference: softmax over logits -> sample (PLAY) or argmax (TEST).
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
    int logEvery    = 200;
    float[] baselineProbs;
    float[] baselineWindowSums;
    int baselineWindowCount = 0;

    ShotTypeSelector(GradientEnsemble ensemble) {
        this.ensemble  = ensemble;
        this.numTypes  = ensemble.count;
        // Input size matches the expert policies' input encoding.
        this.inputSize = TOTAL_STONES * 4 + 2;
        initializeHiddenLayers(defaultHiddenLayerCount);
        outputLayer = new NeuronLayer(numTypes,   hiddenSize, ActivationKind.LINEAR);
        initBaseline();
    }

    void reset() {
        initializeHiddenLayers(defaultHiddenLayerCount);
        outputLayer = new NeuronLayer(numTypes,   hiddenSize, ActivationKind.LINEAR);
        updateCount = 0;
        initBaseline();
    }

    // Compute softmax probabilities for the given state.
    float[] probs(float[] state) {
        float[] hidden  = feedForwardHidden(state);
        float[] logits  = outputLayer.feedForward(hidden);
        return softmax(logits);
    }

    void initializeHiddenLayers(int count) {
        hiddenLayers = new NeuronLayer[max(1, count)];
        int previousSize = inputSize;
        for (int i = 0; i < hiddenLayers.length; i++) {
            hiddenLayers[i] = new NeuronLayer(hiddenSize, previousSize,
                                              ActivationKind.RELU, NormKind.LAYERNORM);
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

    // Sample a type index weighted by probabilities.
    int sample(float[] state) {
        return sampleFromProbs(probs(state));
    }

    // Argmax type index (deterministic, for TEST mode).
    int argmax(float[] state) {
        return argmaxFromProbs(probs(state));
    }

    int sampleFromProbs(float[] p) {
        float r = random(1.0);
        float acc = 0;
        for (int i = 0; i < p.length; i++) {
            acc += p[i];
            if (r <= acc) return i;
        }
        return p.length - 1;
    }

    int argmaxFromProbs(float[] p) {
        int best = 0;
        for (int i = 1; i < p.length; i++) {
            if (p[i] > p[best]) best = i;
        }
        return best;
    }

    int sampleTopKFromProbs(float[] p, float eps) {
        float maxP = 0;
        for (float v : p) maxP = max(maxP, v);
        float sum = 0;
        for (float v : p) if (v >= maxP - eps) sum += v;
        if (sum <= 0) return argmaxFromProbs(p);
        float r = random(sum);
        float acc = 0;
        for (int i = 0; i < p.length; i++) {
            if (p[i] < maxP - eps) continue;
            acc += p[i];
            if (r <= acc) return i;
        }
        return argmaxFromProbs(p);
    }

    void initBaseline() {
        baselineProbs = new float[numTypes];
        baselineWindowSums = new float[numTypes];
        baselineWindowCount = 0;
        for (int i = 0; i < numTypes; i++) baselineProbs[i] = 1.0f / max(1, numTypes);
    }

    float[] relativeOdds(float[] p) {
        if (baselineProbs == null || baselineProbs.length != p.length) initBaseline();
        float[] odds = new float[p.length];
        float sum = 0;
        for (int i = 0; i < p.length; i++) {
            odds[i] = max(0, p[i] - baselineProbs[i]);
            sum += odds[i];
        }
        if (sum <= 1e-6f) {
            odds[argmaxFromProbs(p)] = 1;
            return odds;
        }
        for (int i = 0; i < odds.length; i++) odds[i] /= sum;
        return odds;
    }

    void updateBaselineWindow(float[] p) {
        if (baselineProbs == null || baselineProbs.length != p.length) initBaseline();
        for (int i = 0; i < p.length; i++) baselineWindowSums[i] += p[i];
        baselineWindowCount++;
        if (baselineWindowCount >= SELECTOR_BASELINE_WINDOW) {
            for (int i = 0; i < p.length; i++) {
                baselineProbs[i] = baselineWindowSums[i] / max(1, baselineWindowCount);
                baselineWindowSums[i] = 0;
            }
            baselineWindowCount = 0;
        }
    }

    // One training update. Randomizes a board at the given curriculum depth,
    // evaluates each expert's shot with the FinalScoreHeuristic, then trains
    // the selector toward the reward-weighted softmax target.
    void updateStep(int depthCap) {
        int depth         = max(1, (int) random(1, depthCap + 1));
        int stonesToPlace = TOTAL_STONES - depth;
        randomState.randomizeForDepth(stones, stonesToPlace);
        int stonesLeft = max(1, (int) ceil(depth / 2.0));

        // Use the first expert's convertState (all policies share the same encoding).
        NeuralPolicy refPolicy = ensemble.anyPolicy();
        float[] state = refPolicy.convertState(stones, stonesLeft, TEAM_RED);

        // Evaluate each expert's mean shot via final score plus a type-specific hint.
        float[] rewards = new float[numTypes];
        float[] finalRewards = new float[numTypes];
        float[] typeRewards = new float[numTypes];
        FinalScoreHeuristic fsh = ensemble.finalHeuristic;
        for (int t = 0; t < numTypes; t++) {
            NeuralPolicy p = ensemble.trainers[t].policy;
            float[] expertState = p.convertState(stones, stonesLeft, TEAM_RED);
            Shot shot = p.predictMean(expertState);
            ShotResult result = simulateShot(shot, stones, depth - 1);
            finalRewards[t] = fsh.contribute(fsh.scoreResult(result));
            float typeReward = 0;
            for (Heuristic h : ensemble.typeHeuristics[t]) {
                if (h instanceof FinalScoreHeuristic) continue;
                typeReward += h.contribute(h.scoreResult(result));
            }
            typeRewards[t] = typeReward;
            rewards[t] = finalRewards[t] + SELECTOR_TYPE_REWARD_BLEND * typeReward;
        }

        // Build the target distribution from reward-weighted softmax.
        float[] target = rewardSoftmax(rewards, SELECTOR_TEMP);

        // Forward pass (stores lastInput / lastPreActivation for backward).
        float[] hidden = feedForwardHidden(state);
        float[] logits = outputLayer.feedForward(hidden);
        float[] pred   = softmax(logits);

        // Cross-entropy gradient: dL/d(logit_i) = pred_i - target_i.
        // Subtract entropy gradient: -dH/d(logit_i) = pred_i*(log(pred_i)+H)
        //   simplified: push toward target + push away from argmax to keep entropy.
        float[] gradLogits = new float[numTypes];
        float H = 0;
        for (int i = 0; i < numTypes; i++) {
            if (pred[i] > 0) H -= pred[i] * log(pred[i]);
        }
        for (int i = 0; i < numTypes; i++) {
            // CE gradient: pred - target (we ascend, so negate to get gradient of -loss)
            float ceGrad = -(pred[i] - target[i]); // ascent on reward
            // Entropy gradient: +dH/d(logit_i) = pred_i*(H + log(pred_i)) (push entropy up)
            float entropyGrad = pred[i] * (H + (pred[i] > 0 ? log(pred[i]) : 0));
            gradLogits[i] = ceGrad + SELECTOR_ENTROPY * entropyGrad;
        }

        // Backprop through 2-layer network.
        float[] gradHidden = outputLayer.backward(gradLogits, SELECTOR_LR);
        backwardHidden(gradHidden, SELECTOR_LR);

        updateCount++;
        updateBaselineWindow(pred);
        boolean shouldLog = (updateCount == 1 || updateCount % logEvery == 0);
        if (shouldLog) {
            print("Selector step " + updateCount + "  depth=" + depth + "  probs=[");
            for (int i = 0; i < numTypes; i++) {
                print(ensemble.names[i] + ":" + nf(pred[i], 0, 2));
                if (i < numTypes - 1) print(", ");
            }
            println("]  H=" + nf(H, 0, 3));
            print("  selectorReward=[");
            for (int i = 0; i < numTypes; i++) {
                print(ensemble.names[i] + ":" + nf(rewards[i], 0, 2)
                      + "(F=" + nf(finalRewards[i], 0, 2)
                      + ",T=" + nf(typeRewards[i], 0, 2) + ")");
                if (i < numTypes - 1) print(", ");
            }
            println("]");
            print("  selectorTarget=[");
            for (int i = 0; i < numTypes; i++) {
                print(ensemble.names[i] + ":" + nf(target[i], 0, 2));
                if (i < numTypes - 1) print(", ");
            }
            println("]");
            print("  selectorBaseline=[");
            for (int i = 0; i < numTypes; i++) {
                print(ensemble.names[i] + ":" + nf(baselineProbs[i], 0, 2));
                if (i < numTypes - 1) print(", ");
            }
            println("]");
        }
    }

    // ---- Helpers ----

    private float[] softmax(float[] x) {
        float maxX = Float.NEGATIVE_INFINITY;
        for (float v : x) maxX = max(maxX, v);
        float[] out = new float[x.length];
        float sum = 0;
        for (int i = 0; i < x.length; i++) {
            out[i] = exp(x[i] - maxX);
            sum += out[i];
        }
        if (sum > 0) for (int i = 0; i < out.length; i++) out[i] /= sum;
        return out;
    }

    // Softmax of rewards / temperature -> target probabilities.
    private float[] rewardSoftmax(float[] rewards, float temp) {
        float[] scaled = new float[rewards.length];
        for (int i = 0; i < rewards.length; i++) scaled[i] = rewards[i] / max(temp, 1e-6f);
        return softmax(scaled);
    }

    private ShotResult simulateShot(Shot shot, ArrayList<Stone> layout, int shotsRemainingAfter) {
        ArrayList<Stone> before    = copyLayout(layout);
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
            for (Stone s : simStones) if (s.isMoving()) { anyMoving = true; break; }
            if (!anyMoving) break;
        }

        ShotResult result = new ShotResult(simStones, before, fired, scoreBefore, shot);
        result.shotsRemainingAfter = shotsRemainingAfter;
        return result;
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

    // ---- Serialization ----

    void appendSave(StringBuilder sb) {
        sb.append("@selector\n");
        sb.append("update_count=").append(updateCount).append('\n');
        if (baselineProbs != null) {
            sb.append("baseline_probs=");
            for (int i = 0; i < baselineProbs.length; i++) {
                if (i > 0) sb.append(' ');
                sb.append(baselineProbs[i]);
            }
            sb.append('\n');
        }
        for (int i = 0; i < hiddenLayers.length; i++) {
            hiddenLayers[i].appendSave(sb, "sel_hidden" + (i + 1));
        }
        outputLayer.appendSave(sb, "sel_output");
        sb.append("@end\n");
    }

    // Load from lines starting at @selector, return index after @end.
    int loadFromLines(String[] lines, int startIdx) {
        int idx = startIdx + 1; // skip @selector line
        hiddenLayers = new NeuronLayer[0];
        while (idx < lines.length) {
            String line = trim(lines[idx]);
            if (line.equals("@end")) return idx + 1;
            if (line.startsWith("update_count=")) {
                updateCount = parseInt(line.substring(13));
                idx++;
                continue;
            }
            if (line.startsWith("baseline_probs=")) {
                String[] parts = split(trim(line.substring(15)), ' ');
                if (baselineProbs == null || baselineProbs.length != numTypes) initBaseline();
                for (int i = 0; i < min(numTypes, parts.length); i++) {
                    baselineProbs[i] = parseFloat(parts[i]);
                }
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
