// Shot-type selector network.
//
// A small 2-layer MLP: inputSize -> hidden (RELU) -> numTypes (LINEAR logits).
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

    NeuronLayer hiddenLayer;
    NeuronLayer outputLayer;

    RandomState randomState = new RandomState();
    ArrayList<Stone> stones = new ArrayList<Stone>();

    int updateCount = 0;
    int logEvery    = 200;

    ShotTypeSelector(GradientEnsemble ensemble) {
        this.ensemble  = ensemble;
        this.numTypes  = ensemble.count;
        // Input size matches the expert policies' input encoding.
        this.inputSize = TOTAL_STONES * 4 + 2;
        hiddenLayer = new NeuronLayer(hiddenSize, inputSize,  ActivationKind.RELU);
        outputLayer = new NeuronLayer(numTypes,   hiddenSize, ActivationKind.LINEAR);
    }

    void reset() {
        hiddenLayer = new NeuronLayer(hiddenSize, inputSize,  ActivationKind.RELU);
        outputLayer = new NeuronLayer(numTypes,   hiddenSize, ActivationKind.LINEAR);
        updateCount = 0;
    }

    // Compute softmax probabilities for the given state.
    float[] probs(float[] state) {
        float[] hidden  = hiddenLayer.feedForward(state);
        float[] logits  = outputLayer.feedForward(hidden);
        return softmax(logits);
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

        // Evaluate each expert's mean shot via FinalScoreHeuristic.
        float[] rewards = new float[numTypes];
        FinalScoreHeuristic fsh = ensemble.finalHeuristic;
        for (int t = 0; t < numTypes; t++) {
            NeuralPolicy p = ensemble.trainers[t].policy;
            float[] expertState = p.convertState(stones, stonesLeft, TEAM_RED);
            Shot shot = p.predictMean(expertState);
            ShotResult result = simulateShot(shot, stones, depth - 1);
            rewards[t] = fsh.contribute(fsh.scoreResult(result));
        }

        // Build the target distribution from reward-weighted softmax.
        float[] target = rewardSoftmax(rewards, SELECTOR_TEMP);

        // Forward pass (stores lastInput / lastPreActivation for backward).
        float[] hidden = hiddenLayer.feedForward(state);
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
        hiddenLayer.backward(gradHidden, SELECTOR_LR);

        updateCount++;
        boolean shouldLog = (updateCount == 1 || updateCount % logEvery == 0);
        if (shouldLog) {
            print("Selector step " + updateCount + "  depth=" + depth + "  probs=[");
            for (int i = 0; i < numTypes; i++) {
                print(ensemble.names[i] + ":" + nf(pred[i], 0, 2));
                if (i < numTypes - 1) print(", ");
            }
            println("]  H=" + nf(H, 0, 3));
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
        hiddenLayer.appendSave(sb, "sel_hidden");
        outputLayer.appendSave(sb, "sel_output");
        sb.append("@end\n");
    }

    // Load from lines starting at @selector, return index after @end.
    int loadFromLines(String[] lines, int startIdx) {
        int idx = startIdx + 1; // skip @selector line
        while (idx < lines.length) {
            String line = trim(lines[idx]);
            if (line.equals("@end")) return idx + 1;
            if (line.startsWith("update_count=")) {
                updateCount = parseInt(line.substring(13));
                idx++;
                continue;
            }
            if (line.startsWith("@layer sel_hidden")) {
                idx = hiddenLayer.loadSelf(lines, idx);
                continue;
            }
            if (line.startsWith("@layer sel_output")) {
                idx = outputLayer.loadSelf(lines, idx);
                continue;
            }
            idx++;
        }
        return idx;
    }
}
