// REINFORCE-style policy gradient training for a meanAndStd NeuralPolicy.
//
// Each update step:
//   1. Randomize board state.
//   2. Single forward pass → (μ, log σ) per action dimension (stored for backprop).
//   3. Sample shotsPerUpdate shots from the Gaussian without re-running forward.
//   4. Simulate each with physics, score with active heuristics.
//   5. Subtract baseline (mean reward) → advantages.
//   6. Keep elite top-10% by advantage.
//   7. Compute weighted-average REINFORCE gradient across elites + entropy bonus.
//   8. Backprop through the network (Adam).
class PolicyGradientTraining {
    NeuralPolicy policy;
    String expertName = "";  // optional label for ensemble / diagnostics

    int   shotsPerUpdate  = 50;
    float eliteFraction   = 0.10f;
    float learningRate    = 0.001f;
    float entropyBonus    = 0.04f;  // H[π] coefficient — fights σ collapse (was 0.01)
    int   logEvery        = 100;  // print training diagnostics every N update steps

    int updateCount = 0;

    RandomState randomState = new RandomState();
    ArrayList<Stone> stones = new ArrayList<Stone>();

    PolicyGradientTraining() {
        policy = new NeuralPolicy(true);
    }

    PolicyGradientTraining(NeuralPolicy seed) {
        policy = seed;
    }

    void reset() {
        if (policy != null) {
            policy = new NeuralPolicy(policy.minCurl, policy.maxCurl,
                                      policy.minSpeed, policy.maxSpeed,
                                      policy.minAngleDeg, policy.maxAngleDeg,
                                      policy.meanAndStd);
        } else {
            policy = new NeuralPolicy(true);
        }
        updateCount = 0;
    }

    void setPolicy(NeuralPolicy p, int loadedUpdateCount) {
        if (p == null) return;
        policy = p;
        if (loadedUpdateCount >= 0) updateCount = loadedUpdateCount;
    }

    // Run one gradient update on a fresh random board state.
    void updateStep(ArrayList<Heuristic> heuristics) {
        if (heuristics == null || heuristics.isEmpty()) return;

        // 1. Random board state.
        randomState.randomize(stones, STONES_PER_TEAM);
        float[] state = policy.convertState(stones, 1, TEAM_RED);

        // 2. Single forward pass. Stores lastInput/lastPreActivation in each layer
        //    so backward() can use them. No further feedForward calls until after backward().
        float[] hidden  = policy.hiddenLayer.feedForward(state);
        float[] means   = policy.outputLayer.feedForward(hidden);
        float[] logStds = policy.outputLogStd.feedForward(hidden);

        float[] sigmas = new float[3];
        for (int i = 0; i < 3; i++) {
            sigmas[i] = policy.sigmaFromLogStd(i, logStds[i]);
        }

        // 3. Sample N actions directly from N(μ, σ) — no feedForward calls.
        int N = shotsPerUpdate;
        float[][] actionRaws = new float[N][3]; // raw tanh-space samples
        Shot[]    shots      = new Shot[N];

        for (int k = 0; k < N; k++) {
            actionRaws[k] = new float[3];
            for (int i = 0; i < 3; i++) {
                actionRaws[k][i] = means[i] + sigmas[i] * randomGaussian();
            }
            // Map to world action space.
            float curl  = policy.mapTanhToRange(constrain(actionRaws[k][0], -1f, 1f),
                                                policy.minCurl,     policy.maxCurl);
            float speed = policy.mapTanhToRange(constrain(actionRaws[k][1], -1f, 1f),
                                                policy.minSpeed,    policy.maxSpeed);
            float angle = policy.mapTanhToRange(constrain(actionRaws[k][2], -1f, 1f),
                                                policy.minAngleDeg, policy.maxAngleDeg);
            shots[k] = new Shot(curl, speed, radians(angle));
        }

        // 4. Simulate each shot and score.
        float[] rewards = new float[N];
        for (int k = 0; k < N; k++) {
            ShotResult result = simulateShot(shots[k], stones);
            float score = 0;
            for (Heuristic h : heuristics) score += h.contribute(h.scoreResult(result));
            rewards[k] = score;
        }

        // 5. Baseline subtraction.
        float meanReward = 0;
        float maxReward = Float.NEGATIVE_INFINITY;
        float minReward = Float.POSITIVE_INFINITY;
        for (float r : rewards) {
            meanReward += r;
            maxReward = max(maxReward, r);
            minReward = min(minReward, r);
        }
        meanReward /= N;

        float[] advantages = new float[N];
        for (int k = 0; k < N; k++) advantages[k] = rewards[k] - meanReward;

        // 6. Keep elite top-k.
        int eliteCount = max(1, round(N * eliteFraction));
        int[] eliteIdx = topKIndices(advantages, eliteCount);

        float advSum = 0;
        for (int idx : eliteIdx) {
            if (advantages[idx] > 0) advSum += advantages[idx];
        }
        updateCount++;
        boolean shouldLog = (updateCount == 1 || updateCount % logEvery == 0);
        if (advSum <= 0) {
            if (shouldLog) {
                policyDiagnostics.logGradientTrainingStep(
                    expertName, policy, hidden,
                    updateCount, meanReward, maxReward, minReward,
                    eliteCount, advSum, true, means, sigmas, new float[3], new float[3]);
            }
            return; // all elites are below baseline — skip update
        }

        // 7. Accumulate weighted gradient (REINFORCE + entropy bonus).
        float[] gradMean   = new float[3];
        float[] gradLogStd = new float[3];

        for (int idx : eliteIdx) {
            float adv = advantages[idx];
            if (adv <= 0) continue;

            float w = adv / advSum;
            float[] lpGrad = policy.logProbGradients(means, logStds, actionRaws[idx]);

            for (int i = 0; i < 3; i++) {
                gradMean[i]   += w * lpGrad[i];
                gradLogStd[i] += w * lpGrad[3 + i];
            }
        }

        // Entropy bonus: +dH/d(log σ) = +1 per dim. At σ floor, block REINFORCE from
        // pushing log σ lower so exploration minimum is preserved.
        for (int i = 0; i < 3; i++) {
            if (policy.logStdAtMin(i, logStds[i])) {
                gradLogStd[i] = max(gradLogStd[i], 0) + entropyBonus;
            } else {
                gradLogStd[i] += entropyBonus;
            }
        }

        // 8. Backprop through the network (uses stored forward-pass state).
        policy.backwardMeanAndStd(gradMean, gradLogStd, learningRate);

        if (shouldLog) {
            policyDiagnostics.logGradientTrainingStep(
                expertName, policy, hidden,
                updateCount, meanReward, maxReward, minReward,
                eliteCount, advSum, false, means, sigmas, gradMean, gradLogStd);
        }
    }

    // ---- Physics simulation ----

    private ShotResult simulateShot(Shot shot, ArrayList<Stone> layout) {
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
        return new ShotResult(simStones, fired, scoreBefore, shot);
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

    // Returns indices of the top-k elements in descending order.
    private int[] topKIndices(float[] values, int k) {
        int n = values.length;
        int[] indices = new int[n];
        for (int i = 0; i < n; i++) indices[i] = i;

        // Insertion sort — N is small (≤ 500).
        for (int i = 1; i < n; i++) {
            int key = indices[i];
            int j = i - 1;
            while (j >= 0 && values[indices[j]] < values[key]) {
                indices[j + 1] = indices[j];
                j--;
            }
            indices[j + 1] = key;
        }

        int[] result = new int[k];
        for (int i = 0; i < k; i++) result[i] = indices[i];
        return result;
    }
}
