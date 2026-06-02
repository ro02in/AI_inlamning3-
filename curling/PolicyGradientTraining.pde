// REINFORCE-style policy gradient training for a meanAndStd NeuralPolicy.
//
// Each update step:
//   1. Randomize board state for the given curriculum depth.
//   2. Single forward pass → (μ, log σ) per action dimension (stored for backprop).
//   3. Sample shotsPerUpdate shots from the Gaussian without re-running forward.
//   4. Simulate each with physics, score with type-specific + finalScore heuristics.
//   5. Subtract baseline (mean reward) → advantages.
//   6. Keep elite top-10% by advantage.
//   7. Compute weighted-average REINFORCE gradient across elites + entropy bonus.
//   8. Backprop through the network (Adam).
class PolicyGradientTraining {
    NeuralPolicy policy;
    String expertName = "";

    int   shotsPerUpdate  = 50;
    float eliteFraction   = 0.25f;
    float learningRate    = 0.0007f;
    int   logEvery        = 100;

    final float TARGET_STD_CURL_TANH  = 0.45f;
    final float TARGET_STD_SPEED_TANH = 0.35f;
    final float TARGET_STD_ANGLE_TANH = 0.30f;
    final float STD_REG_STRENGTH      = 0.08f;
    final float GRAD_LOG_STD_LIMIT    = 0.25f;
    final float GRAD_MEAN_LIMIT       = 1.50f;
    final float RAW_CLIP_PENALTY      = 0.08f;
    final float MEAN_SAT_PENALTY      = 0.08f;
    final float MEAN_SAT_GRAD         = 0.12f;

    int updateCount = 0;

    RandomState randomState = new RandomState();
    ArrayList<Stone> stones = new ArrayList<Stone>();

    PolicyGradientTraining() {
        policy = new NeuralPolicy(true);
        initializeLogStdToTargets();
    }

    PolicyGradientTraining(NeuralPolicy seed) {
        policy = seed;
        initializeLogStdToTargets();
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
        initializeLogStdToTargets();
        updateCount = 0;
    }

    void setPolicy(NeuralPolicy p, int loadedUpdateCount) {
        if (p == null) return;
        policy = p;
        if (loadedUpdateCount >= 0) updateCount = loadedUpdateCount;
    }

    // Main training entry point, called by GradientEnsemble.
    // depth = how many shots remain including the one being trained (1..TOTAL_STONES).
    void updateStep(ArrayList<Heuristic> heuristics, int depth) {
        updateStep(heuristics, depth, expertName);
    }

    void updateStep(ArrayList<Heuristic> heuristics, int depth, String curriculumExpert) {
        if (heuristics == null || heuristics.isEmpty()) return;

        // 1. Random board state based on curriculum depth.
        //    stonesToPlace = stones already on the ice before yellow throws.
        int stonesToPlace = TOTAL_STONES - depth;
        randomState.randomizeForExpert(stones, stonesToPlace, curriculumExpert);

        // stonesLeft for yellow: how many shots yellow still has including this one.
        int stonesLeft = max(1, (int) ceil(depth / 2.0));
        float[] state = policy.convertState(stones, stonesLeft, TEAM_RED);

        // 2. Single forward pass.
        float[] hidden  = policy.feedForwardHidden(state);
        float[] means   = policy.outputLayer.feedForward(hidden);
        float[] logStds = policy.outputLogStd.feedForward(hidden);

        float[] sigmas = new float[3];
        for (int i = 0; i < 3; i++) {
            sigmas[i] = policy.sigmaFromLogStd(i, logStds[i]);
        }
        int meanSatCount = 0;
        for (int i = 0; i < 3; i++) {
            if (abs(means[i]) > 0.9f) meanSatCount++;
        }

        // 3. Sample N actions.
        int N = shotsPerUpdate;
        float[][] actionRaws = new float[N][3];
        Shot[]    shots      = new Shot[N];
        int[]      rawClips   = new int[N];
        int        rawClipTotal = 0;

        for (int k = 0; k < N; k++) {
            actionRaws[k] = new float[3];
            for (int i = 0; i < 3; i++) {
                actionRaws[k][i] = means[i] + sigmas[i] * randomGaussian();
                if (actionRaws[k][i] < -1f || actionRaws[k][i] > 1f) {
                    rawClips[k]++;
                    rawClipTotal++;
                }
            }
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
            ShotResult result = simulateShot(shots[k], stones, depth - 1);
            float score = 0;
            for (Heuristic h : heuristics) score += h.contribute(h.scoreResult(result));
            rewards[k] = score
                       - rawClips[k] * RAW_CLIP_PENALTY
                       - meanSatCount * MEAN_SAT_PENALTY;
        }

        // 5. Baseline subtraction.
        float meanReward = 0;
        float maxReward  = Float.NEGATIVE_INFINITY;
        float minReward  = Float.POSITIVE_INFINITY;
        for (float rv : rewards) {
            meanReward += rv;
            maxReward   = max(maxReward, rv);
            minReward   = min(minReward, rv);
        }
        meanReward /= N;

        float rewardVar = 0;
        for (float rv : rewards) {
            float d = rv - meanReward;
            rewardVar += d * d;
        }
        float rewardStd = sqrt(rewardVar / max(1, N));

        float[] advantages = new float[N];
        for (int k = 0; k < N; k++) {
            advantages[k] = rewardStd > 1e-6f ? (rewards[k] - meanReward) / rewardStd : 0;
        }

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
                    eliteCount, advSum, true, means, sigmas, new float[3], new float[3],
                    stdAtMaxPct(logStds), stdAtMinPct(logStds), rawLogStdMin(logStds),
                    rawLogStdMax(logStds), logStdBelowMinGap(logStds), logStdAboveMaxGap(logStds),
                    rawClipPct(rawClipTotal, N),
                    100.0f * meanSatCount / 3.0f, maxReward - minReward, 0, 0);
            }
            return;
        }

        // 7. Accumulate weighted gradient.
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

        float entropyGradSum = 0;
        float stdRegGradSum  = 0;
        for (int i = 0; i < 3; i++) {
            float target = targetStd(i);
            float entropyGrad = 0;
            float stdRegGrad = 0;
            if (sigmas[i] < target) {
                entropyGrad = EXPERT_ENTROPY * ((target - sigmas[i]) / max(target, 1e-6f));
            } else if (sigmas[i] > target) {
                stdRegGrad = -STD_REG_STRENGTH
                           * ((sigmas[i] - target) / max(policy.maxStdTanh(i) - target, 1e-6f));
            }
            gradLogStd[i] += entropyGrad + stdRegGrad;
            if (logStds[i] <= policy.logStdMin(i) && gradLogStd[i] < 0) gradLogStd[i] = 0;
            if (logStds[i] >= policy.logStdMax(i) && gradLogStd[i] > 0) gradLogStd[i] = 0;
            gradLogStd[i] = constrain(gradLogStd[i], -GRAD_LOG_STD_LIMIT, GRAD_LOG_STD_LIMIT);
            entropyGradSum += entropyGrad;
            stdRegGradSum  += stdRegGrad;

            if (abs(means[i]) > 0.9f) {
                gradMean[i] -= (means[i] > 0 ? 1 : -1) * MEAN_SAT_GRAD;
            }
            gradMean[i] = constrain(gradMean[i], -GRAD_MEAN_LIMIT, GRAD_MEAN_LIMIT);
        }

        // 8. Backprop.
        policy.backwardMeanAndStd(gradMean, gradLogStd, learningRate);
        policy.clipWeights();

        if (shouldLog) {
            policyDiagnostics.logGradientTrainingStep(
                expertName, policy, hidden,
                updateCount, meanReward, maxReward, minReward,
                eliteCount, advSum, false, means, sigmas, gradMean, gradLogStd,
                stdAtMaxPct(logStds), stdAtMinPct(logStds), rawLogStdMin(logStds),
                rawLogStdMax(logStds), logStdBelowMinGap(logStds), logStdAboveMaxGap(logStds),
                rawClipPct(rawClipTotal, N),
                100.0f * meanSatCount / 3.0f, maxReward - minReward,
                entropyGradSum / 3.0f, stdRegGradSum / 3.0f);
        }
    }

    float targetStd(int dim) {
        if (dim == 0) return TARGET_STD_CURL_TANH;
        if (dim == 1) return TARGET_STD_SPEED_TANH;
        return TARGET_STD_ANGLE_TANH;
    }

    void initializeLogStdToTargets() {
        if (policy == null || !policy.meanAndStd || policy.outputLogStd == null) return;
        for (int i = 0; i < min(3, policy.outputLogStd.neurons.length); i++) {
            Neuron n = policy.outputLogStd.neurons[i];
            for (int j = 0; j < n.weights.length; j++) n.weights[j] = 0;
            n.bias = constrain(log(targetStd(i)), policy.logStdMin(i), policy.logStdMax(i));
        }
        policy.outputLogStd.initAdam(policy.outputLogStd.neurons.length,
                                     policy.outputLogStd.neurons[0].weights.length);
    }

    float rawClipPct(int rawClipTotal, int sampleCount) {
        return 100.0f * rawClipTotal / max(1, sampleCount * 3);
    }

    float stdAtMaxPct(float[] logStds) {
        int count = 0;
        for (int i = 0; i < 3; i++) {
            if (policy.logStdAtMax(i, logStds[i])) count++;
        }
        return 100.0f * count / 3.0f;
    }

    float stdAtMinPct(float[] logStds) {
        int count = 0;
        for (int i = 0; i < 3; i++) {
            if (policy.logStdAtMin(i, logStds[i])) count++;
        }
        return 100.0f * count / 3.0f;
    }

    float rawLogStdMin(float[] logStds) {
        float v = Float.POSITIVE_INFINITY;
        for (int i = 0; i < 3; i++) v = min(v, logStds[i]);
        return v;
    }

    float rawLogStdMax(float[] logStds) {
        float v = Float.NEGATIVE_INFINITY;
        for (int i = 0; i < 3; i++) v = max(v, logStds[i]);
        return v;
    }

    float logStdBelowMinGap(float[] logStds) {
        float gap = 0;
        for (int i = 0; i < 3; i++) gap = max(gap, policy.logStdMin(i) - logStds[i]);
        return max(0, gap);
    }

    float logStdAboveMaxGap(float[] logStds) {
        float gap = 0;
        for (int i = 0; i < 3; i++) gap = max(gap, logStds[i] - policy.logStdMax(i));
        return max(0, gap);
    }

    // Physics simulation. Sets shotsRemainingAfter so FinalScoreHeuristic knows depth.
    ShotResult simulateShot(Shot shot, ArrayList<Stone> layout, int shotsRemainingAfter) {
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

    private int[] topKIndices(float[] values, int k) {
        int n = values.length;
        int[] indices = new int[n];
        for (int i = 0; i < n; i++) indices[i] = i;

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
