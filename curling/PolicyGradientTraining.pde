// Oliwer Carpman, Rafal Galinski, Robin Karim
// Simple gradient training for one shot expert.
//
// Each update:
//   1. Build a random board from the curriculum depth.
//   2. Let the policy predict one shot.
//   3. Try many random nearby shots.
//   4. Simulate and score those shots.
//   5. Move the policy output toward the best sampled shots.
class PolicyGradientTraining {
    NeuralPolicy policy;
    String expertName = "";

    int   shotsPerUpdate = 50;
    float eliteFraction  = 0.25f;
    float learningRate   = 0.0004f;
    int   logEvery       = 100;

    final float EXPLORE_CURL_TANH  = 0.35f;
    final float EXPLORE_SPEED_TANH = 0.30f;
    final float EXPLORE_ANGLE_TANH = 0.30f;
    final float GRAD_LIMIT         = 1.25f;

    int updateCount = 0;

    RandomState randomState = new RandomState();
    ArrayList<Stone> stones = new ArrayList<Stone>();

    PolicyGradientTraining() {
        policy = new NeuralPolicy();
    }

    PolicyGradientTraining(NeuralPolicy seed) {
        policy = seed;
    }

    void reset() {
        if (policy != null) {
            policy = new NeuralPolicy(policy.minCurl, policy.maxCurl,
                                      policy.minSpeed, policy.maxSpeed,
                                      policy.minAngleDeg, policy.maxAngleDeg);
        } else {
            policy = new NeuralPolicy();
        }
        updateCount = 0;
    }

    void setPolicy(NeuralPolicy p, int loadedUpdateCount) {
        if (p == null) return;
        policy = p;
        if (loadedUpdateCount >= 0) updateCount = loadedUpdateCount;
    }

    void updateStep(ArrayList<Heuristic> heuristics, int depth) {
        updateStep(heuristics, depth, expertName);
    }

    void updateStep(ArrayList<Heuristic> heuristics, int depth, String curriculumExpert) {
        if (heuristics == null || heuristics.isEmpty()) return;

        int stonesToPlace = TOTAL_STONES - depth;
        randomState.randomizeForExpert(stones, stonesToPlace, curriculumExpert);

        int stonesLeft = max(1, (int) ceil(depth / 2.0));
        int remainingShotsAfterCurrent = max(0, depth - 1);
        float[] state = policy.convertState(stones, stonesLeft, TEAM_RED);

        float[] predictedRaw = policy.rawOutput(state);

        int n = shotsPerUpdate;
        float[][] sampledRaw = new float[n][3];
        float[] rewards = new float[n];

        for (int k = 0; k < n; k++) {
            sampledRaw[k][0] = constrain(predictedRaw[0] + randomGaussian() * EXPLORE_CURL_TANH, -1f, 1f);
            sampledRaw[k][1] = constrain(predictedRaw[1] + randomGaussian() * EXPLORE_SPEED_TANH, -1f, 1f);
            sampledRaw[k][2] = constrain(predictedRaw[2] + randomGaussian() * EXPLORE_ANGLE_TANH, -1f, 1f);

            ShotResult result = simulateShot(policy.shotFromRaw(sampledRaw[k]),
                                             stones,
                                             remainingShotsAfterCurrent);
            float score = 0;
            for (Heuristic h : heuristics) {
                score += h.contribute(h.scoreResult(result));
            }
            rewards[k] = score;
        }

        int eliteCount = max(1, round(n * eliteFraction));
        int[] eliteIdx = topKIndices(rewards, eliteCount);

        float[] targetRaw = new float[3];
        for (int idx : eliteIdx) {
            for (int i = 0; i < 3; i++) {
                targetRaw[i] += sampledRaw[idx][i];
            }
        }
        for (int i = 0; i < 3; i++) {
            targetRaw[i] /= eliteCount;
        }

        float[] gradOutput = new float[3];
        for (int i = 0; i < 3; i++) {
            gradOutput[i] = constrain(targetRaw[i] - predictedRaw[i], -GRAD_LIMIT, GRAD_LIMIT);
        }

        policy.backwardOutput(gradOutput, learningRate);
        policy.clipWeights();
        updateCount++;

        if (updateCount == 1 || updateCount % logEvery == 0) {
            println("Train " + expertName
                + " step=" + updateCount
                + " depth=" + depth
                + " best=" + nf(rewards[eliteIdx[0]], 0, 3)
                + " grad=[" + nf(gradOutput[0], 0, 3)
                + "," + nf(gradOutput[1], 0, 3)
                + "," + nf(gradOutput[2], 0, 3) + "]");
        }
    }

    ShotResult simulateShot(Shot shot, ArrayList<Stone> layout,
                            int remainingShotsAfterCurrent) {
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

        return new ShotResult(simStones, before, fired, scoreBefore, shot,
                              remainingShotsAfterCurrent);
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
