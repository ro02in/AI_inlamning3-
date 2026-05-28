class PolicyGradientSample {
    float[] state;
    float   reward;
    Shot    mean;
    Shot    noisy;

    PolicyGradientSample(float[] state, float reward, Shot mean, Shot noisy) {
        this.state  = state;
        this.reward = reward;
        this.mean   = mean;
        this.noisy  = noisy;
    }
}

class PolicyGradientTraining {
    NeuralPolicy policy;

    float learningRate        = 0.001f;
    float weightDecay         = 0.002f;
    float outputBiasDecay     = 0.002f;
    float baseline            = 0;
    float baselineDecay       = 0.99f;
    float maxGrad             = 1.0f;
    float maxAdvantage        = 20.0f;
    float contrastWeight      = 3.0f;
    float contrastStateThresh = 0.12f;
    float contrastShotThresh  = 0.08f;

    float sigmaCurl      = 0.15f;
    float sigmaSpeed     = 3.0f;
    float sigmaAngle     = 0.08f;
    float sigmaDecay     = 0.9995f;
    float sigmaCurlMin   = 0.08f;
    float sigmaSpeedMin  = 1.5f;
    float sigmaAngleMin  = 0.04f;

    float amountChallengeSet = 0.5f;

    ArrayList<Stone>   stones = new ArrayList<Stone>();
    ArrayList<Stone>[] challengeSet;
    int challengeSetSize  = 0;
    int challengeSetIndex = 0;

    RandomState randomState = new RandomState();
    ShotSimilarityPenalty shotSimilarity = new ShotSimilarityPenalty();

    PolicyGradientTraining() {
        policy = new NeuralPolicy();
    }

    void reset() {
        policy = new NeuralPolicy();
        baseline = 0;
        sigmaCurl  = 0.15f;
        sigmaSpeed = 3.0f;
        sigmaAngle = 0.08f;
        challengeSet     = null;
        challengeSetSize  = 0;
        challengeSetIndex = 0;
    }

    ArrayList<Stone> copyLayout(ArrayList<Stone> layout) {
        ArrayList<Stone> copy = new ArrayList<Stone>();
        for (Stone s : layout) {
            Stone c = new Stone(s.pos.x, s.pos.y, s.team);
            c.hogPassed = s.hogPassed;
            copy.add(c);
        }
        return copy;
    }

    void ensureChallengeSet(int layoutsPerStep) {
        if (challengeSet != null) return;
        challengeSetSize = max(1, round(layoutsPerStep * amountChallengeSet));
        challengeSet = new ArrayList[challengeSetSize];
        for (int i = 0; i < challengeSetSize; i++) {
            challengeSet[i] = new ArrayList<Stone>();
            randomState.randomize(challengeSet[i], STONES_PER_TEAM);
        }
        challengeSetIndex = 0;
    }

    float combinedScore(ArrayList<Heuristic> heuristics, ShotResult result) {
        float total = 0;
        for (Heuristic h : heuristics) total += h.contribute(h.scoreResult(result));
        return total;
    }

    Shot noisyShot(Shot mean) {
        return new Shot(
            mean.curl  + randomGaussian() * sigmaCurl,
            mean.speed + randomGaussian() * sigmaSpeed,
            mean.angle + randomGaussian() * sigmaAngle
        );
    }

    void accumulatePolicyGrad(Shot mean, Shot noisy, float advantage) {
        if (!shotIsFinite(mean) || !shotIsFinite(noisy)) return;

        advantage = constrain(advantage, -maxAdvantage, maxAdvantage);

        float dLogProbDCurl  = (noisy.curl  - mean.curl)  / (sigmaCurl * sigmaCurl);
        float dLogProbDSpeed = (noisy.speed - mean.speed) / (sigmaSpeed * sigmaSpeed);
        float dLogProbDAngle = (noisy.angle - mean.angle) / (sigmaAngle * sigmaAngle);

        policy.backwardFromShotGrad(
            -advantage * dLogProbDCurl,
            -advantage * dLogProbDSpeed,
            -advantage * dLogProbDAngle
        );
    }

    boolean shotIsFinite(Shot s) {
        return Float.isFinite(s.curl) && Float.isFinite(s.speed) && Float.isFinite(s.angle);
    }

    float stateDistance(float[] a, float[] b) {
        float sum = 0;
        for (int i = 0; i < a.length; i++) {
            float d = a[i] - b[i];
            sum += d * d;
        }
        return sqrt(sum / a.length);
    }

    float shotDistance(Shot a, Shot b) {
        float dc = (a.curl - b.curl) / 0.5f;
        float ds = (a.speed - b.speed) / 12.0f;
        float da = (a.angle - b.angle) / 0.35f;
        return sqrt(dc * dc + ds * ds + da * da);
    }

    // When two layouts differ but the policy fires the same mean shot, push outputs apart.
    // Uniform batch penalties cancel in advantage; this adds per-sample gradient direction.
    void accumulateContrastiveGrad(PolicyGradientSample a, PolicyGradientSample b) {
        float stateDist = stateDistance(a.state, b.state);
        if (stateDist < contrastStateThresh) return;

        float shotDist = shotDistance(a.mean, b.mean);
        if (shotDist > contrastShotThresh) return;

        float push = contrastWeight * stateDist * (1.0f - shotDist / contrastShotThresh);
        float dirCurl  = a.state[0] - b.state[0];
        float dirSpeed = a.state[1] - b.state[1];
        float dirAngle = a.state[4] - b.state[4];
        if (abs(dirCurl) + abs(dirSpeed) + abs(dirAngle) < 0.01f) {
            dirCurl  = random(-1, 1);
            dirSpeed = random(-1, 1);
            dirAngle = random(-1, 1);
        }

        policy.forward(a.state);
        policy.backwardFromShotGrad(push * dirCurl, push * dirSpeed * 0.1f, push * dirAngle * 0.5f);
    }

    float perSampleSimilarityPenalty(int index, ArrayList<Shot> meanShots) {
        if (meanShots.size() < 2) return 0;

        float totalDist = 0;
        int pairs = 0;
        Shot mine = meanShots.get(index);
        for (int j = 0; j < meanShots.size(); j++) {
            if (j == index) continue;
            totalDist += shotDistance(mine, meanShots.get(j));
            pairs++;
        }
        float avgDist = totalDist / pairs;
        float batchPenalty = shotSimilarity.penalty(meanShots);
        float lackOfDiversity = constrain(1.0f - avgDist / contrastShotThresh, 0, 1);
        return batchPenalty * lackOfDiversity / meanShots.size();
    }

    PolicyGradientSample runLayout(ArrayList<Heuristic> heuristics,
                                   ArrayList<Stone> layout, boolean challengeLayout) {
        float[] state = policy.convertState(layout, 1, TEAM_RED);
        policy.forward(state);
        Shot mean = policy.predictFromCache();
        Shot noisy = noisyShot(mean);

        ShotResult result = heuristics.get(0).simulateShot(noisy, layout);
        float reward = combinedScore(heuristics, result);
        if (challengeLayout && reward > 0) reward *= 2;

        return new PolicyGradientSample(state, reward, mean, noisy);
    }

    void trainStep(ArrayList<Heuristic> heuristics) {
        if (heuristics.isEmpty()) return;
        if (!policy.hasFiniteWeights()) {
            println("REINFORCE: weights became non-finite — resetting policy.");
            reset();
            return;
        }

        int layoutsPerStep = heuristics.get(0).shotsPerComparison;
        ensureChallengeSet(layoutsPerStep);

        ArrayList<PolicyGradientSample> samples = new ArrayList<PolicyGradientSample>();
        ArrayList<Shot> meanShots = new ArrayList<Shot>();

        int randomLayouts = layoutsPerStep - challengeSetSize;
        if (randomLayouts < 0) randomLayouts = 0;

        for (int j = 0; j < randomLayouts; j++) {
            randomState.randomize(stones, STONES_PER_TEAM);
            PolicyGradientSample sample = runLayout(heuristics, stones, false);
            samples.add(sample);
            meanShots.add(sample.mean);

            if (sample.reward < 0) {
                challengeSet[challengeSetIndex] = copyLayout(stones);
                challengeSetIndex = (challengeSetIndex + 1) % challengeSetSize;
            }
        }

        for (int h = 0; h < challengeSetSize; h++) {
            PolicyGradientSample sample = runLayout(heuristics, challengeSet[h], true);
            samples.add(sample);
            meanShots.add(sample.mean);
        }

        if (samples.isEmpty()) return;

        policy.zeroGrads();
        for (int i = 0; i < samples.size(); i++) {
            PolicyGradientSample sample = samples.get(i);
            float adjustedReward = sample.reward - perSampleSimilarityPenalty(i, meanShots);
            baseline = baselineDecay * baseline + (1 - baselineDecay) * adjustedReward;
            float advantage = adjustedReward - baseline;

            policy.forward(sample.state);
            accumulatePolicyGrad(policy.predictFromCache(), sample.noisy, advantage);
        }

        for (int i = 0; i < samples.size(); i++) {
            for (int j = i + 1; j < samples.size(); j++) {
                accumulateContrastiveGrad(samples.get(i), samples.get(j));
            }
        }

        policy.scaleGrads(1.0f / samples.size());
        policy.clipGrads(maxGrad);

        if (!policy.hasFiniteGrads()) {
            println("REINFORCE: non-finite gradients — skipping update.");
            policy.zeroGrads();
            return;
        }

        policy.applyGrads(learningRate, weightDecay, outputBiasDecay);
        policy.clipWeights();
    }

    void decaySigmas(int trainingStep) {
        if (trainingStep > 0 && trainingStep % 100 == 0) {
            sigmaCurl  = max(sigmaCurlMin,  sigmaCurl  * sigmaDecay);
            sigmaSpeed = max(sigmaSpeedMin, sigmaSpeed * sigmaDecay);
            sigmaAngle = max(sigmaAngleMin, sigmaAngle * sigmaDecay);
        }
    }
}
