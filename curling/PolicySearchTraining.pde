class PolicySearchTraining {
    NeuralPolicy current;
    int   mutationsPerComparison = 5;
    float amountChallengeSet = 0.5; // 0..1, fraction of shots drawn from the fixed challenge set
    float mutationRate       = 0.08;
    float mutationStrength   = 0.04;
    // L2 penalty per comparison: subtracted from each policy's score proportional to sum(w^2).
    // Prevents weights drifting to the clip ceiling. Raise if saturation still grows; lower if model stops improving.
    float weightDecay        = 0.02;
    ArrayList<Stone>   stones = new ArrayList<Stone>();
    ArrayList<Stone>[] challengeSet;
    int challengeSetSize  = 0;
    int challengeSetIndex = 0;

    RandomState randomState = new RandomState();
    ShotSimilarityPenalty shotSimilarity = new ShotSimilarityPenalty();

    PolicySearchTraining() {
        current = new NeuralPolicy();
    }

    void reset() {
        current = new NeuralPolicy();
        challengeSet  = null;
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

    void ensureChallengeSet(int shotsPerComparison) {
        if (challengeSet != null) return;
        challengeSetSize = max(1, round(shotsPerComparison * amountChallengeSet));
        challengeSet = new ArrayList[challengeSetSize];
        for (int i = 0; i < challengeSetSize; i++) {
            challengeSet[i] = new ArrayList<Stone>();
            randomState.randomize(challengeSet[i], STONES_PER_TEAM);
        }
        challengeSetIndex = 0;
    }

    // Combined weighted score for all heuristics on one pre-simulated shot.
    float combinedScore(ArrayList<Heuristic> heuristics, ShotResult result) {
        float total = 0;
        for (Heuristic h : heuristics) total += h.contribute(h.scoreResult(result));
        return total;
    }

    // (1 + mutationsPerComparison) evolution: keep current unless a mutant scores higher on same random layouts.
    // Physics runs once per policy per layout; all heuristics score the same positions.
    void comparePolicies(ArrayList<Heuristic> heuristics) {
        if (heuristics.isEmpty()) return;

        ensureChallengeSet(heuristics.get(0).shotsPerComparison);

        NeuralPolicy[] candidates = new NeuralPolicy[mutationsPerComparison];
        float[]        candidateScores = new float[mutationsPerComparison];
        ArrayList<Shot>  currentShots = new ArrayList<Shot>();
        ArrayList<Shot>[] candidateShots = new ArrayList[mutationsPerComparison];
        for (int m = 0; m < mutationsPerComparison; m++) {
            candidates[m] = current.copy();
            candidates[m].mutate(mutationRate, mutationStrength);
            candidateScores[m] = 0;
            candidateShots[m] = new ArrayList<Shot>();
        }

        float currentScore = 0;

        // Use first heuristic's shotsPerComparison as the layout budget.
        int randomShots = heuristics.get(0).shotsPerComparison - challengeSetSize;
        if (randomShots < 0) randomShots = 0;

        // --- Random layouts ---
        for (int j = 0; j < randomShots; j++) {
            randomState.randomize(stones, STONES_PER_TEAM);
            float[] state = current.convertState(stones, 1, TEAM_RED);

            // Simulate once per policy, then score cheaply with every heuristic.
            ShotResult currentResult = heuristics.get(0).simulate(current, state, stones);
            float layoutScore = combinedScore(heuristics, currentResult);
            currentShots.add(currentResult.plannedShot);

            if (layoutScore < 0) {
                challengeSet[challengeSetIndex] = copyLayout(stones);
                challengeSetIndex = (challengeSetIndex + 1) % challengeSetSize;
            }
            currentScore += layoutScore;

            for (int m = 0; m < mutationsPerComparison; m++) {
                ShotResult cr = heuristics.get(0).simulate(candidates[m], state, stones);
                candidateScores[m] += combinedScore(heuristics, cr);
                candidateShots[m].add(cr.plannedShot);
            }
        }

        // --- Challenge-set layouts ---
        for (int h = 0; h < challengeSetSize; h++) {
            ArrayList<Stone> layout = challengeSet[h];
            float[] state = current.convertState(layout, 1, TEAM_RED);

            ShotResult currentResult = heuristics.get(0).simulate(current, state, layout);
            float currentLayoutScore = combinedScore(heuristics, currentResult);
            currentShots.add(currentResult.plannedShot);
            if (currentLayoutScore > 0) currentLayoutScore *= 2;
            currentScore += currentLayoutScore;

            for (int m = 0; m < mutationsPerComparison; m++) {
                ShotResult cr = heuristics.get(0).simulate(candidates[m], state, layout);
                float score = combinedScore(heuristics, cr);
                candidateShots[m].add(cr.plannedShot);
                if (score > 0) score *= 2;
                candidateScores[m] += score;
            }
        }

        // Penalise firing the same curl/angle (same direction) on every layout.
        currentScore -= shotSimilarity.penalty(currentShots);
        for (int m = 0; m < mutationsPerComparison; m++) {
            candidateScores[m] -= shotSimilarity.penalty(candidateShots[m]);
        }

        // Apply L2 weight penalty once per comparison to discourage weight growth.
        currentScore -= weightDecay * current.weightL2();
        for (int m = 0; m < mutationsPerComparison; m++) {
            candidateScores[m] -= weightDecay * candidates[m].weightL2();
        }

        // Keep best candidate if it beats current.
        int bestIdx = -1;
        float bestScore = currentScore;
        for (int m = 0; m < mutationsPerComparison; m++) {
            if (candidateScores[m] > bestScore) {
                bestScore = candidateScores[m];
                bestIdx = m;
            }
        }
        if (bestIdx >= 0) {
            current = candidates[bestIdx].copy();
        }
        current.clipWeights();
    }
}
