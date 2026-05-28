class PolicySearchTraining {
    NeuralPolicy current;
    int   mutationsPerComparison = 5;
    float amountChallengeSet = 0.3; // 0..1, fraction of shots drawn from the fixed challenge set
    float mutationRate       = 0.08;
    float mutationStrength   = 0.04;
    // L2 penalty per comparison: subtracted from each policy's score proportional to sum(w^2).
    // Prevents weights drifting to the clip ceiling. Raise if saturation still grows; lower if model stops improving.
    float weightDecay        = 0.03;
    // Penalise many near-zero ReLU hidden units per layout (fraction 0..1).
    float inactiveHiddenPenalty = 0.1f;
    // Penalise large hidden activations (mean of h^2). Prevents the hidden layer
    // from getting so loud that the output tanh saturates and shots peg at the caps.
    float hiddenMagnitudePenalty = 0.05f;
    ArrayList<Stone>   stones = new ArrayList<Stone>();
    ArrayList<Stone>[] challengeSet;
    int challengeSetSize  = 0;
    int challengeSetIndex = 0;

    RandomState randomState = new RandomState();
    ShotSimilarityPenalty shotSimilarity = new ShotSimilarityPenalty();
    ExplorationNoise explorationNoise = new ExplorationNoise();

    PolicySearchTraining() {
        current = new NeuralPolicy();
    }

    PolicySearchTraining(NeuralPolicy seed) {
        current = seed;
    }

    void reset() {
        // Re-create with the same output caps so the expert identity is preserved.
        current = new NeuralPolicy(current.minCurl, current.maxCurl,
                                   current.minSpeed, current.maxSpeed,
                                   current.minAngleDeg, current.maxAngleDeg);
        challengeSet  = null;
        challengeSetSize  = 0;
        challengeSetIndex = 0;
        explorationNoise.reset();
    }

    void invalidateChallengeSet() {
        challengeSet = null;
        challengeSetSize = 0;
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

    float expertLayoutScore(ExpertShotHeuristic expertH, Shot predicted) {
        return expertH.contribute(expertH.matchScore(predicted));
    }

    // Combined hidden-layer regulariser: pushes activations away from zero
    // (inactive penalty) AND away from being very large (magnitude penalty).
    float hiddenPenalty(NeuralPolicy policy, float[] state) {
        float[] stats = policy.hiddenStats(state);
        return inactiveHiddenPenalty * stats[0]
             + hiddenMagnitudePenalty * stats[1];
    }

  // (1 + mutationsPerComparison) evolution on random/challenge + expert CSV layouts.
    void comparePolicies(ArrayList<Heuristic> heuristics,
            int shotsPerComparison,
            ExpertShotHeuristic expertHeuristic,
            int expertShotsPerComparison,
            ExpertShotDataset expertDataset) {
        boolean hasSimHeuristics = heuristics != null
                && !heuristics.isEmpty()
                && shotsPerComparison > 0;
        boolean hasExpert = expertHeuristic != null
                        && expertDataset != null
                        && expertDataset.count() > 0
                        && expertShotsPerComparison > 0;
        if (!hasSimHeuristics && !hasExpert) return;

        explorationNoise.onComparison();

        if (hasSimHeuristics) {
            ensureChallengeSet(shotsPerComparison);
        } else {
            invalidateChallengeSet();
        }

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

        if (hasSimHeuristics) {
            int randomShots = shotsPerComparison - challengeSetSize;
            if (randomShots < 0) randomShots = 0;

            // --- Random layouts ---
            for (int j = 0; j < randomShots; j++) {
                randomState.randomize(stones, STONES_PER_TEAM);
                float[] state = current.convertState(stones, 1, TEAM_RED);

                ShotResult currentResult = heuristics.get(0).simulate(
                    current, state, stones, explorationNoise);
                float layoutScore = combinedScore(heuristics, currentResult);
                currentShots.add(currentResult.plannedShot);

                if (layoutScore < 0) {
                    challengeSet[challengeSetIndex] = copyLayout(stones);
                    challengeSetIndex = (challengeSetIndex + 1) % challengeSetSize;
                }
                currentScore += layoutScore - hiddenPenalty(current, state);

                for (int m = 0; m < mutationsPerComparison; m++) {
                    ShotResult cr = heuristics.get(0).simulate(
                        candidates[m], state, stones, explorationNoise);
                    candidateScores[m] += combinedScore(heuristics, cr)
                        - hiddenPenalty(candidates[m], state);
                    candidateShots[m].add(cr.plannedShot);
                }
            }

            // --- Challenge-set layouts ---
            for (int h = 0; h < challengeSetSize; h++) {
                ArrayList<Stone> layout = challengeSet[h];
                float[] state = current.convertState(layout, 1, TEAM_RED);

                ShotResult currentResult = heuristics.get(0).simulate(
                    current, state, layout, explorationNoise);
                float currentLayoutScore = combinedScore(heuristics, currentResult);
                currentShots.add(currentResult.plannedShot);
                if (currentLayoutScore > 0) currentLayoutScore *= 2;
                currentScore += currentLayoutScore - hiddenPenalty(current, state);

                for (int m = 0; m < mutationsPerComparison; m++) {
                    ShotResult cr = heuristics.get(0).simulate(
                        candidates[m], state, layout, explorationNoise);
                    float score = combinedScore(heuristics, cr);
                    candidateShots[m].add(cr.plannedShot);
                    if (score > 0) score *= 2;
                    candidateScores[m] += score - hiddenPenalty(candidates[m], state);
                }
            }
        }

        // --- Expert CSV layouts (prediction match only, no physics) ---
        if (hasExpert) {
            for (int j = 0; j < expertShotsPerComparison; j++) {
                ExpertShotEntry entry = expertDataset.randomEntry();
                if (entry == null) break;

                ArrayList<Stone> layout = entry.toLayout();
                int stonesLeft = max(1, round(entry.stonesLeft));
                float[] state = current.convertState(layout, stonesLeft, entry.lastTeam);
                Shot target = new Shot(entry.curl, entry.speed, entry.angle);
                expertHeuristic.setTarget(target);

                Shot currentShot = current.predict(state);
                currentScore += expertLayoutScore(expertHeuristic, currentShot)
                    - hiddenPenalty(current, state);
                currentShots.add(currentShot);

                for (int m = 0; m < mutationsPerComparison; m++) {
                    Shot candidateShot = candidates[m].predict(state);
                    candidateScores[m] += expertLayoutScore(expertHeuristic, candidateShot)
                        - hiddenPenalty(candidates[m], state);
                    candidateShots[m].add(candidateShot);
                }
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

