// 8-expert gradient ensemble. Each expert specialises in one shot type
// and receives its own tailored heuristic list.
//
// Expert index mapping (used by ShotTypeSelector):
//   0 Draw         1 DrawCurlR  2 DrawCurlL
//   3 CurlR        4 CurlL      5 Takeout
//   6 Guard        7 Freeze
class GradientEnsemble {
    String[] names;
    PolicyGradientTraining[] trainers;
    ArrayList<Heuristic>[] typeHeuristics;
    int count;
    PolicyDiagnostics diagnostics = new PolicyDiagnostics();
    FinalScoreHeuristic finalHeuristic = new FinalScoreHeuristic();
    float[] lastAdjustedSelectorProbs;
    float[] lastRolloutRankProbs;
    float[] lastFinalDecisionProbs;
    float[] lastDecisionScores;
    float[] lastRolloutScores;
    float[] lastTypeScores;
    boolean[] lastDecisionCandidates;

    GradientEnsemble() {
        NeuralPolicy seed = new NeuralPolicy();
        names = new String[]{
            "Draw", "DrawCurlR", "DrawCurlL",
            "CurlR", "CurlL",
            "Takeout",
            "Guard", "Freeze"
        };
        NeuralPolicy[] seeds = {
            seed.expertDraw(true),
            seed.expertDrawCurlRight(true),
            seed.expertDrawCurlLeft(true),
            seed.expertCurlRight(true),
            seed.expertCurlLeft(true),
            seed.expertTakeout(true),
            seed.expertGuard(true),
            seed.expertFreeze(true)
        };
        count = seeds.length;
        trainers = new PolicyGradientTraining[count];
        for (int i = 0; i < count; i++) {
            trainers[i] = new PolicyGradientTraining(seeds[i]);
            trainers[i].expertName = names[i];
        }

        // Per-type heuristic lists (finalHeuristic appended to all)
        typeHeuristics = new ArrayList[count];
        DrawHeuristic    drawH    = new DrawHeuristic();
        CurlHeuristic    curlH    = new CurlHeuristic();
        TakeoutHeuristic takeoutH = new TakeoutHeuristic();
        GuardHeuristic   guardH   = new GuardHeuristic();
        FreezeHeuristic  freezeH  = new FreezeHeuristic();

        for (int i = 0; i < count; i++) {
            typeHeuristics[i] = new ArrayList<Heuristic>();
        }
        // Draw, DrawCurlR, DrawCurlL
        typeHeuristics[0].add(drawH);
        typeHeuristics[1].add(drawH);
        typeHeuristics[2].add(drawH);
        // CurlR, CurlL
        typeHeuristics[3].add(curlH);
        typeHeuristics[4].add(curlH);
        // Takeout
        typeHeuristics[5].add(takeoutH);
        // Guard
        typeHeuristics[6].add(guardH);
        // Freeze
        typeHeuristics[7].add(freezeH);

        // Append FinalScoreHeuristic to every expert's list
        for (int i = 0; i < count; i++) {
            typeHeuristics[i].add(finalHeuristic);
        }
    }

    // Train all experts. Each draws its own random depth from the curriculum cap.
    void trainAll(int depthCap, int shotsPerUpdate) {
        for (int i = 0; i < count; i++) {
            int depth = max(1, (int) random(1, depthCap + 1));
            trainers[i].shotsPerUpdate = shotsPerUpdate;
            trainers[i].updateStep(typeHeuristics[i], depth, names[i]);
        }
    }

    void reset() {
        for (int i = 0; i < count; i++) trainers[i].reset();
    }

    // Use the selector to filter likely expert types, then evaluate their mean shots
    // with final-score rollout and pick the best deterministic action.
    Shot bestShot(ArrayList<Stone> layout, int stonesLeft, int lastTeam,
                  ShotTypeSelector selector, boolean logScores, boolean sampleMode) {
        NeuralPolicy refPolicy = trainers[0].policy;
        float[] state = refPolicy.convertState(layout, stonesLeft, lastTeam);

        int chosen;
        float[] probs = new float[count];
        if (selector != null) {
            probs = selector.probs(state);
        } else {
            for (int i = 0; i < count; i++) probs[i] = 1.0f / count;
        }

        if (USE_ROLLOUT_DECISION) {
            chosen = chooseByRollout(layout, stonesLeft, lastTeam, probs, selector);
        } else if (selector != null) {
            if (sampleMode) {
                chosen = selector.sampleFromProbs(probs);
            } else if (TEST_TOP_K_ENABLED) {
                chosen = selector.sampleTopKFromProbs(probs, TEST_TOP_K_EPS);
            } else {
                chosen = selector.argmaxFromProbs(probs);
            }
        } else {
            chosen = 0;
        }

        Shot best = expertMeanShot(chosen, layout, stonesLeft, lastTeam);

        if (logScores) {
            String mode = sampleMode ? "PLAY" : "TEST";
            String selectorMode = USE_ROLLOUT_DECISION ? "ROLLOUT_BEST"
                                : (sampleMode ? "SAMPLE"
                                : (TEST_TOP_K_ENABLED ? "TEST_TOP_K" : "ARGMAX"));
            diagnostics.logEnsembleInference(this, probs, chosen,
                layout, stonesLeft, lastTeam, best, mode, selectorMode);
        }
        return best;
    }

    int chooseByRollout(ArrayList<Stone> layout, int stonesLeft, int lastTeam,
                        float[] probs, ShotTypeSelector selector) {
        lastAdjustedSelectorProbs = selector != null
            ? selector.relativeOdds(probs)
            : uniformProbs(count);
        lastRolloutRankProbs = new float[count];
        lastFinalDecisionProbs = new float[count];
        lastDecisionScores = new float[count];
        lastRolloutScores = new float[count];
        lastTypeScores = new float[count];
        lastDecisionCandidates = new boolean[count];

        int fallback = argmaxArray(probs);
        int chosen = fallback;
        for (int i = 0; i < count; i++) {
            Shot shot = expertMeanShot(i, layout, stonesLeft, lastTeam);
            ShotResult result = simulateCandidateShot(shot, layout, rolloutShotsRemainingAfter(stonesLeft));
            lastRolloutScores[i] = finalHeuristic.contribute(finalHeuristic.scoreResult(result));
            lastTypeScores[i] = typeHeuristicScore(i, result);
        }

        assignTopFourRolloutOdds();

        float bestScore = Float.NEGATIVE_INFINITY;
        for (int i = 0; i < count; i++) {
            lastFinalDecisionProbs[i] = DECISION_NN_WEIGHT * lastAdjustedSelectorProbs[i]
                                      + DECISION_FINAL_SCORE_WEIGHT * lastRolloutRankProbs[i];
            lastDecisionScores[i] = lastFinalDecisionProbs[i];
            lastDecisionCandidates[i] = lastFinalDecisionProbs[i] > 0;

            if (lastDecisionScores[i] > bestScore) {
                bestScore = lastDecisionScores[i];
                chosen = i;
            }
        }
        if (bestScore <= 0) chosen = fallback;
        return chosen;
    }

    void assignTopFourRolloutOdds() {
        int[] order = rankedIndices(lastRolloutScores);
        float[] weights = { 0.50f, 0.25f, 0.125f, 0.125f };
        for (int rank = 0; rank < min(4, order.length); rank++) {
            lastRolloutRankProbs[order[rank]] = weights[rank];
        }
    }

    int[] rankedIndices(float[] values) {
        int[] idx = new int[values.length];
        for (int i = 0; i < values.length; i++) idx[i] = i;
        for (int i = 1; i < idx.length; i++) {
            int key = idx[i];
            int j = i - 1;
            while (j >= 0 && values[idx[j]] < values[key]) {
                idx[j + 1] = idx[j];
                j--;
            }
            idx[j + 1] = key;
        }
        return idx;
    }

    float[] uniformProbs(int n) {
        float[] p = new float[n];
        for (int i = 0; i < n; i++) p[i] = 1.0f / max(1, n);
        return p;
    }

    int argmaxArray(float[] values) {
        int best = 0;
        for (int i = 1; i < values.length; i++) {
            if (values[i] > values[best]) best = i;
        }
        return best;
    }

    float typeHeuristicScore(int expertIdx, ShotResult result) {
        float score = 0;
        for (Heuristic h : typeHeuristics[expertIdx]) {
            if (h instanceof FinalScoreHeuristic) continue;
            score += h.contribute(h.scoreResult(result));
        }
        return score;
    }

    int rolloutShotsRemainingAfter(int stonesLeft) {
        return max(0, (stonesLeft - 1) * 2);
    }

    ShotResult simulateCandidateShot(Shot shot, ArrayList<Stone> layout, int shotsRemainingAfter) {
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

    ArrayList<Stone> copyLayout(ArrayList<Stone> layout) {
        ArrayList<Stone> copy = new ArrayList<Stone>();
        for (Stone s : layout) {
            Stone c = new Stone(s.pos.x, s.pos.y, s.team);
            c.hogPassed = s.hogPassed;
            copy.add(c);
        }
        return copy;
    }

    NeuralPolicy anyPolicy() { return trainers[0].policy; }

    // Return the mean shot for expert i given the board state.
    Shot expertMeanShot(int expertIdx, ArrayList<Stone> layout, int stonesLeft, int lastTeam) {
        NeuralPolicy p = trainers[expertIdx].policy;
        float[] state = p.convertState(layout, stonesLeft, lastTeam);
        return p.predictMean(state);
    }
}
