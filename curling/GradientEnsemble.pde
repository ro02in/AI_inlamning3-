// Ensemble of 8 specialised experts. Each expert predicts one kind of shot.
//
// Expert index mapping:
//   0 Draw         1 DrawCurlR  2 DrawCurlL
//   3 CurlR        4 CurlL      5 Takeout
//   6 Guard        7 Freeze
class GradientEnsemble {
    String[] names;
    PolicyGradientTraining[] trainers;
    ArrayList<Heuristic>[] typeHeuristics;
    int count;
    FinalScoreHeuristic finalHeuristic = new FinalScoreHeuristic();

    GradientEnsemble() {
        NeuralPolicy seed = new NeuralPolicy();
        names = new String[]{
            "Draw", "DrawCurlR", "DrawCurlL",
            "CurlR", "CurlL",
            "Takeout",
            "Guard", "Freeze"
        };
        NeuralPolicy[] seeds = {
            seed.expertDraw(),
            seed.expertDrawCurlRight(),
            seed.expertDrawCurlLeft(),
            seed.expertCurlRight(),
            seed.expertCurlLeft(),
            seed.expertTakeout(),
            seed.expertGuard(),
            seed.expertFreeze()
        };
        count = seeds.length;
        trainers = new PolicyGradientTraining[count];
        for (int i = 0; i < count; i++) {
            trainers[i] = new PolicyGradientTraining(seeds[i]);
            trainers[i].expertName = names[i];
        }

        typeHeuristics = new ArrayList[count];
        DrawHeuristic    drawH    = new DrawHeuristic();
        CurlHeuristic    curlH    = new CurlHeuristic();
        TakeoutHeuristic takeoutH = new TakeoutHeuristic();
        GuardHeuristic   guardH   = new GuardHeuristic();
        FreezeHeuristic  freezeH  = new FreezeHeuristic();

        for (int i = 0; i < count; i++) {
            typeHeuristics[i] = new ArrayList<Heuristic>();
        }
        typeHeuristics[0].add(drawH);
        typeHeuristics[1].add(drawH);
        typeHeuristics[2].add(drawH);
        typeHeuristics[3].add(curlH);
        typeHeuristics[4].add(curlH);
        typeHeuristics[5].add(takeoutH);
        typeHeuristics[6].add(guardH);
        typeHeuristics[7].add(freezeH);

        for (int i = 0; i < count; i++) {
            typeHeuristics[i].add(finalHeuristic);
        }
    }

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

    Shot bestShot(ArrayList<Stone> layout, int stonesLeft, int lastTeam,
                  int remainingShotsAfterCurrent,
                  ShotTypeSelector selector, boolean logScores, boolean sampleMode) {
        int chosen = 0;
        Shot best = null;
        float bestUtility = Float.NEGATIVE_INFINITY;
        float[] probs = selectorProbs(layout, stonesLeft, lastTeam, selector);
        int[] candidates = topExperts(probs, min(PLAY_CANDIDATE_EXPERTS, count));

        for (int i = 0; i < candidates.length; i++) {
            int expertIdx = candidates[i];
            Shot shot = expertMeanShot(expertIdx, layout, stonesLeft, lastTeam);
            float utility = averageUtilityAfterShot(shot, layout,
                                                    remainingShotsAfterCurrent,
                                                    selector);
            if (utility > bestUtility) {
                bestUtility = utility;
                chosen = expertIdx;
                best = shot;
            }
        }

        if (logScores) {
            String mode = sampleMode ? "sample" : "deterministic";
            println("NN shot (" + mode + "): " + names[chosen]
                + " EU=" + nf(bestUtility, 0, 3)
                + " curl=" + nf(best.curl, 0, 3)
                + " speed=" + nf(best.speed, 0, 1)
                + " angle=" + nf(degrees(best.angle), 0, 1));
        }
        return best;
    }

    NeuralPolicy anyPolicy() {
        return trainers[0].policy;
    }

    Shot expertMeanShot(int expertIdx, ArrayList<Stone> layout, int stonesLeft, int lastTeam) {
        NeuralPolicy p = trainers[expertIdx].policy;
        float[] state = p.convertState(layout, stonesLeft, lastTeam);
        return p.predictMean(state);
    }

    float[] selectorProbs(ArrayList<Stone> stateBoard, int stonesLeft, int lastTeam,
                          ShotTypeSelector selector) {
        if (selector == null) {
            float[] uniform = new float[count];
            for (int i = 0; i < count; i++) uniform[i] = 1.0f / count;
            return uniform;
        }
        float[] state = anyPolicy().convertState(stateBoard, stonesLeft, lastTeam);
        return selector.probs(state);
    }

    int[] topExperts(float[] probs, int limit) {
        limit = max(1, min(limit, probs.length));
        int[] result = new int[limit];
        boolean[] used = new boolean[probs.length];

        for (int rank = 0; rank < limit; rank++) {
            int bestIdx = 0;
            float bestProb = Float.NEGATIVE_INFINITY;
            for (int i = 0; i < probs.length; i++) {
                if (!used[i] && probs[i] > bestProb) {
                    bestProb = probs[i];
                    bestIdx = i;
                }
            }
            result[rank] = bestIdx;
            used[bestIdx] = true;
        }
        return result;
    }

    float averageUtilityAfterShot(Shot candidate, ArrayList<Stone> layout,
                                  int remainingShotsAfterCurrent,
                                  ShotTypeSelector selector) {
        int sims = max(1, SIMS_PER_EXPERT_SHOT);
        float total = 0;
        for (int i = 0; i < sims; i++) {
            total += expectedUtilityAfterShot(candidate, layout,
                                              remainingShotsAfterCurrent,
                                              selector);
        }
        return total / sims;
    }

    float expectedUtilityAfterShot(Shot candidate, ArrayList<Stone> layout,
                                   int remainingShotsAfterCurrent,
                                   ShotTypeSelector selector) {
        return longTermUtilityAfterYellowShot(candidate, layout,
                                              remainingShotsAfterCurrent,
                                              selector);
    }

    float longTermUtilityAfterYellowShot(Shot candidate, ArrayList<Stone> layout,
                                         int remainingShotsAfterCurrent,
                                         ShotTypeSelector selector) {
        ArrayList<Stone> afterCandidate = simulateOneShot(layout, candidate, TEAM_YELLOW);
        return longTermUtilityFromBoard(afterCandidate,
                                        remainingShotsAfterCurrent,
                                        TEAM_RED,
                                        selector);
    }

    float longTermUtilityFromBoard(ArrayList<Stone> board,
                                   int remainingShots,
                                   int nextTeam,
                                   ShotTypeSelector selector) {
        remainingShots = max(0, remainingShots);
        if (remainingShots == 0) {
            return house.scoreEnd(board).yellowFitness();
        }
        return expectedUtilityAfterNextOpponentShot(board, remainingShots,
                                                    nextTeam, selector);
    }

    float expectedUtilityAfterNextOpponentShot(ArrayList<Stone> board,
                                               int remainingShots,
                                               int nextTeam,
                                               ShotTypeSelector selector) {
        int stonesLeft = max(1, (int) ceil(remainingShots / 2.0));

        ArrayList<Stone> selectorBoard = boardForTeamPerspective(board, nextTeam);
        int lastTeam = lastTeamFromShooterPerspective();
        float[] probs = selectorProbs(selectorBoard, stonesLeft, lastTeam, selector);
        int[] nextExperts = topExperts(probs, min(NEXT_UTILITY_EXPERTS, count));

        float probSum = 0;
        for (int i = 0; i < nextExperts.length; i++) {
            probSum += max(0, probs[nextExperts[i]]);
        }
        boolean useUniformWeights = probSum <= 0;

        float expected = 0;
        for (int i = 0; i < nextExperts.length; i++) {
            int expertIdx = nextExperts[i];
            float weight = useUniformWeights
                ? 1.0f / nextExperts.length
                : max(0, probs[expertIdx]) / probSum;
            float utility = averageNextExpertUtility(expertIdx, board,
                                                     remainingShots,
                                                     nextTeam, selector);
            expected += weight * utility;
        }
        return expected;
    }

    float averageNextExpertUtility(int expertIdx, ArrayList<Stone> board,
                                   int remainingShots,
                                   int nextTeam,
                                   ShotTypeSelector selector) {
        int sims = max(1, SIMS_PER_NEXT_EXPERT);
        int stonesLeft = max(1, (int) ceil(remainingShots / 2.0));
        float total = 0;

        for (int i = 0; i < sims; i++) {
            Shot nextShot = shotForTeam(expertIdx, board, nextTeam, stonesLeft);
            ArrayList<Stone> afterNext = simulateOneShot(board, nextShot, nextTeam);
            total += rolloutToEndDeterministic(afterNext,
                                               max(0, remainingShots - 1),
                                               otherTeam(nextTeam),
                                               selector);
        }
        return total / sims;
    }

    float rolloutToEndDeterministic(ArrayList<Stone> startBoard,
                                    int remainingShots,
                                    int nextTeam,
                                    ShotTypeSelector selector) {
        ArrayList<Stone> board = copyLayout(startBoard);
        int team = nextTeam;
        int remaining = max(0, remainingShots);

        while (remaining > 0) {
            int stonesLeft = max(1, (int) ceil(remaining / 2.0));
            ArrayList<Stone> selectorBoard = boardForTeamPerspective(board, team);
            int lastTeam = lastTeamFromShooterPerspective();
            float[] probs = selectorProbs(selectorBoard, stonesLeft, lastTeam, selector);
            int expertIdx = topExperts(probs, 1)[0];

            Shot shot = shotForTeam(expertIdx, board, team, stonesLeft);
            board = simulateOneShot(board, shot, team);
            team = otherTeam(team);
            remaining--;
        }
        return house.scoreEnd(board).yellowFitness();
    }

    Shot shotForTeam(int expertIdx, ArrayList<Stone> board, int team, int stonesLeft) {
        ArrayList<Stone> stateBoard = boardForTeamPerspective(board, team);
        int lastTeam = lastTeamFromShooterPerspective();
        return expertMeanShot(expertIdx, stateBoard, stonesLeft, lastTeam);
    }

    ArrayList<Stone> boardForTeamPerspective(ArrayList<Stone> board, int team) {
        return team == TEAM_YELLOW ? board : flipTeams(board);
    }

    int lastTeamFromShooterPerspective() {
        return TEAM_RED;
    }

    int otherTeam(int team) {
        return team == TEAM_RED ? TEAM_YELLOW : TEAM_RED;
    }

    ArrayList<Stone> simulateOneShot(ArrayList<Stone> layout, Shot shot, int team) {
        ArrayList<Stone> simStones = copyLayout(layout);

        PVector h = sheet.hackWorld();
        Stone fired = new Stone(h.x, h.y, team);
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
        return simStones;
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

    ArrayList<Stone> flipTeams(ArrayList<Stone> board) {
        ArrayList<Stone> flipped = new ArrayList<Stone>();
        for (Stone s : board) {
            Stone c = new Stone(s.pos.x, s.pos.y,
                                s.team == TEAM_RED ? TEAM_YELLOW : TEAM_RED);
            c.hogPassed = s.hogPassed;
            flipped.add(c);
        }
        return flipped;
    }
}
