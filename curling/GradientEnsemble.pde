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
            trainers[i].updateStep(typeHeuristics[i], depth);
        }
    }

    void reset() {
        for (int i = 0; i < count; i++) trainers[i].reset();
    }

    // Use the shot-type selector to pick the best expert, then return its mean shot.
    // sampleMode=true -> weighted-random from selector probabilities (PLAY).
    // sampleMode=false -> argmax (TEST, deterministic).
    Shot bestShot(ArrayList<Stone> layout, int stonesLeft, int lastTeam,
                  ShotTypeSelector selector, boolean logScores, boolean sampleMode) {
        NeuralPolicy refPolicy = trainers[0].policy;
        float[] state = refPolicy.convertState(layout, stonesLeft, lastTeam);

        int chosen;
        float[] probs = new float[count];
        if (selector != null) {
            probs = selector.probs(state);
            chosen = sampleMode ? selector.sampleFromProbs(probs)
                                : selector.argmaxFromProbs(probs);
        } else {
            chosen = 0;
        }

        Shot best = trainers[chosen].policy.predictMean(
            trainers[chosen].policy.convertState(layout, stonesLeft, lastTeam));

        if (logScores) {
            String mode = sampleMode ? "PLAY" : "TEST";
            diagnostics.logEnsembleInference(names, probs, chosen,
                trainers[chosen].policy, layout, stonesLeft, lastTeam, best, mode);
        }
        return best;
    }

    NeuralPolicy anyPolicy() { return trainers[0].policy; }

    // Return the mean shot for expert i given the board state.
    Shot expertMeanShot(int expertIdx, ArrayList<Stone> layout, int stonesLeft, int lastTeam) {
        NeuralPolicy p = trainers[expertIdx].policy;
        float[] state = p.convertState(layout, stonesLeft, lastTeam);
        return p.predictMean(state);
    }
}
