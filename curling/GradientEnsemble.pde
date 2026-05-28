// Holds one PolicyGradientTraining per expert specialisation.
// Training: one updateStep per expert per frame.
// Inference: simulate each expert's mean shot, pick the best heuristic score.
class GradientEnsemble {
    String[] names;
    PolicyGradientTraining[] trainers;
    int count;
    PolicyDiagnostics diagnostics = new PolicyDiagnostics();

    GradientEnsemble() {
        NeuralPolicy seed = new NeuralPolicy();
        names = new String[]{
            "Draw", "CurlR", "CurlL", "Takeout", "Guard", "Freeze"
        };
        NeuralPolicy[] seeds = {
            seed.expertDraw(true),
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
    }

    void trainAll(ArrayList<Heuristic> heuristics, int shotsPerUpdate) {
        for (int i = 0; i < count; i++) {
            trainers[i].shotsPerUpdate = shotsPerUpdate;
            trainers[i].updateStep(heuristics);
        }
    }

    void reset() {
        for (int i = 0; i < count; i++) {
            trainers[i].reset();
        }
    }

    Shot bestShot(ArrayList<Stone> layout, int stonesLeft, int lastTeam,
                  Heuristic scorer, boolean logScores) {
        Shot best = null;
        float bestScore = Float.NEGATIVE_INFINITY;
        String bestName = "";
        float satSum = 0;
        float deadSum = 0;
        float outSatSum = 0;

        if (logScores) println("--- PG expert scores ---");

        for (int i = 0; i < count; i++) {
            NeuralPolicy p = trainers[i].policy;
            float[] state = p.convertState(layout, stonesLeft, lastTeam);
            float[] hidden = p.hiddenLayer.feedForward(state);
            float[] means = p.outputLayer.feedForward(hidden);
            float[] logStds = p.outputLogStd.feedForward(hidden);
            float[] health = diagnostics.hiddenHealthPcts(hidden, p.HIDDEN_ACTIVATION);
            float satPct = health[0];
            float deadPct = health[1];
            float outSatPct = diagnostics.outputSaturationPct(means, p.OUTPUT_ACTIVATION);
            satSum += satPct;
            deadSum += deadPct;
            outSatSum += outSatPct;
            Shot shot = p.predictMean(state);

            float score = 0;
            if (scorer != null) {
                ShotResult result = scorer.simulate(p, state, layout, null);
                score = scorer.scoreResult(result);
            }

            if (logScores) {
                diagnostics.logGradientExpertLine(names[i], p, hidden, means, logStds,
                                                  shot, score, satPct, deadPct, outSatPct);
            }

            if (score > bestScore) {
                bestScore = score;
                best = shot;
                bestName = names[i];
            }
        }

        if (logScores) {
            NeuralPolicy any = trainers[0].policy;
            boolean reluHidden = (any.HIDDEN_ACTIVATION == ActivationKind.RELU);
            boolean tanhOut = (any.OUTPUT_ACTIVATION == ActivationKind.TANH);
            String avgLine = "Avg hidden sat: " + nf(satSum / count, 0, 1) + "%";
            if (reluHidden) {
                avgLine += "  inactive: " + nf(deadSum / count, 0, 1) + "%";
            }
            if (tanhOut) {
                avgLine += "  outSat: " + nf(outSatSum / count, 0, 1) + "%";
            }
            println(avgLine);
            if (best != null) {
                println("Chosen shot: " + bestName
                    + "  score=" + nf(bestScore, 0, 3)
                    + "  curl=" + nf(best.curl, 0, 3)
                    + "  spd=" + nf(best.speed, 0, 1)
                    + "  ang=" + nf(degrees(best.angle), 0, 1) + "deg");
            } else {
                println("Chosen shot: (none)");
            }
        }
        return best;
    }

    NeuralPolicy anyPolicy() {
        return trainers[0].policy;
    }
}
