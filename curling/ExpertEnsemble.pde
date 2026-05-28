// Holds one PolicySearchTraining per expert specialisation.
// Training: one comparePolicies step per expert per frame.
// Inference: simulate each expert, pick the one with the best heuristic score.
class ExpertEnsemble {
    String[] names;
    PolicySearchTraining[] trainers;
    int count;
    PolicyDiagnostics diagnostics = new PolicyDiagnostics();

    ExpertEnsemble() {
        NeuralPolicy seed = new NeuralPolicy();
        names = new String[]{
            "Draw", "CurlR", "CurlL", "Takeout", "Guard", "Freeze"
        };
        NeuralPolicy[] seeds = {
            seed.expertDraw(),
            seed.expertCurlRight(),
            seed.expertCurlLeft(),
            seed.expertTakeout(),
            seed.expertGuard(),
            seed.expertFreeze()
        };
        count = seeds.length;
        trainers = new PolicySearchTraining[count];
        for (int i = 0; i < count; i++) {
            trainers[i] = new PolicySearchTraining(seeds[i]);
        }
    }

    // Train all experts one comparison step each.
    void trainAll(ArrayList<Heuristic> heuristics,
                int shotsPerPrediction,
                ExpertShotHeuristic expertHeuristic,
                int expertShotsPerPrediction,
                ExpertShotDataset expertDataset) {
        for (int i = 0; i < count; i++) {
            trainers[i].comparePolicies(heuristics, shotsPerPrediction,
                                        expertHeuristic, expertShotsPerPrediction,
                                        expertDataset);
        }
    }

    // Reset all experts (preserves output caps).
    void reset() {
        for (int i = 0; i < count; i++) {
            trainers[i].reset();
        }
    }

    // Simulate each expert on the given layout and return the best shot.
    // Logs per-expert scores to console when logScores = true.
    Shot bestShot(ArrayList<Stone> layout, int stonesLeft, int lastTeam,
                Heuristic scorer, boolean logScores) {
        Shot best = null;
        float bestScore = Float.NEGATIVE_INFINITY;
        String bestName = "";
        float satSum = 0;
        float deadSum = 0;
        float outSatSum = 0;

        if (logScores) println("--- Expert scores ---");

        for (int i = 0; i < count; i++) {
            NeuralPolicy p = trainers[i].current;
            float[] state = p.convertState(layout, stonesLeft, lastTeam);
            float[] hidden = p.hiddenLayer.feedForward(state);
            float[] outputs = p.outputLayer.feedForward(hidden);
            float[] health = diagnostics.hiddenHealthPcts(hidden, p.HIDDEN_ACTIVATION);
            float satPct = health[0];
            float deadPct = health[1];
            float outSatPct = diagnostics.outputSaturationPct(outputs, p.OUTPUT_ACTIVATION);
            satSum += satPct;
            deadSum += deadPct;
            outSatSum += outSatPct;
            Shot shot = p.predict(state);

            float score = 0;
            if (scorer != null) {
                ShotResult result = scorer.simulate(p, state, layout, null);
                score = scorer.scoreResult(result);
            }

            if (logScores) {
                String inactiveStr = (p.HIDDEN_ACTIVATION == ActivationKind.RELU)
                    ? "  inactive=" + nf(deadPct, 0, 1) + "%"
                    : "";
                String outSatStr = (p.OUTPUT_ACTIVATION == ActivationKind.TANH)
                    ? "  outSat=" + nf(outSatPct, 0, 1) + "%"
                    : "";
                println("  " + names[i]
                    + "  curl=" + nf(shot.curl, 0, 3)
                    + "  spd=" + nf(shot.speed, 0, 1)
                    + "  ang=" + nf(degrees(shot.angle), 0, 1) + "deg"
                    + "  hSat=" + nf(satPct, 0, 1) + "%"
                    + inactiveStr
                    + outSatStr
                    + "  score=" + nf(score, 0, 4));
            }

            if (score > bestScore) {
                bestScore = score;
                best = shot;
                bestName = names[i];
            }
        }

        if (logScores) {
            NeuralPolicy any = trainers[0].current;
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

    // Convenience: any policy (for state encoding or slider preview).
    NeuralPolicy anyPolicy() {
        return trainers[0].current;
    }
}
