class PolicySearchTraining {
    NeuralPolicy current;
    NeuralPolicy mutated;
    float mutationRate = 0.1;
    int gamesPerComparison = 10;
    ArrayList<Stone> stones = new ArrayList<Stone>();

    RandomState randomState = new RandomState();

    PolicySearchTraining() {
        current = new NeuralPolicy();
        mutated = current.copy();
        mutated.mutate(mutationRate);
    }

    void comparePolicies() {

        float currentScoreSum = 0;
        float mutatedScoreSum = 0;
        for (int i = 0; i < gamesPerComparison; i++) {
            // Set up random stone layout (all but last yellow stone)
            randomState.randomize(stones, STONES_PER_TEAM);

            // Build state and predict shot (1 yellow stone left to throw, then game ends)
            float[] state = current.convertState(stones, 1, TEAM_RED);
            currentScoreSum += simulateGame(current, state);
            mutatedScoreSum += simulateGame(mutated, state);
        }

        if (mutatedScoreSum > currentScoreSum) {
            current = mutated.copy();
        }
        mutated = current.copy();
        mutated.mutate(mutationRate);
    }

    float simulateGame(NeuralPolicy policy, float[] state) {
        ArrayList<Stone> simStones = copyLayout(stones);
        Shot shot = policy.predict(state);

        // Manually fire yellow stone from the hack
        PVector h = sheet.hackWorld();
        Stone fired = new Stone(h.x, h.y, TEAM_YELLOW);
        fired.curl = constrain(shot.curl, -1, 1);
        fired.vel.set(sin(shot.angle) * shot.speed, cos(shot.angle) * shot.speed);
        simStones.add(fired);

        // Step physics until all stones stop (cap at 10000 steps to prevent infinite loops)
        for (int step = 0; step < 10000; step++) {
            physics.step(simStones, DT);
            boolean anyMoving = false;
            for (Stone s : simStones) if (s.isMoving()) { anyMoving = true; break; }
            if (!anyMoving) break;
        }

        // Score: positive = good (yellow wins = AI wins), negative = bad (red wins = AI loses)
        ScoreResult score = house.scoreEnd(simStones);
        if (score.team == TEAM_YELLOW) return score.points;
        if (score.team == TEAM_RED)    return -score.points;
        return 0; // tie
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
}
