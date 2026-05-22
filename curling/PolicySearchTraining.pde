abstract class Heuristic {
    int shotsPerComparison = 30;
    abstract float simulateShot(NeuralPolicy policy, float[] state, ArrayList<Stone> stones);

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

// Heuristic that simulates the shot and scores it based on the end result (win/loss/tie).
class ScoreHeuristic extends Heuristic {
    ScoreHeuristic() {
        this.shotsPerComparison = 50;
    }

    @Override
    float simulateShot(NeuralPolicy policy, float[] state, ArrayList<Stone> stones) {
        ArrayList<Stone> simStones = copyLayout(stones);
        Shot shot = policy.predict(state);

        // Manually fire yellow stone from the hack
        PVector h = sheet.hackWorld();
        Stone fired = new Stone(h.x, h.y, TEAM_YELLOW);
        fired.curl = constrain(shot.curl, -1, 1);
        fired.vel.set(sin(shot.angle) * shot.speed, cos(shot.angle) * shot.speed);
        simStones.add(fired);

        // Step physics until all stones stop (cap at 10000 steps to prevent infinite loops)
        for (int step = 0; step < 100000; step++) {
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
}

// Heuristic that scores a shot only based on how close it is to the button.
class CloseToButtonHeuristic extends Heuristic {
    CloseToButtonHeuristic() {
        this.shotsPerComparison = 50;
    }
    @Override
    float simulateShot(NeuralPolicy policy, float[] state, ArrayList<Stone> stones) {
        ArrayList<Stone> simStones = copyLayout(stones);
        Shot shot = policy.predict(state);

        // Manually fire yellow stone from the hack
        PVector h = sheet.hackWorld();
        Stone fired = new Stone(h.x, h.y, TEAM_YELLOW);
        fired.curl = constrain(shot.curl, -1, 1);
        fired.vel.set(sin(shot.angle) * shot.speed, cos(shot.angle) * shot.speed);
        simStones.add(fired);

        // Step physics until all stones stop (cap at 10000 steps to prevent infinite loops)
        for (int step = 0; step < 1000000; step++) {
            physics.step(simStones, DT);
            boolean anyMoving = false;
            for (Stone s : simStones) if (s.isMoving()) { anyMoving = true; break; }
            if (!anyMoving) break;
        }

        float fitness = 0;
        // Score: how close is the closest stone to the button? Closer = better, farther = worse
        if (fired.pos.x < 0 || fired.pos.x > sheet.SHEET_WIDTH_FT) fitness -=100;
        if (fired.pos.y + fired.radius < sheet.hogY) fitness -=50;  // never reached house end
        float d = house.distanceToButton(fired);
        fitness -= d; // closer to button = higher fitness
        if (house.inHouse(fired)) fitness += 10;
        if (d < sheet.BUTTON_R_FT) fitness += 20;
        return fitness;
    }
}

class PolicySearchTraining {
    NeuralPolicy current;
    NeuralPolicy mutated;
    float mutationRate = 0.1;
    float mutationStrength = 0.3;
    ArrayList<Stone> stones = new ArrayList<Stone>();

    RandomState randomState = new RandomState();

    PolicySearchTraining() {
        current = new NeuralPolicy();
        mutated = current.copy();
        mutated.mutate(mutationRate, mutationStrength);
    }

    // Compare policies in a simulated scenario where AI has the last stone with simulated random stone layouts.
    void comparePolicies(Heuristic heuristic) {
        float currentScoreSum = 0;
        float mutatedScoreSum = 0;
        for (int i = 0; i < heuristic.shotsPerComparison; i++) {
            // Set up random stone layout (all but last yellow stone)
            randomState.randomize(stones, STONES_PER_TEAM);

            // Build state and predict shot (1 yellow stone left to throw, then game ends)
            float[] state = current.convertState(stones, 1, TEAM_RED);
            currentScoreSum += heuristic.simulateShot(current, state, stones);
            mutatedScoreSum += heuristic.simulateShot(mutated, state, stones);
        }

        if (mutatedScoreSum > currentScoreSum) {
            current = mutated.copy();
            current.saveToFile("data/policy.txt"); // Sparar Senaste
        }
        mutated = current.copy();
        mutated.mutate(mutationRate, mutationStrength);
    }
}
