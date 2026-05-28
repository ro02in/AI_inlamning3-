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
        float scoreReward = 0;

        if (score.team == TEAM_YELLOW) scoreReward = 1;
        if (score.team == TEAM_RED)    scoreReward = -1;
        return scoreReward;
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
        fitness = 1.0 / (1.0 + d);
        return fitness;
    }
}

class CombinedHeuristic extends Heuristic {
    CombinedHeuristic() {
        this.shotsPerComparison = 50;
    }

    @Override
    float simulateShot(NeuralPolicy policy, float[] state, ArrayList<Stone> stones) {
        ArrayList<Stone> simStones = copyLayout(stones);
        Shot shot = policy.predict(state);

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

        // Punishments
        if (fired.pos.x < 0 || fired.pos.x > sheet.SHEET_WIDTH_FT) return -1;
        if (fired.pos.y + fired.radius < sheet.hogY) return -1;
        if (!house.inHouse(fired)) return -0.7;

        // Count yellow stones in house before and after
        int yellowBefore = 0, yellowAfter = 0;
        int redBefore = 0, redAfter = 0;
        for (Stone s : stones) {
            if (s.team == TEAM_YELLOW && house.inHouse(s)) yellowBefore++;
            if (s.team == TEAM_RED    && house.inHouse(s)) redBefore++;
        }
        for (Stone s : simStones) {
            if (s.team == TEAM_YELLOW && house.inHouse(s)) yellowAfter++;
            if (s.team == TEAM_RED    && house.inHouse(s)) redAfter++;
        }
        float allyKnockout  = (yellowBefore - yellowAfter) * -0.2;
        float enemyKnockout = (redBefore - redAfter)       *  0.2;

        // Score signal
        ScoreResult score = house.scoreEnd(simStones);
        float scoreReward = 0;
        if (score.team == TEAM_YELLOW) scoreReward = 1;
        if (score.team == TEAM_RED)    scoreReward = -1;

        // Distance signal
        float distanceReward = 1.0 / (1.0 + house.distanceToButton(fired));

        return scoreReward * 0.8 + distanceReward * 0.2 + allyKnockout + enemyKnockout;
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
            //float[] state = current.convertState(stones, 1, TEAM_RED);
            float[] currentState = current.convertState(stones, 1, TEAM_RED);
            float[] mutatedState = mutated.convertState(stones, 1, TEAM_RED);
            currentScoreSum += heuristic.simulateShot(current, currentState, stones);
            mutatedScoreSum += heuristic.simulateShot(mutated, mutatedState, stones);
        }

        if (mutatedScoreSum > currentScoreSum) {
            current = mutated.copy();
            current.saveToFile("data/policy.txt"); // Sparar Senaste
        }
        mutated = current.copy();
        mutated.mutate(mutationRate, mutationStrength);
    }
}
