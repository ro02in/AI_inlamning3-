// Holds the outcome of one simulated shot: final stone positions, the fired stone,
// and the board score before the shot (needed by ScoreHeuristic).
class ShotResult {
    ArrayList<Stone> stonesAfter;
    Stone            fired;
    ScoreResult      scoreBefore;
    Shot             plannedShot;

    ShotResult(ArrayList<Stone> stonesAfter, Stone fired, ScoreResult scoreBefore, Shot plannedShot) {
        this.stonesAfter = stonesAfter;
        this.fired       = fired;
        this.scoreBefore = scoreBefore;
        this.plannedShot = plannedShot;
    }
}

abstract class Heuristic {
    float weight = 1.0; // relative importance when combining heuristics
    float scale  = 1.0; // divide raw score by this to normalise magnitude across heuristics

    // Normalised contribution: weight * (raw / scale).
    // For single-heuristic use scale=1 preserves the raw value unchanged.
    float contribute(float raw) {
        return weight * (raw / scale);
    }

    // Run physics for one shot and return the result.
    // Called once per policy per layout so all heuristics score the same positions.
    ShotResult simulate(NeuralPolicy policy, float[] state, ArrayList<Stone> layout) {
        ArrayList<Stone> simStones = copyLayout(layout);
        ScoreResult scoreBefore = house.scoreEnd(simStones);
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

        return new ShotResult(simStones, fired, scoreBefore, shot);
    }

    // Score a pre-simulated result (no physics; cheap).
    abstract float scoreResult(ShotResult result);

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
        this.scale = 3.0; // typical delta range -3..+3 → normalised to ~-1..+1
        this.weight = 2.5; // make this heuristic more important relative to others
    }

    @Override
    float scoreResult(ShotResult result) {
        ScoreResult scoreAfter = house.scoreEnd(result.stonesAfter);
        float score = scoreAfter.diffFrom(result.scoreBefore);

        return score;
    }
}

// Heuristic that scores a shot only based on how close it is to the button.
class CloseToButtonHeuristic extends Heuristic {
    CloseToButtonHeuristic() {
        this.scale = 50.0;
        this.weight = 0.8;
    }

    @Override
    float scoreResult(ShotResult result) {
        Stone fired = result.fired;
        float fitness = 0;
        float d = house.distanceToButton(fired);
        fitness -= d; // closer to button = higher fitness
        if (d < sheet.BUTTON_R_FT) fitness += 20;
        return fitness;
    }
}

// Heuristic that scores a shot based on various penalties and bonuses related to the fired stone's position.
class PenaltyHeuristic extends Heuristic {
    PenaltyHeuristic() {
        this.scale = 50.0;
        this.weight = 1.2;
    }

    @Override
    float scoreResult(ShotResult result) {
        Stone fired = result.fired;
        float fitness = 0;
        if (fired.pos.x < 0 || fired.pos.x > sheet.SHEET_WIDTH_FT) fitness -= 100;
        if (fired.pos.y + fired.radius < sheet.hogY || fired.pos.y + fired.radius > sheet.backFarY) fitness -= 100; // too short (hog not cleared) or fully past far back line
        return fitness;
    }
}

// Matches a predicted shot directly against an expert target shot (no physics simulation).
class ExpertShotHeuristic extends Heuristic {
    Shot targetShot;

    ExpertShotHeuristic() {
        this.weight = 5.0;
        this.scale  = 1.0;
    }

    void setTarget(Shot target) {
        targetShot = target;
    }

    // Higher is better; 0 is perfect match.
    float matchScore(Shot predicted) {
        if (targetShot == null || predicted == null) return 0;
        float curlErr  = abs(predicted.curl - targetShot.curl);
        float speedErr = abs(predicted.speed - targetShot.speed) / UI.SPEED_MAX;
        // Normalize against +/- 10 degrees used by NeuralPolicy output mapping.
        float angleErr = abs(predicted.angle - targetShot.angle) / (PI / 18.0);
        return -(curlErr + speedErr + angleErr);
    }

    @Override
    float scoreResult(ShotResult result) {
        return matchScore(result.plannedShot);
    }
}
