// Plays the remaining shots of a curling end using the trained ensemble + selector,
// returning the final score from YELLOW's perspective.
//
// Alternation after yellow's shot:
//   shotsRemaining=1 → red throws once, end; shotsRemaining=2 → red, yellow; etc.
// All shots use argmax selector + deterministic mean for low-variance reward signal.
//
// Perspective flip for red: stones are re-labeled so that yellow's network interprets
// the board from red's point of view (team labels swapped), then the fired stone is
// re-labeled back to TEAM_RED.
class SelfPlayRollout {

    // Simulate `shotsRemaining` alternating plies starting with RED (opponent),
    // then score the final board from YELLOW's perspective.
    float rollout(ArrayList<Stone> board, int shotsRemaining) {
        ArrayList<Stone> current = copyLayout(board);
        int nextTeam = TEAM_RED; // first ply after yellow's shot is opponent (red)
        int remaining = shotsRemaining;

        while (remaining > 0) {
            current = playOnePly(current, nextTeam, remaining);
            nextTeam = (nextTeam == TEAM_RED) ? TEAM_YELLOW : TEAM_RED;
            remaining--;
        }

        ScoreResult final_ = house.scoreEnd(current);
        return final_.yellowFitness();
    }

    // Play one ply for `team`. Returns board after the shot settles.
    private ArrayList<Stone> playOnePly(ArrayList<Stone> board, int team, int remaining) {
        // Compute stonesLeft for the acting team so convertState is correct.
        int stonesLeft = max(1, (int) ceil(remaining / 2.0));

        // Build the state. For both perspectives we use convertState with the
        // board's actual stone labels but pass the acting team as if it were yellow.
        // To keep it simple we always call convertState as if yellow is the actor:
        //   - If team == YELLOW: use board as-is, lastTeam = RED.
        //   - If team == RED: flip team labels so red looks like yellow to the net.
        ArrayList<Stone> stateBoard;
        int lastTeam;
        if (team == TEAM_YELLOW) {
            stateBoard = board;
            lastTeam   = TEAM_RED;
        } else {
            stateBoard = flipTeams(board);
            lastTeam   = TEAM_YELLOW; // after flip, what was yellow is now "red"
        }

        NeuralPolicy refPolicy = gradientEnsemble.anyPolicy();
        float[] state = refPolicy.convertState(stateBoard, stonesLeft, lastTeam);

        // Use the selector (argmax for low variance).
        int expertIdx = (shotSelector != null)
            ? shotSelector.argmax(state)
            : 0;

        NeuralPolicy expert = gradientEnsemble.trainers[expertIdx].policy;
        float[] expertState = expert.convertState(stateBoard, stonesLeft, lastTeam);
        Shot shot = expert.predictMean(expertState);

        // Fire the shot on the real board (not the flipped one).
        ArrayList<Stone> simStones = copyLayout(board);
        PVector h = sheet.hackWorld();
        Stone fired = new Stone(h.x, h.y, TEAM_YELLOW); // always fires as yellow physically
        fired.curl = constrain(shot.curl, -1, 1);
        fired.vel.set(sin(shot.angle) * shot.speed, cos(shot.angle) * shot.speed);
        simStones.add(fired);

        for (int step = 0; step < 100000; step++) {
            physics.step(simStones, DT);
            boolean anyMoving = false;
            for (Stone s : simStones) if (s.isMoving()) { anyMoving = true; break; }
            if (!anyMoving) break;
        }

        // Relabel fired stone to the actual acting team.
        fired.team = team;
        return simStones;
    }

    // Swap TEAM_RED <-> TEAM_YELLOW labels for perspective flip.
    private ArrayList<Stone> flipTeams(ArrayList<Stone> board) {
        ArrayList<Stone> flipped = new ArrayList<Stone>();
        for (Stone s : board) {
            Stone c = new Stone(s.pos.x, s.pos.y,
                                s.team == TEAM_RED ? TEAM_YELLOW : TEAM_RED);
            c.hogPassed = s.hogPassed;
            flipped.add(c);
        }
        return flipped;
    }

    private ArrayList<Stone> copyLayout(ArrayList<Stone> layout) {
        ArrayList<Stone> copy = new ArrayList<Stone>();
        for (Stone s : layout) {
            Stone c = new Stone(s.pos.x, s.pos.y, s.team);
            c.hogPassed = s.hogPassed;
            copy.add(c);
        }
        return copy;
    }
}
