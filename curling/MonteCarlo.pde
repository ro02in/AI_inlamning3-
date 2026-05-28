class MonteCarloAI {
  final int   SAMPLES    = 200;
  final float SPEED_MIN  = 0.0;
  final float SPEED_MAX  = UI.SPEED_MAX;
  final float ANGLE_MAX  = 0.18; // 18 grader

  Shot chooseBestShot(ArrayList<Stone> currentStones) {
    Shot best = null;
    float bestScore = -10000;

    for (int i = 0; i < SAMPLES; i++) {
      Shot candidate = randomShot();
      float score = simulate(currentStones, candidate);
      if (score > bestScore) {
        bestScore = score;
        best = candidate;
      }
    }
    return best;
  }

  Shot randomShot() {
    float curl = random(-1, 1);
    float speed = random(SPEED_MIN, SPEED_MAX);
    float angle = random(-ANGLE_MAX, ANGLE_MAX);
    return new Shot(curl, speed, angle);
  }

  float simulate(ArrayList<Stone> currentStones, Shot shot) {
    ArrayList<Stone> sim = copyStones(currentStones);

    PVector h = sheet.hackWorld();
    Stone fired = new Stone(h.x, h.y, TEAM_YELLOW);
    fired.curl = constrain(shot.curl, -1, 1);
    PVector dir = new PVector(sin(shot.angle), cos(shot.angle));
    fired.vel.set(PVector.mult(dir, shot.speed));
    sim.add(fired);

    for (int step = 0; step < 3000; step++) {
      physics.step(sim, DT);
      boolean moving = false;
      for (Stone s : sim){
        if (s.isMoving()) {
         moving = true; 
         break; 
        }
      }
      if (!moving) break;
    }

    return scoreStones(sim);
  }

  float scoreStones(ArrayList<Stone> sim) {
    float score = 0;
    float closestYellow = 10000;
    float closestRed = 10000;

    for (Stone s : sim) {
      if (!house.inHouse(s)) continue;
      float d = house.distanceToButton(s);
      if (s.team == TEAM_YELLOW) {
        score += 1;
        if (d < closestYellow) closestYellow = d;
      } else {
        score -= 1;
        if (d < closestRed) closestRed = d;
      }
    }

    if (closestYellow < closestRed) score += 2;
    return score;
  }

  ArrayList<Stone> copyStones(ArrayList<Stone> src) {
    ArrayList<Stone> result = new ArrayList<Stone>();
    for (Stone s : src) result.add(s.copy());
    return result;
  }
}
