// Oliwer Carpman, Rafal Galinski, Robin Karpman
// =============================================================
// House - tee / ring geometry and end-of-end scoring.
// =============================================================
// Ring sizes and tee position come from the global Sheet (feet).
// The sheet is drawn procedurally; House only handles distances
// and scoring rules.
// =============================================================

class ScoreResult {
  int team;    // TEAM_RED, TEAM_YELLOW, or -1 for blank end
  int points;  // 0 if blank

  ScoreResult(int team, int points) {
    this.team   = team;
    this.points = points;
  }

  float yellowFitness() {
    if (team == TEAM_YELLOW) return points;
    if (team == TEAM_RED)    return -points;
    return 0;
  }

  float diffFrom(ScoreResult before) {
    return yellowFitness() - before.yellowFitness();
  }
}

class House {
  final PVector BUTTON;
  final float OUTER_RING;
  final float MID_RING;
  final float INNER_RING;
  final float BUTTON_R;

  House() {
    BUTTON     = new PVector(sheet.centerX, sheet.teeY);
    OUTER_RING = sheet.HOUSE_12_R_FT;
    MID_RING   = sheet.HOUSE_8_R_FT;
    INNER_RING = sheet.HOUSE_4_R_FT;
    BUTTON_R   = sheet.BUTTON_R_FT;
  }

  float distanceToButton(Stone s) {
    return PVector.dist(s.pos, BUTTON);
  }

  boolean inHouse(Stone s) {
    return distanceToButton(s) <= OUTER_RING + s.radius;
  }

  // -----------------------------------------------------------
  // scoreEnd: real-curling scoring for a single end.
  //   1. Only stones that cleared the hog line count toward score.
  //   2. Among those, keep stones inside the 12-foot ring.
  //   3. Sort by distance to button; winner = closest stone's team.
  //   4. Count consecutive same-team stones before first opponent.
  //   5. Blank if nobody in the house with hog cleared.
  // -----------------------------------------------------------
  ScoreResult scoreEnd(ArrayList<Stone> stones) {
    ArrayList<Stone> inPlay = new ArrayList<Stone>();
    for (Stone s : stones) {
      if (s.hogPassed && inHouse(s)) inPlay.add(s);
    }

    if (inPlay.isEmpty()) return new ScoreResult(-1, 0);

    for (int i = 1; i < inPlay.size(); i++) {
      Stone si = inPlay.get(i);
      float di = distanceToButton(si);
      int j = i - 1;
      while (j >= 0 && distanceToButton(inPlay.get(j)) > di) {
        inPlay.set(j + 1, inPlay.get(j));
        j--;
      }
      inPlay.set(j + 1, si);
    }

    int winner = inPlay.get(0).team;
    int points = 0;
    for (Stone s : inPlay) {
      if (s.team == winner) points++;
      else break;
    }
    return new ScoreResult(winner, points);
  }
}
