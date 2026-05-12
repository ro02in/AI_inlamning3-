// =============================================================
// House - geometry of the rings and end-of-end scoring.
// =============================================================
// The map PNG already draws the rings, so House does not render
// them; it only owns the BUTTON location and the ring radii used
// by the scoring rules. Real scoring is implemented in TODO 7.
// =============================================================

class ScoreResult {
  int team;    // TEAM_RED, TEAM_YELLOW, or -1 for blank end
  int points;  // 0 if blank

  ScoreResult(int team, int points) {
    this.team   = team;
    this.points = points;
  }
}

class House {
  // Button (tee) center in world coordinates.
  final PVector BUTTON = new PVector(540, 360);

  // Ring radii (world units), measured from the actual map pixels
  // (sample along y=360 from x=540 outward).
  final float OUTER_RING = 290;   // 12-foot (outer edge of blue ring)
  final float MID_RING   = 213;   //  8-foot (outer edge of inner white)
  final float INNER_RING = 137;   //  4-foot (outer edge of red center)
  final float BUTTON_R   =  59;   //  central white "button"

  float distanceToButton(Stone s) {
    return PVector.dist(s.pos, BUTTON);
  }

  boolean inHouse(Stone s) {
    return distanceToButton(s) <= OUTER_RING + s.radius;
  }

  // -----------------------------------------------------------
  // scoreEnd: real-curling scoring for a single end.
  //   1. Drop stones whose nearest edge is outside the 12-foot ring.
  //   2. Sort the remaining stones by distance to the button.
  //   3. The team of the closest stone is the winner.
  //   4. They score 1 point for each of their stones closer to the
  //      button than the nearest opponent stone (i.e. the run of
  //      same-team stones at the head of the sorted list).
  //   5. If no stone is in the house, the end is blank.
  // -----------------------------------------------------------
  ScoreResult scoreEnd(ArrayList<Stone> stones) {
    ArrayList<Stone> inPlay = new ArrayList<Stone>();
    for (Stone s : stones) if (inHouse(s)) inPlay.add(s);

    if (inPlay.isEmpty()) return new ScoreResult(-1, 0);

    // Insertion sort by distanceToButton (small N -> trivial).
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
