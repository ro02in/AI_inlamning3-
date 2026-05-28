class RandomState {

    float hogY   = 45.0;
    float backY  = 72.0;
    float sheetW = 14.5;
    float r      = 0.475;
    int maxTries = 200;

    // Original: place up to stonesPerTeam red + (stonesPerTeam-1) yellow.
    // Called from UI setRndStones and startAiTest.
    void randomize(ArrayList<Stone> stones, int stonesPerTeam) {
        stones.clear();
        for (int i = 0; i < stonesPerTeam; i++) {
            if (random(1) > 0.3) placeStone(stones, TEAM_RED);
        }
        for (int i = 0; i < stonesPerTeam - 1; i++) {
            if (random(1) > 0.3) placeStone(stones, TEAM_YELLOW);
        }
    }

    // Curriculum variant: place exactly stonesToPlace stones following throw order
    // (RED first, then YELLOW, alternating). Each has a 70% chance of being in play.
    // stonesToPlace = TOTAL_STONES - depth, so deeper curriculum = fewer pre-placed stones.
    void randomizeForDepth(ArrayList<Stone> stones, int stonesToPlace) {
        stones.clear();
        for (int i = 0; i < stonesToPlace; i++) {
            int team = (i % 2 == 0) ? TEAM_RED : TEAM_YELLOW;
            if (random(1) > 0.3) {
                placeStone(stones, team);
            }
        }
    }

    private void placeStone(ArrayList<Stone> stones, int team) {
        PVector p = new PVector();
        int tries = 0;
        do {
            p.x = random(r, sheetW - r);
            p.y = random(hogY + r, backY - r);
            tries++;
        } while (overlaps(p, stones) && tries < maxTries);

        Stone s = new Stone(p.x, p.y, team);
        s.hogPassed = true;
        stones.add(s);
    }

    private boolean overlaps(PVector p, ArrayList<Stone> stones) {
        for (Stone s : stones) {
            if (PVector.dist(p, s.pos) < 2 * r) return true;
        }
        return false;
    }
}
