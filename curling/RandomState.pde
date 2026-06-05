// Oliwer Carpman, Rafal Galinski, Robin Karpman
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

    void randomizeForExpert(ArrayList<Stone> stones, int stonesToPlace, String expertName) {
        randomizeForDepth(stones, stonesToPlace);
        if (expertName == null) return;

        if (expertName.equals("Guard")) {
            ensureYellowNearHouse(stones);
        } else if (expertName.equals("Freeze")) {
            ensureStoneInHouse(stones, random(1) < 0.7f ? TEAM_RED : TEAM_YELLOW);
        } else if (expertName.equals("Takeout")) {
            if (random(1) < 0.75f) ensureStoneInHouse(stones, TEAM_RED);
            else ensureRedGuard(stones);
        }
    }

    private void ensureYellowNearHouse(ArrayList<Stone> stones) {
        for (Stone s : stones) {
            if (s.team == TEAM_YELLOW && s.hogPassed && house.inHouse(s)) return;
        }
        placeInHouse(stones, TEAM_YELLOW);
    }

    private void ensureStoneInHouse(ArrayList<Stone> stones, int team) {
        for (Stone s : stones) {
            if (s.team == team && s.hogPassed && house.inHouse(s)) return;
        }
        placeInHouse(stones, team);
    }

    private void ensureRedGuard(ArrayList<Stone> stones) {
        for (Stone s : stones) {
            if (s.team == TEAM_RED && s.hogPassed && !house.inHouse(s)
                && s.pos.y < house.BUTTON.y && s.pos.y > sheet.hogY) {
                return;
            }
        }
        placeInBox(stones, TEAM_RED,
                   sheet.centerX - 2.0f, sheet.centerX + 2.0f,
                   house.BUTTON.y - house.OUTER_RING - 4.0f,
                   house.BUTTON.y - house.OUTER_RING - 0.5f);
    }

    private void placeInHouse(ArrayList<Stone> stones, int team) {
        for (int tries = 0; tries < maxTries; tries++) {
            float angle = random(TWO_PI);
            float dist = sqrt(random(1)) * (house.OUTER_RING - r);
            PVector p = new PVector(house.BUTTON.x + cos(angle) * dist,
                                    house.BUTTON.y + sin(angle) * dist);
            if (!overlaps(p, stones)) {
                addPlacedStone(stones, p, team);
                return;
            }
        }
    }

    private void placeInBox(ArrayList<Stone> stones, int team,
                            float minX, float maxX, float minY, float maxY) {
        for (int tries = 0; tries < maxTries; tries++) {
            PVector p = new PVector(random(max(r, minX), min(sheetW - r, maxX)),
                                    random(max(hogY + r, minY), min(backY - r, maxY)));
            if (!overlaps(p, stones)) {
                addPlacedStone(stones, p, team);
                return;
            }
        }
    }

    private void addPlacedStone(ArrayList<Stone> stones, PVector p, int team) {
        Stone s = new Stone(p.x, p.y, team);
        s.hogPassed = true;
        stones.add(s);
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
