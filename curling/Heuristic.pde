// Holds the outcome of one simulated shot.
class ShotResult {
    ArrayList<Stone> stonesAfter;
    ArrayList<Stone> stonesBefore;   // snapshot before the shot fired
    Stone            fired;
    ScoreResult      scoreBefore;
    Shot             plannedShot;
    int              remainingShotsAfterCurrent;

    ShotResult(ArrayList<Stone> stonesAfter, ArrayList<Stone> stonesBefore,
               Stone fired, ScoreResult scoreBefore, Shot plannedShot,
               int remainingShotsAfterCurrent) {
        this.stonesAfter         = stonesAfter;
        this.stonesBefore        = stonesBefore;
        this.fired               = fired;
        this.scoreBefore         = scoreBefore;
        this.plannedShot         = plannedShot;
        this.remainingShotsAfterCurrent = remainingShotsAfterCurrent;
    }
}

abstract class Heuristic {
    float weight = 1.0;
    float scale  = 1.0;

    float contribute(float raw) {
        return weight * (raw / scale);
    }

    abstract float scoreResult(ShotResult result);
}

// ---- Draw heuristic: score improvement only ----
// Used by: Draw, DrawCurlLeft, DrawCurlRight
class DrawHeuristic extends Heuristic {
    DrawHeuristic() {
        this.scale  = 3.0;
        this.weight = 2.5;
    }

    @Override
    float scoreResult(ShotResult result) {
        ScoreResult after = house.scoreEnd(result.stonesAfter);
        return after.diffFrom(result.scoreBefore);
    }
}

// ---- Curl heuristic: score + close-to-button bonus ----
// Used by: CurlLeft, CurlRight
class CurlHeuristic extends Heuristic {
    CurlHeuristic() {
        this.scale  = 3.0;
        this.weight = 2.0;
    }

    @Override
    float scoreResult(ShotResult result) {
        ScoreResult after = house.scoreEnd(result.stonesAfter);
        float scoreDelta = after.diffFrom(result.scoreBefore);

        // Close-to-button bonus for the fired stone
        Stone fired = result.fired;
        float d = house.distanceToButton(fired);
        float pinBonus = 0;
        if (fired.hogPassed && house.inHouse(fired)) {
            pinBonus = (house.OUTER_RING - d) / house.OUTER_RING;  // 0..1
            if (d < house.BUTTON_R) pinBonus += 0.5;
        }

        return scoreDelta * (this.scale / 3.0) + pinBonus * 0.4;
    }
}

// ---- Takeout heuristic: score + displacement/removal bonus ----
// Used by: Takeout
class TakeoutHeuristic extends Heuristic {
    TakeoutHeuristic() {
        this.scale  = 4.0;
        this.weight = 2.5;
    }

    @Override
    float scoreResult(ShotResult result) {
        ScoreResult after = house.scoreEnd(result.stonesAfter);
        float scoreDelta = after.diffFrom(result.scoreBefore);

        // Bonus: for each red stone that moved further from button or left play
        float displacementBonus = 0;
        for (Stone before : result.stonesBefore) {
            if (before.team != TEAM_RED) continue;
            float distBefore = house.distanceToButton(before);
            boolean wasHouseStone = before.hogPassed && house.inHouse(before);
            boolean wasGuard = before.hogPassed
                && !house.inHouse(before)
                && before.pos.y < house.BUTTON.y
                && before.pos.y > sheet.hogY;

            // Find the closest red stone in stonesAfter matching approximate pre-position
            Stone matchAfter = null;
            float minMove = Float.MAX_VALUE;
            for (Stone after2 : result.stonesAfter) {
                if (after2.team != TEAM_RED) continue;
                float moved = PVector.dist(before.pos, after2.pos);
                if (moved < minMove) {
                    minMove = moved;
                    matchAfter = after2;
                }
            }

            boolean removed = matchAfter == null || minMove > before.radius * 4.0f;
            if (removed) {
                // Red stone removed entirely (swept off ice)
                displacementBonus += 3.0;
                if (wasHouseStone) displacementBonus += 2.0;
                if (wasGuard) displacementBonus += 1.5;
            } else if (matchAfter != null) {
                Stone movedStone = matchAfter;
                float distAfter = house.distanceToButton(movedStone);
                // Bonus for moving it further from center
                float pushed = max(0, distAfter - distBefore);
                displacementBonus += pushed * 0.8;
                if (minMove > 0.75f) displacementBonus += min(1.5f, minMove * 0.4f);
                // Extra bonus for removing it from the house
                if (wasHouseStone && !house.inHouse(movedStone)) {
                    displacementBonus += 2.0;
                }
                // Bonus for clearing a guard
                if (wasGuard && !house.inHouse(movedStone) && movedStone.pos.y >= house.BUTTON.y) {
                    displacementBonus += 1.5;
                }
            }
        }

        return scoreDelta + displacementBonus * 0.7;
    }
}

// ---- Guard heuristic: coverage of button/own stones, no score ----
// Used by: Guard
// Reward fired stone for adding new protective lateral coverage in front
// of the house that existing yellow guards do not already provide.
class GuardHeuristic extends Heuristic {
    GuardHeuristic() {
        this.scale  = 2.0;
        this.weight = 2.0;
    }

    @Override
    float scoreResult(ShotResult result) {
        Stone fired = result.fired;

        // Guard must be in front of the house (between hog and tee lines)
        // and not inside the house itself.
        float teeY = house.BUTTON.y;
        float hogY = sheet.hogY;
        if (!fired.hogPassed) return -1.0;
        if (house.inHouse(fired)) return -1.4; // guards should not become draws
        if (fired.pos.y > teeY) return -0.8;   // past tee line, too deep

        // How close to the center line (ideal guard position)?
        float lateralOffset = abs(fired.pos.x - sheet.centerX);
        float lateralScore = max(0, 1.0 - lateralOffset / (sheet.SHEET_WIDTH_FT * 0.35));

        // Guard depth target: just beyond the hog line, not near/in the house.
        float guardTargetY = hogY + 2.0f;
        float depthScore = max(0, 1.0 - abs(fired.pos.y - guardTargetY) / 5.0);
        float deepPenalty = max(0, (fired.pos.y - (hogY + 8.0f)) / 10.0);

        // Coverage: how much does this stone block access to the button/own stones?
        // Model as: reward fraction of horizontal "shadow" cast toward the button
        // that is not already covered by existing yellow guards.
        float newCoverage = guardCoverageAdded(fired, result.stonesBefore);
        float protectsStone = protectsImportantStone(fired, result.stonesBefore);
        float speedPenalty = max(0, result.plannedShot.speed - 19f) / 3.0f;

        return lateralScore * 0.25
             + depthScore * 0.75
             + newCoverage * 1.0
             + protectsStone * 0.6
             - speedPenalty * 0.45
             - deepPenalty * 0.8;
    }

    // Approximate the new coverage added by the fired stone.
    // Coverage = lateral width blocked at stone's y-position not already covered
    // by other yellow non-house stones between hog and tee.
    private float guardCoverageAdded(Stone fired, ArrayList<Stone> before) {
        float stoneDiam = fired.radius * 2;
        float firedLeft  = fired.pos.x - fired.radius;
        float firedRight = fired.pos.x + fired.radius;

        // Compute existing coverage from yellow guard stones (not in house)
        float existingCoverage = 0;
        for (Stone s : before) {
            if (s.team != TEAM_YELLOW) continue;
            if (house.inHouse(s)) continue;
            if (!s.hogPassed) continue;
            float teeY = house.BUTTON.y;
            if (s.pos.y > teeY) continue;
            // Lateral overlap with fired stone's zone
            float sLeft  = s.pos.x - s.radius;
            float sRight = s.pos.x + s.radius;
            float overlap = max(0, min(firedRight, sRight) - max(firedLeft, sLeft));
            existingCoverage = max(existingCoverage, overlap);
        }
        float addedCoverage = max(0, stoneDiam - existingCoverage);
        return addedCoverage / stoneDiam; // 0..1
    }

    private float protectsImportantStone(Stone fired, ArrayList<Stone> before) {
        Stone target = null;
        float bestDist = Float.MAX_VALUE;
        for (Stone s : before) {
            if (s.team != TEAM_YELLOW) continue;
            if (!s.hogPassed || !house.inHouse(s)) continue;
            float d = house.distanceToButton(s);
            if (d < bestDist) { bestDist = d; target = s; }
        }
        PVector targetPos = target != null ? target.pos : house.BUTTON;
        if (fired.pos.y >= targetPos.y) return 0;
        float dx = abs(fired.pos.x - targetPos.x);
        float laneScore = max(0, 1.0 - dx / (fired.radius * 3.0));
        float dy = targetPos.y - fired.pos.y;
        float distanceScore = max(0, 1.0 - abs(dy - (house.OUTER_RING + 1.0f)) / 6.0);
        return laneScore * distanceScore;
    }
}

// ---- Freeze heuristic: proximity to target stone + small score bonus ----
// Used by: Freeze
class FreezeHeuristic extends Heuristic {
    FreezeHeuristic() {
        this.scale  = 3.0;
        this.weight = 2.5;
    }

    @Override
    float scoreResult(ShotResult result) {
        Stone fired = result.fired;
        if (!fired.hogPassed) return -1.0;

        // Find the target: red stone closest to button inside house
        Stone target = null;
        float bestDist = Float.MAX_VALUE;
        for (Stone s : result.stonesBefore) {
            if (s.team != TEAM_RED) continue;
            if (!house.inHouse(s)) continue;
            float d = house.distanceToButton(s);
            if (d < bestDist) { bestDist = d; target = s; }
        }
        // If no red stone in house, try the yellow stone closest to button
        // (freeze onto own stone to protect it)
        if (target == null) {
            for (Stone s : result.stonesBefore) {
                if (s.team != TEAM_YELLOW) continue;
                if (!house.inHouse(s)) continue;
                float d = house.distanceToButton(s);
                if (d < bestDist) { bestDist = d; target = s; }
            }
        }
        if (target == null) {
            // No in-house stone at all — just reward being close to button
            float d = house.distanceToButton(fired);
            return max(0, (house.OUTER_RING - d) / house.OUTER_RING);
        }

        // Gap between the fired stone's edge and the target's edge (ideally 0)
        float gap = PVector.dist(fired.pos, target.pos) - fired.radius - target.radius;
        float gapNorm = max(0, gap); // ft
        float proximityScore = max(0, 1.0 - gapNorm / (house.OUTER_RING * 0.5));

        // Small score bonus
        ScoreResult after = house.scoreEnd(result.stonesAfter);
        float scoreDelta = after.diffFrom(result.scoreBefore);
        float scoreBonus = constrain(scoreDelta / 3.0, -0.3, 0.3);

        return proximityScore * 1.5 + scoreBonus;
    }
}

// ---- FinalScore heuristic: long-term score after this shot ----
class FinalScoreHeuristic extends Heuristic {
    FinalScoreHeuristic() {
        this.scale  = 3.0;
        this.weight = 4.0; // most important heuristic
    }

    @Override
    float scoreResult(ShotResult result) {
        if (gradientEnsemble == null) {
            ScoreResult after = house.scoreEnd(result.stonesAfter);
            return after.diffFrom(result.scoreBefore);
        }
        float longTerm = gradientEnsemble.longTermUtilityFromBoard(
            result.stonesAfter,
            result.remainingShotsAfterCurrent,
            TEAM_RED,
            shotSelector
        );
        return longTerm - result.scoreBefore.yellowFitness();
    }
}
