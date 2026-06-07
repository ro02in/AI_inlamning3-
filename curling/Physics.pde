// Oliwer Carpman, Rafal Galinski, Robin Karim
// =============================================================
// Physics - all motion / collision logic for stones.
// =============================================================
// Everything tunable about how stones behave lives here as a
// `final` field on the Physics class. Swap a constant or replace
// the body of curlAccel() / resolveCollision() to change the
// model without touching anything else.
// =============================================================

class Physics {

    // ----- Tunables (feet, seconds) -----------------------------
    // Friction: deceleration along velocity (ft / s^2).
    final float FRICTION       = 4.2;

    // Snap-to-rest threshold (ft / s).
    final float STOP_SPEED     = 0.14;

    // Curl peak lateral acceleration (ft / s^2) at |curl| = 1.
    final float CURL_STRENGTH   = 14.0;

    final float CURL_PEAK_SPEED = 0.65;
    final float CURL_SPEED_EXP  = 0.85;

    final float RESTITUTION    = 0.98;

    // -----------------------------------------------------------
    // step: advance the world one fixed timestep dt (seconds).
    // When cull==false (aim preview), skips hog bookkeeping and removal.
    // -----------------------------------------------------------
    void step(ArrayList<Stone> stones, float dt) {
        step(stones, dt, true);
    }

    void step(ArrayList<Stone> stones, float dt, boolean cull) {
        for (Stone s : stones) {
            if (!s.isMoving()) continue;
            PVector accel = new PVector(0, 0);
            accel.add(curlAccel(s));
            accel.add(frictionAccel(s));
            s.vel.add(PVector.mult(accel, dt));
            s.pos.add(PVector.mult(s.vel, dt));
            s.spinAngle += s.curl * STONE_SPIN_RATE * dt;
        }
        resolveCollisions(stones);
        snapToRest(stones);

        if (cull) {
            for (Stone s : stones) {
                if (leadingEdgeY(s) >= sheet.hogY) s.hogPassed = true;
            }
            removeOutOfBounds(stones);
        }
    }

    // -----------------------------------------------------------
    // frictionAccel: constant deceleration opposing motion.
    // -----------------------------------------------------------
    PVector frictionAccel(Stone s) {
        float speed = s.vel.mag();
        if (speed < 1e-4) return new PVector(0, 0);
        return PVector.mult(s.vel, -FRICTION / speed);
    }

    // -----------------------------------------------------------
    // curlAccel: lateral acceleration perpendicular to velocity.
    // Replace this body to try a different curl model.
    //
    // Current model has two regimes around CURL_PEAK_SPEED:
    //   speed > peak : |a| = K * (peak/speed)^EXP  (back-loaded curl)
    //   speed < peak : |a| = K * (speed/peak)      (linear taper to 0)
    // The taper is what guarantees the stone eventually stops; without it,
    // strong curl produces a self-sustaining spiral at low speed.
    //
    // Direction: lateral accel perpendicular to velocity; positive
    // curl bends to +x when the stone travels toward +y (house at top).
    // -----------------------------------------------------------
    PVector curlAccel(Stone s) {
        float speed = s.vel.mag();
        if (speed < 1e-4 || abs(s.curl) < 1e-4) return new PVector(0, 0);

        float scale;
        if (speed >= CURL_PEAK_SPEED) {
            scale = pow(CURL_PEAK_SPEED / speed, CURL_SPEED_EXP);
        } else {
            scale = speed / CURL_PEAK_SPEED;
        }
        float mag = s.curl * CURL_STRENGTH * scale;

        // For v = (0, +V): "right" in screen x is (+1,0) = (vy, -vx)/|v|.
        PVector perp = new PVector(s.vel.y, -s.vel.x).normalize();
        return PVector.mult(perp, mag);
    }

    // -----------------------------------------------------------
    // resolveCollisions: O(n^2) circle-circle, equal mass.
    // Two passes per call:
    //   1. positional correction so overlapping pairs are pushed apart
    //   2. impulse along the contact normal scaled by RESTITUTION
    // -----------------------------------------------------------
    void resolveCollisions(ArrayList<Stone> stones) {
        int n = stones.size();
        for (int i = 0; i < n; i++) {
            Stone a = stones.get(i);
            for (int j = i + 1; j < n; j++) {
                Stone b = stones.get(j);
                resolvePair(a, b);
            }
        }
    }

    void resolvePair(Stone a, Stone b) {
        PVector delta = PVector.sub(b.pos, a.pos);
        float   dist  = delta.mag();
        float   minD  = a.radius + b.radius;
        if (dist >= minD || dist < 1e-4) return;

        PVector n = PVector.div(delta, dist);

        // Positional correction: split overlap evenly between the two stones.
        float overlap = (minD - dist) * 0.5;
        a.pos.sub(PVector.mult(n, overlap));
        b.pos.add(PVector.mult(n, overlap));

        // Velocity along the contact normal (equal mass).
        PVector relVel = PVector.sub(b.vel, a.vel);
        float   vn     = PVector.dot(relVel, n);
        if (vn > 0) return;  // already separating

        float   jImp   = -(1 + RESTITUTION) * vn * 0.5;
        PVector impulse = PVector.mult(n, jImp);
        a.vel.sub(impulse);
        b.vel.add(impulse);
    }

    // -----------------------------------------------------------
    // snapToRest: zero out velocity when below STOP_SPEED.
    // Also clears curl so that a struck stone (which inherited its
    // thrower's curl value) doesn't keep curling after being knocked.
    // Only the most recently fired stone has a non-zero curl.
    // -----------------------------------------------------------
    void snapToRest(ArrayList<Stone> stones) {
        for (Stone s : stones) {
            if (s.vel.mag() < STOP_SPEED) {
                s.vel.set(0, 0);
                s.curl = 0;
                // Sync hog flag to final resting position (stone may have rolled back).
                s.hogPassed = leadingEdgeY(s) >= sheet.hogY;
            }
        }
    }

    // Leading edge toward the house (+y direction).
    float leadingEdgeY(Stone s) {
        return s.pos.y + s.radius;
    }

    // -----------------------------------------------------------
    // removeOutOfBounds: past far back line, off-sheet laterally,
    // well past the hack end, or came to rest without clearing the hog line.
    // -----------------------------------------------------------
    void removeOutOfBounds(ArrayList<Stone> stones) {
        for (int i = stones.size() - 1; i >= 0; i--) {
            Stone s = stones.get(i);
            // Hog violation: at rest with leading edge short of the hog line.
            if (!s.isMoving() && leadingEdgeY(s) < sheet.hogY) {
                stones.remove(i);
                continue;
            }
            // Entire stone past far back line, continuing toward hog / center.
            if (s.pos.y - s.radius > sheet.backFarY + 0.5) {
                stones.remove(i);
                continue;
            }
            // Flew off the top of the modeled sheet past hog.
            if (s.pos.y - s.radius > sheet.yMax + 2.0) {
                stones.remove(i);
                continue;
            }
            if (s.pos.x + s.radius < 0 || s.pos.x - s.radius > sheet.SHEET_WIDTH_FT) {
                stones.remove(i);
                continue;
            }
            // Past hack end (wrong direction / off back).
            if (s.pos.y + s.radius < sheet.yMin - 1.0) {
                stones.remove(i);
            }
        }
    }

    // -----------------------------------------------------------
    // predictPath: forward-simulate a single stone for the given
    // shot, ignoring other stones / collisions. Used for aim preview.
    // Returns world-space points sampled every step.
    // -----------------------------------------------------------
    ArrayList<PVector> predictPath(PVector start, Shot shot, int maxSteps, float dt) {
        Stone tmp = new Stone(start.x, start.y, TEAM_RED);
        tmp.curl = shot.curl;
        tmp.vel.set(sin(shot.angle) * shot.speed, cos(shot.angle) * shot.speed);

        ArrayList<Stone>   single = new ArrayList<Stone>();
        single.add(tmp);
        ArrayList<PVector> path   = new ArrayList<PVector>();
        path.add(tmp.pos.copy());

        for (int i = 0; i < maxSteps; i++) {
            step(single, dt, false);
            path.add(tmp.pos.copy());
            if (!tmp.isMoving()) break;
        }
        return path;
    }
}
