// =============================================================
// Physics - all motion / collision logic for stones.
// =============================================================
// Everything tunable about how stones behave lives here as a
// `final` field on the Physics class. Swap a constant or replace
// the body of curlAccel() / resolveCollision() to change the
// model without touching anything else.
// =============================================================

class Physics {

  // ----- Tunables -----
  // Friction (deceleration along velocity direction), world units / s^2.
  // Tuned so a fast shot (~1100 wu/s) just clears the back of the house
  // from the hack, and a medium shot (~900 wu/s) lands near the button.
  final float FRICTION       = 350.0;

  // Snap-to-rest threshold, world units / s.
  final float STOP_SPEED     = 8.0;

  // Curl: peak lateral acceleration (world units / s^2) at slider = 1.
  // Heads up: pushing this much above ~1400 with the current FRICTION
  // and CURL_PEAK_SPEED makes Euler integration error inject more
  // energy than friction can dissipate, and the stone orbits forever.
  final float CURL_STRENGTH   = 1200.0;

  // Speed (world units / s) at which curl is at its peak. Above this,
  // curl ramps up as the stone slows down toward this speed. Below it,
  // curl tapers linearly to 0 at speed=0 so the stone always comes to rest.
  final float CURL_PEAK_SPEED = 60.0;

  // Curl ramp above peak speed: how strongly curl grows as the stone
  // slows down toward CURL_PEAK_SPEED.
  // 0 = constant, 1 = linear in (1/speed), >1 = late-curl bias.
  final float CURL_SPEED_EXP  = 0.8;

  // Coefficient of restitution for stone-stone collisions (1 = elastic).
  final float RESTITUTION    = 0.98;

  // -----------------------------------------------------------
  // step: advance the world one fixed timestep dt (seconds).
  // Order: curl accel -> friction accel -> integrate -> collisions
  //        -> stop snap -> remove out-of-bounds stones.
  // -----------------------------------------------------------
  void step(ArrayList<Stone> stones, float dt) {
    for (Stone s : stones) {
      if (!s.isMoving()) continue;
      PVector accel = new PVector(0, 0);
      accel.add(curlAccel(s));
      accel.add(frictionAccel(s));
      s.vel.add(PVector.mult(accel, dt));
      s.pos.add(PVector.mult(s.vel, dt));
      // Visual spin: only while moving, signed by curl.
      s.spinAngle += s.curl * STONE_SPIN_RATE * dt;
    }
    resolveCollisions(stones);
    snapToRest(stones);
    removeOutOfBounds(stones);
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
  // Direction: rotated 90 deg from velocity so that positive curl
  // pushes the stone to its RIGHT in screen space (y-down). For a
  // stone traveling "up" the sheet (vy negative), positive curl
  // bends it toward +x.
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

    // Visual right of velocity in y-down screen coords:
    // for v=(0,-V) (up) we want perp=(+1,0). Formula: (-vy, vx).
    PVector perp = new PVector(-s.vel.y, s.vel.x).normalize();
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
      }
    }
  }

  // -----------------------------------------------------------
  // removeOutOfBounds: drop stones whose center has left the world
  // rectangle. Simple stand-in for real curling rails / back line.
  // -----------------------------------------------------------
  void removeOutOfBounds(ArrayList<Stone> stones) {
    for (int i = stones.size() - 1; i >= 0; i--) {
      Stone s = stones.get(i);
      if (s.pos.x < 0 || s.pos.x > WORLD_W
       || s.pos.y < 0 || s.pos.y > WORLD_H) {
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
    tmp.vel.set(sin(shot.angle) * shot.speed, -cos(shot.angle) * shot.speed);

    ArrayList<Stone>   single = new ArrayList<Stone>();
    single.add(tmp);
    ArrayList<PVector> path   = new ArrayList<PVector>();
    path.add(tmp.pos.copy());

    for (int i = 0; i < maxSteps; i++) {
      step(single, dt);
      path.add(tmp.pos.copy());
      if (!tmp.isMoving()) break;
    }
    return path;
  }
}
