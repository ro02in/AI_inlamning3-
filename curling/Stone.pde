// Oliwer Carpman, Rafal Galinski, Robin Karim
// =============================================================
// Stone - one curling stone (data + rendering only).
// =============================================================
// Physics is intentionally NOT implemented here. Physics.pde
// owns all motion / collision logic; Stone is just a value
// object plus a draw() helper.
// =============================================================

static final int TEAM_RED    = 0;
static final int TEAM_YELLOW = 1;

// Physics radius in world units (feet). ~5.7 in granite radius.
static final float STONE_RADIUS = 0.475;

// Visual rotation rate (radians / s) at |curl| = 1, while the stone
// is moving. Sign matches curl, so positive curl spins clockwise on
// screen (matching its rightward-drift direction).
static final float STONE_SPIN_RATE = 5.0;

class Stone {
  PVector pos;        // world coords
  PVector vel;        // world units / second
  int     team;       // TEAM_RED or TEAM_YELLOW
  float   curl;       // -1..+1, set at release, used by Physics.curlAccel
  float   radius;     // world units
  float   spinAngle;  // radians, advanced by Physics while moving + curling
  boolean hogPassed;  // leading edge cleared hog line toward house

  Stone(float wx, float wy, int team) {
    this.pos       = new PVector(wx, wy);
    this.vel       = new PVector(0, 0);
    this.team      = team;
    this.curl      = 0;
    this.radius    = STONE_RADIUS;
    this.spinAngle = 0;
    this.hogPassed = false;
  }

  boolean isMoving() {
    return vel.magSq() > 0.0001;
  }

  PImage sprite() {
    return team == TEAM_RED ? stoneRedImg : stoneYellowImg;
  }

  void draw() {
    PVector s = worldToScreen(pos);
    float   d = worldToScreen(radius * 2.0);
    pushMatrix();
    translate(s.x, s.y);
    rotate(spinAngle);
    pushStyle();
    imageMode(CENTER);
    image(sprite(), 0, 0, d, d);
    popStyle();
    popMatrix();
  }
}
