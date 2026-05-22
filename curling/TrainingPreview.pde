// Shows the latest training-policy shot on the ice while evolution runs at full speed.
// Each finished animation waits briefly, then jumps to whatever evolution is current.
class TrainingPreviewSnapshot {
  int              evolution;
  ArrayList<Stone> layout;
  Shot             shot;

  TrainingPreviewSnapshot(int evolution, ArrayList<Stone> layout, Shot shot) {
    this.evolution = evolution;
    this.layout    = layout;
    this.shot       = shot;
  }
}

class TrainingPreview {
  static final float PAUSE_SEC = 0.5f;

  TrainingPreviewSnapshot latest;
  ArrayList<Stone> displayStones;
  Shot             displayShot;
  int              displayedEvolution = -1;
  boolean          simulating = false;
  float            waitUntilMs = 0;

  RandomState randomState = new RandomState();

  void reset() {
    latest             = null;
    displayStones      = null;
    displayShot        = null;
    displayedEvolution = -1;
    simulating         = false;
    waitUntilMs        = 0;
  }

  ArrayList<Stone> copyLayout(ArrayList<Stone> layout) {
    ArrayList<Stone> copy = new ArrayList<Stone>();
    for (Stone s : layout) {
      Stone c = new Stone(s.pos.x, s.pos.y, s.team);
      c.hogPassed = s.hogPassed;
      copy.add(c);
    }
    return copy;
  }

  // Called once per evolution; overwrites the pending snapshot.
  void recordSnapshot(int evolution, NeuralPolicy policy) {
    ArrayList<Stone> layout = new ArrayList<Stone>();
    randomState.randomize(layout, STONES_PER_TEAM);
    float[] state = policy.convertState(layout, 1, TEAM_RED);
    Shot shot = policy.predict(state);
    latest = new TrainingPreviewSnapshot(evolution, copyLayout(layout), shot);
  }

  void update(float dt) {
    if (simulating && displayStones != null) {
      physics.step(displayStones, dt);
      boolean anyMoving = false;
      for (Stone s : displayStones) {
        if (s.isMoving()) { anyMoving = true; break; }
      }
      if (!anyMoving) {
        simulating  = false;
        waitUntilMs = millis() + PAUSE_SEC * 1000;
      }
      return;
    }

    if (millis() < waitUntilMs) return;

    if (latest != null && latest.evolution > displayedEvolution) {
      startDisplay(latest);
    }
  }

  void startDisplay(TrainingPreviewSnapshot snap) {
    displayedEvolution = snap.evolution;
    displayShot          = snap.shot;
    displayStones        = copyLayout(snap.layout);

    PVector h = sheet.hackWorld();
    Stone fired = new Stone(h.x, h.y, TEAM_YELLOW);
    fired.curl = constrain(snap.shot.curl, -1, 1);
    fired.vel.set(sin(snap.shot.angle) * snap.shot.speed,
                  cos(snap.shot.angle) * snap.shot.speed);
    displayStones.add(fired);
    simulating = true;
  }

  void drawStones() {
    if (displayStones == null) return;
    if (displayShot != null) {
      drawShotPreview(displayShot, color(235, 205, 60));
    }
    for (Stone s : displayStones) s.draw();
  }

  void drawOverlay(int trainingDone, int trainingTarget) {
    pushStyle();
    fill(0, 160);
    noStroke();
    rect(0, ICE_H - 52, ICE_W, 52);

    fill(240);
    textAlign(CENTER, CENTER);
    textSize(14);
    text("Tr\u00e4nar: " + trainingDone + " / " + trainingTarget, ICE_W * 0.5, ICE_H - 36);

    textSize(12);
    fill(200);
    if (displayedEvolution > 0) {
      String phase = simulating ? "visar jamf. " + displayedEvolution
                                : "paus → senaste jamf. " + (latest != null ? latest.evolution : displayedEvolution);
      text(phase, ICE_W * 0.5, ICE_H - 16);
    } else {
      text("startar visning...", ICE_W * 0.5, ICE_H - 16);
    }
    popStyle();
  }
}
