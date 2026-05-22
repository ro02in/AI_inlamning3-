// =============================================================
// UI - sliders + Skjut button drawn in the right sidebar.
// =============================================================
// No external libraries. Each Slider is a vertical track with a
// draggable handle; the Button is a clickable rectangle.
// =============================================================

class Slider {
  String  label;
  float   minV, maxV, value;
  // Track centerline: x = cx, vertical span [y, y+h]
  float   cx, y, h;
  // Hit area half-width on either side of the track.
  float   hitHalfW;
  // Decimals in the readout.
  int     decimals;
  boolean dragging;

  Slider(String label, float minV, float maxV, float value,
        float cx, float y, float h, int decimals) {
    this.label    = label;
    this.minV     = minV;
    this.maxV     = maxV;
    this.value    = value;
    this.cx       = cx;
    this.y        = y;
    this.h        = h;
    this.hitHalfW = 26;
    this.decimals = decimals;
    this.dragging = false;
  }

  float fraction() {
    return (value - minV) / (maxV - minV);
  }

  void setFromMouseY(float my) {
    float f = constrain(1 - (my - y) / h, 0, 1);
    value = minV + f * (maxV - minV);
  }

  boolean trackHit(float mx, float my) {
    return mx >= cx - hitHalfW && mx <= cx + hitHalfW
        && my >= y - 4         && my <= y + h + 4;
  }

  void draw() {
    pushStyle();
    rectMode(CORNER);

    // label centered above the track
    noStroke();
    fill(220);
    textAlign(CENTER, BOTTOM);
    textSize(12);
    text(label, cx, y - 8);

    // track
    stroke(80);
    strokeWeight(2);
    line(cx, y, cx, y + h);

    // ticks at min/max
    stroke(60);
    strokeWeight(1);
    line(cx - 8, y,     cx + 8, y);     // top tick (max)
    line(cx - 8, y + h, cx + 8, y + h); // bottom tick (min)
    if (minV < 0 && maxV > 0) {         // zero tick
      float zy = y + h * (maxV / (maxV - minV));
      line(cx - 8, zy, cx + 8, zy);
    }

    // handle
    float hy = y + h * (1 - fraction());
    noStroke();
    fill(dragging ? color(255) : color(200));
    rect(cx - 14, hy - 5, 28, 10, 3);

    // readout centered below the track
    fill(230);
    textAlign(CENTER, TOP);
    textSize(12);
    text(nf(value, 0, decimals), cx, y + h + 8);

    popStyle();
  }
}

// =============================================================
// TimingBar - horizontal accuracy bar with green center / red
// edges and an oscillating indicator. Click to lock the indicator
// at its current position. signedError() returns the perturbation
// scale (-1..+1) the lock should inject into the corresponding
// shot parameter; 0 if the lock landed in the green band.
// =============================================================
class TimingBar {
  String  label;
  float   x, y, w, h;     // bar bounds (excluding label)

  // Oscillation state. oscPhase advances 0..1 over time and we
  // map it to a triangle wave for the indicator position.
  float   oscPhase;
  float   oscHz;          // configured per frame from curl
  float   greenFrac;      // configured per frame from curl

  boolean locked;
  float   lockedPhase;

  TimingBar(String label, float x, float y, float w, float h) {
    this.label = label;
    this.x = x; this.y = y; this.w = w; this.h = h;
    reset();
  }

  void configure(float hz, float greenFraction) {
    this.oscHz     = hz;
    this.greenFrac = greenFraction;
  }

  void reset() {
    oscPhase    = 0;
    locked      = false;
    lockedPhase = 0;
  }

  void update(float dt) {
    if (locked) return;
    oscPhase += oscHz * dt;
    while (oscPhase >= 1) oscPhase -= 1;
  }

  void lockNow() {
    if (locked) return;
    lockedPhase = oscPhase;
    locked      = true;
  }

  void toggleLock() {
    if (locked) locked = false;
    else        lockNow();
  }

  // Triangle wave 0 -> 1 -> 0 over phase 0 -> 1.
  float indicatorFraction() {
    float p = locked ? lockedPhase : oscPhase;
    return p < 0.5 ? p * 2.0 : 2.0 - p * 2.0;
  }

  // Signed error in [-1, +1]. 0 inside the green band.
  // Negative = locked left of center, positive = right of center.
  float signedError() {
    float pos       = indicatorFraction();
    float deviation = pos - 0.5;
    float magnitude = abs(deviation) * 2.0;       // 0..1
    if (magnitude <= greenFrac) return 0;
    float em        = (magnitude - greenFrac) / (1.0 - greenFrac);
    return deviation > 0 ? em : -em;
  }

  boolean hits(float mx, float my) {
    return mx >= x && mx <= x + w
        && my >= y - 6 && my <= y + h + 6;
  }

  void draw() {
    pushStyle();
    rectMode(CORNER);

    // label above
    fill(220);
    noStroke();
    textAlign(LEFT, BOTTOM);
    textSize(11);
    text(label, x, y - 4);
    if (locked) {
      textAlign(RIGHT, BOTTOM);
      fill(235);
      text("L\u00c5ST", x + w, y - 4);
    }

    // red background (full bar)
    noStroke();
    fill(150, 40, 40);
    rect(x, y, w, h, 4);

    // green center band
    float gw = w * greenFrac;
    float gx = x + (w - gw) * 0.5;
    fill(50, 170, 70);
    rect(gx, y, gw, h, 4);

    // indicator
    float ix = x + w * indicatorFraction();
    if (locked) {
      stroke(255);
      strokeWeight(3);
      line(ix, y - 4, ix, y + h + 4);
      noStroke();
      fill(255);
      ellipse(ix, y - 6, 5, 5);
    } else {
      stroke(255, 230);
      strokeWeight(2);
      line(ix, y - 2, ix, y + h + 2);
    }
    popStyle();
  }
}

class Button {
  String  label;
  float   x, y, w, h;
  boolean enabled;
  boolean hover;

  Button(String label, float x, float y, float w, float h) {
    this.label = label;
    this.x = x; this.y = y; this.w = w; this.h = h;
    this.enabled = true;
    this.hover   = false;
  }

  boolean hits(float mx, float my) {
    return mx >= x && mx <= x + w && my >= y && my <= y + h;
  }

  void draw() {
    pushStyle();
    rectMode(CORNER);
    noStroke();
    if (!enabled)     fill(60);
    else if (hover)   fill(70, 140, 200);
    else              fill(50, 110, 170);
    rect(x, y, w, h, 6);

    fill(enabled ? 255 : 140);
    textAlign(CENTER, CENTER);
    textSize(16);
    text(label, x + w * 0.5, y + h * 0.5);
    popStyle();
  }
}

class UI {
  Slider     curlSlider;
  Slider     speedSlider;
  Slider     angleSlider;
  TimingBar  angleBar;
  TimingBar  speedBar;
  Button     shootBtn;
  Slider     activeSlider;

  // Lock workflow phase. 0 = locking angle, 1 = locking speed.
  // Each click of the action button locks the active bar and either
  // advances to the next phase or fires the shot.
  final int PHASE_ANGLE = 0;
  final int PHASE_SPEED = 1;
  int lockPhase;

  // ----- sidebar layout (vertical regions) -------------------
  // y =   0 .. 130 : stats (header, turn, status, stones-left)
  // y = 160 .. ~740 : sliders (label above track, track, value below)
  // y = ~750 .. ~772 : single active timing bar
  // y = ~798 .. ~858 : action button (Lås vinkel / Lås fart / Ny match)
  // -----------------------------------------------------------
  final float STATS_TOP    =  16;
  final float STATS_BOTTOM = 130;
  final float SLIDER_TOP   = 160;
  final float SLIDER_H     = 560;
  final float BAR_Y        = 758;
  final float BAR_H        =  22;
  final float BTN_TOP      = 798;

  // ----- Lock-bar tunables (adjust to taste) -----------------
  // Oscillation rate (cycles per second) at curl=0 vs |curl|=1.
  final float LOCK_BASE_HZ            = 0.7;
  final float LOCK_CURL_HZ_MULT       = 1.1;
  // Width of the green "safe" band as a fraction of the bar
  // (0.4 = 40% of the bar is green) at curl=0 vs |curl|=1.
  final float LOCK_GREEN_FRAC_BASE    = 0.45;
  final float LOCK_GREEN_FRAC_MIN     = 0.18;
  // Maximum perturbation when the lock lands at the very edge.
  final float LOCK_ANGLE_ERR_MAX_DEG  = 4.0;
  final float LOCK_SPEED_ERR_MAX_FRAC = 0.12;

  UI() {
    float panelLeft  = ICE_W;
    float colSpacing = SIDEBAR_W / 3.0;
    float c1 = panelLeft + colSpacing * 0.5;
    float c2 = panelLeft + colSpacing * 1.5;
    float c3 = panelLeft + colSpacing * 2.5;

    curlSlider  = new Slider("Curl",         -1, 1,    0,     c1, SLIDER_TOP, SLIDER_H, 2);
    speedSlider = new Slider("Fart",          0, 1,    0.78f, c2, SLIDER_TOP, SLIDER_H, 3);
    angleSlider = new Slider("Vinkel\u00b0", -15, 15,  0,     c3, SLIDER_TOP, SLIDER_H, 1);

    float barW = SIDEBAR_W - 40;
    float barX = ICE_W + (SIDEBAR_W - barW) * 0.5;
    // Both bars live at the same y: only the active one is rendered,
    // so the user sees a single bar that swaps modes between phases.
    angleBar = new TimingBar("L\u00e5s vinkel", barX, BAR_Y, barW, BAR_H);
    speedBar = new TimingBar("L\u00e5s fart",   barX, BAR_Y, barW, BAR_H);

    float btnW = SIDEBAR_W - 40;
    shootBtn = new Button("L\u00e5s vinkel",
                          ICE_W + (SIDEBAR_W - btnW) * 0.5,
                          BTN_TOP,
                          btnW, 60);

    lockPhase = PHASE_ANGLE;
  }

  TimingBar activeBar() {
    return (lockPhase == PHASE_ANGLE) ? angleBar : speedBar;
  }

  // -----------------------------------------------------------
  // update: advance the active bar's oscillator. Difficulty (rate
  // + green fraction) is recomputed every frame from the curl slider.
  // The inactive bar is frozen (already locked or not yet started).
  // -----------------------------------------------------------
  void update(float dt) {
    float absCurl = abs(curlSlider.value);
    float hz      = LOCK_BASE_HZ + absCurl * LOCK_CURL_HZ_MULT;
    float greenF  = lerp(LOCK_GREEN_FRAC_BASE, LOCK_GREEN_FRAC_MIN, absCurl);
    TimingBar bar = activeBar();
    bar.configure(hz, greenF);
    bar.update(dt);
  }

  // Called when a new shot becomes possible (start of a turn / reset).
  void onTurnStart() {
    angleBar.reset();
    speedBar.reset();
    lockPhase = PHASE_ANGLE;
  }

  // -----------------------------------------------------------
  // Build a Shot from the current slider values, perturbed by the
  // lock bars. A bar in the green band injects 0 error; in the red,
  // error grows linearly out to LOCK_*_ERR_MAX_* at the bar edges.
  // Speed slider is normalized 0..1 -> world units / s via SPEED_MAX.
  // -----------------------------------------------------------
  final static float SPEED_MAX = 40.0;   // ft/s at slider value 1.0

  // The "perfect" shot - slider values only, no lock-bar perturbation.
  // Used for the aim preview so the curve stays steady regardless of
  // where the bars currently are.
  Shot intendedShot() {
    return new Shot(curlSlider.value,
                    speedSlider.value * SPEED_MAX,
                    radians(angleSlider.value));
  }

  // The actual shot used when firing - slider values perturbed by
  // wherever the lock bars happened to be locked.
  Shot currentShot() {
    float angleDeg = angleSlider.value
                   + angleBar.signedError() * LOCK_ANGLE_ERR_MAX_DEG;
    float speedRaw = speedSlider.value
                   * (1.0 + speedBar.signedError() * LOCK_SPEED_ERR_MAX_FRAC);
    speedRaw = max(0, speedRaw);
    return new Shot(curlSlider.value,
                    speedRaw * SPEED_MAX,
                    radians(angleDeg));
  }

  void draw() {
    drawStatsPanel();

    curlSlider.draw();
    speedSlider.draw();
    angleSlider.draw();

    // Only the active bar is visible; the user locks it via the
    // action button, which then swaps the bar to the next phase.
    activeBar().draw();

    // Action button label depends on game state and lock phase.
    if (game.state == GameState.ENDED) {
      shootBtn.label   = "Ny match";
      shootBtn.enabled = true;
    } else if (game.state == GameState.AIMING) {
      shootBtn.label   = (lockPhase == PHASE_ANGLE) ? "L\u00e5s vinkel" : "L\u00e5s fart";
      shootBtn.enabled = true;
    } else {
      shootBtn.label   = "...";
      shootBtn.enabled = false;
    }
    shootBtn.draw();
  }

  // ----- stats panel: header + turn + status + stones-left ----
  void drawStatsPanel() {
    float left  = ICE_W + 16;
    float right = ICE_W + SIDEBAR_W - 16;

    pushStyle();
    textAlign(LEFT, TOP);
    fill(235);
    textSize(15);
    text("Skott", left, STATS_TOP);

    fill(170);
    textSize(11);
    text("Tur",       left, STATS_TOP + 26);
    text("Status",    left, STATS_TOP + 44);

    textAlign(RIGHT, TOP);
    fill(game.currentTeam == TEAM_RED ? color(230, 80, 80) : color(230, 210, 80));
    text(game.teamLabel(game.currentTeam), right, STATS_TOP + 26);
    fill(220);
    text(game.stateLabel(),                right, STATS_TOP + 44);

    // Two rows of stone-count dots.
    drawTeamRow(left, STATS_TOP + 70, TEAM_RED);
    drawTeamRow(left, STATS_TOP + 90, TEAM_YELLOW);

    // Separator line above the slider area.
    stroke(60);
    strokeWeight(1);
    line(ICE_W + 12, STATS_BOTTOM, ICE_W + SIDEBAR_W - 12, STATS_BOTTOM);
    popStyle();
  }

  // One team's row: label + 4 dots (filled = remaining, hollow = thrown).
  void drawTeamRow(float xLeft, float y, int team) {
    color teamColor = team == TEAM_RED ? color(230, 80, 80) : color(230, 210, 80);
    int   remaining = game.stonesRemaining(team);

    pushStyle();
    textAlign(LEFT, CENTER);
    textSize(11);
    fill(teamColor);
    text(team == TEAM_RED ? "Rod" : "Gul", xLeft, y);

    float dx = xLeft + 36;
    float d  = 10;
    noStroke();
    for (int i = 0; i < STONES_PER_TEAM; i++) {
      float x = dx + i * (d + 4);
      if (i < remaining) {
        fill(teamColor);
        ellipse(x, y, d, d);
      } else {
        noFill();
        stroke(teamColor, 110);
        strokeWeight(1);
        ellipse(x, y, d, d);
        noStroke();
      }
    }
    popStyle();
  }

  // The user-facing primary action. Called from a button click or
  // SPACE keypress. Cycles: lock-angle -> lock-speed-and-fire,
  // or resets the game when the end is over.
  void triggerAction() {
    if (game.state == GameState.ENDED) {
      game.reset();
    } else if (game.state == GameState.AIMING) {
      if (game.currentTeam == TEAM_YELLOW) return;
      if (lockPhase == PHASE_ANGLE) {
        angleBar.lockNow();
        lockPhase = PHASE_SPEED;
      } else {
        speedBar.lockNow();
        game.fire(currentShot());
        // lockPhase resets via ui.onTurnStart() on the next AIMING transition.
      }
    }
  }

  // ---- Mouse wiring (called from main sketch) ---------------
  void onMousePressed(float mx, float my) {
    if (shootBtn.enabled && shootBtn.hits(mx, my)) {
      triggerAction();
      return;
    }

    activeSlider = null;
    if      (curlSlider .trackHit(mx, my)) activeSlider = curlSlider;
    else if (speedSlider.trackHit(mx, my)) activeSlider = speedSlider;
    else if (angleSlider.trackHit(mx, my)) activeSlider = angleSlider;
    if (activeSlider != null) {
      activeSlider.dragging = true;
      activeSlider.setFromMouseY(my);
    }
  }

  void onMouseDragged(float mx, float my) {
    if (activeSlider != null) activeSlider.setFromMouseY(my);
  }

  void onMouseReleased(float mx, float my) {
    if (activeSlider != null) activeSlider.dragging = false;
    activeSlider = null;
  }

  void onMouseMoved(float mx, float my) {
    shootBtn.hover = shootBtn.hits(mx, my);
  }
}
