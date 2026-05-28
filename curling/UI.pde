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
  Button     rndStateBtn;
  Button     datasetBtn;
  Button     expertNextBtn;
  Button     expertNewBtn;
  Button     expertExitBtn;
  Button     trainBtn;
  Button     testBtn;
  Button     resetModelBtn;
  Button     penaltyHeuristicBtn;
  Button     pinHeuristicBtn;
  Button     scoreHeuristicBtn;
  Button     simShotsMinusBtn;
  Button     simShotsPlusBtn;
  Button     expertShotsMinusBtn;
  Button     expertShotsPlusBtn;
  Slider     activeSlider;

  int        trainComparisons = 100;
  int        shotsPerPrediction = 50;
  int        expertShotsPerPrediction = 10;
  boolean    draggingTrainBar = false;
  final int  TRAIN_MIN = 10;
  final int  TRAIN_MAX = 1000000;
  final float TRAIN_BAR_Y = 114;
  final float TRAIN_BAR_H = 10;
  final float HEURISTIC_BTN_Y = 146;
  final float HEURISTIC_BTN_H = 18;
  final float AI_STATUS_Y = 168;

  // Lock workflow phase. 0 = locking angle, 1 = locking speed.
  final int PHASE_ANGLE = 0;
  final int PHASE_SPEED = 1;
  int lockPhase;

  // ----- sidebar layout (vertical regions) -------------------
  // y =   0 .. 108 : stats (header, turn, status, stones-left)
  // y = 108 .. 172 : AI panel (train bar, heuristics, status)
  // y = 178 .. 673 : sliders
  // y = 708        : timing bar
  // y = 736        : shoot button
  // y = 794 / 840  : train / extra buttons
  // -----------------------------------------------------------
  final float STATS_TOP    =  16;
  final float STATS_BOTTOM = 108;
  final float SLIDER_TOP   = 178;
  final float SLIDER_H     = 495;
  final float BAR_Y        = 708;
  final float BAR_H        =  20;
  final float BTN_TOP      = 736;
  final float TRAIN_BTN_TOP    = 794;
  final float EXTRA_BTN_TOP    = 840;
  final float DATASET_BTN_TOP  = 882;
  final float EXPERT_BTN_TOP   = TRAIN_BTN_TOP;

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
    float halfW = (btnW - 6) * 0.5f;
    float halfX = ICE_W + (SIDEBAR_W - btnW) * 0.5f;
    shootBtn = new Button("L\u00e5s vinkel",
                          halfX,
                          BTN_TOP,
                          btnW, 52);

    trainBtn = new Button("Trana",
                          halfX,
                          TRAIN_BTN_TOP,
                          halfW, 40);
    testBtn  = new Button("Testa AI",
                          halfX + halfW + 6,
                          TRAIN_BTN_TOP,
                          halfW, 40);
    resetModelBtn = new Button("Ny modell",
                               halfX,
                               EXTRA_BTN_TOP,
                               halfW, 36);
    rndStateBtn = new Button("Random state",
                             halfX + halfW + 6,
                             EXTRA_BTN_TOP,
                             halfW, 36);
    datasetBtn = new Button("Expert dataset",
                            halfX,
                            DATASET_BTN_TOP,
                            btnW, 36);

    float expertGap = 6;
    float expertThirdW = (btnW - expertGap * 2) / 3.0f;
    expertNextBtn = new Button("N\u00e4sta",
                               halfX,
                               EXPERT_BTN_TOP,
                               expertThirdW, 40);
    expertNewBtn  = new Button("Ny state",
                               halfX + expertThirdW + expertGap,
                               EXPERT_BTN_TOP,
                               expertThirdW, 40);
    expertExitBtn = new Button("Avsluta",
                               halfX + (expertThirdW + expertGap) * 2,
                               EXPERT_BTN_TOP,
                               expertThirdW, 40);

    float hx = ICE_W + 20;
    float hgap = 4;
    float hw = (SIDEBAR_W - 40 - hgap * 2) / 3.0f;
    penaltyHeuristicBtn = new Button("Straff", hx,                    HEURISTIC_BTN_Y, hw, HEURISTIC_BTN_H);
    pinHeuristicBtn     = new Button("Pin",    hx + hw + hgap,         HEURISTIC_BTN_Y, hw, HEURISTIC_BTN_H);
    scoreHeuristicBtn   = new Button("Score",  hx + (hw + hgap) * 2,   HEURISTIC_BTN_Y, hw, HEURISTIC_BTN_H);

    float smallW = 18;
    float smallH = 14;
    float smallRight = ICE_W + SIDEBAR_W - 16;
    simShotsMinusBtn    = new Button("-", smallRight - smallW * 2 - 4, 126, smallW, smallH);
    simShotsPlusBtn     = new Button("+", smallRight - smallW,          126, smallW, smallH);
    expertShotsMinusBtn = new Button("-", smallRight - smallW * 2 - 4, 142, smallW, smallH);
    expertShotsPlusBtn  = new Button("+", smallRight - smallW,          142, smallW, smallH);
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

  void onRecordEnter() {
    onTurnStart();
  }

  void onRecordExit() {
    onTurnStart();
  }

  void setSlidersFromShot(Shot shot) {
    curlSlider.value  = constrain(shot.curl, curlSlider.minV, curlSlider.maxV);
    speedSlider.value = constrain(shot.speed / SPEED_MAX,
                                  speedSlider.minV, speedSlider.maxV);
    angleSlider.value = constrain(degrees(shot.angle),
                                  angleSlider.minV, angleSlider.maxV);
  }

  boolean recordMode() {
    return appMode == AppMode.RECORD;
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
    drawAiPanel();

    boolean controlsEnabled = appMode != AppMode.TRAINING;
    boolean showLockBars = controlsEnabled && !recordMode();
    curlSlider.draw();
    speedSlider.draw();
    angleSlider.draw();

    if (showLockBars) activeBar().draw();

    if (recordMode()) drawRecordPanel();

    if (appMode == AppMode.TEST) {
      shootBtn.label   = aiTestSimulating ? "..." : "Nasta test";
      shootBtn.enabled = controlsEnabled && !aiTestSimulating;
    } else if (game.state == GameState.ENDED) {
      shootBtn.label   = "Ny match";
      shootBtn.enabled = controlsEnabled;
    } else if (recordMode()) {
      shootBtn.label   = "Skjut";
      shootBtn.enabled = controlsEnabled
                      && recordSession != null && recordSession.canShoot();
    } else if (game.state == GameState.AIMING) {
      shootBtn.label   = (lockPhase == PHASE_ANGLE) ? "L\u00e5s vinkel" : "L\u00e5s fart";
      shootBtn.enabled = controlsEnabled && game.currentTeam == TEAM_RED;
    } else {
      shootBtn.label   = "...";
      shootBtn.enabled = false;
    }
    shootBtn.draw();

    if (recordMode()) {
      boolean recReady = controlsEnabled && recordSession != null
                      && recordSession.canSave();
      expertNextBtn.enabled = recReady;
      expertNewBtn.enabled  = recReady;
      expertExitBtn.enabled = controlsEnabled;
      expertNextBtn.draw();
      expertNewBtn.draw();
      expertExitBtn.draw();
      return;
    }

    datasetBtn.label   = "Expert dataset";
    datasetBtn.enabled = controlsEnabled && appMode != AppMode.TRAINING
                      && appMode != AppMode.TEST;
    datasetBtn.draw();

    trainBtn.label   = trainingActive ? "Avbryt" : "Trana";
    trainBtn.enabled = appMode != AppMode.TEST
                      && (trainingActive || controlsEnabled);
    trainBtn.draw();

    if (appMode == AppMode.TEST) {
      testBtn.label   = "Avsluta";
      testBtn.enabled = controlsEnabled;
    } else {
      testBtn.label   = "Testa AI";
      testBtn.enabled = controlsEnabled;
    }
    testBtn.draw();
    drawHeuristicButtons();

    resetModelBtn.label   = "Ny modell";
    resetModelBtn.enabled = controlsEnabled && appMode != AppMode.TEST;
    resetModelBtn.draw();

    rndStateBtn.enabled = controlsEnabled && appMode == AppMode.PLAY;
    rndStateBtn.draw();
  }

  void drawHeuristicButtons() {
    boolean canPick = appMode != AppMode.TRAINING && appMode != AppMode.TEST
                   && appMode != AppMode.RECORD;
    drawHeuristicButton(penaltyHeuristicBtn, trainPenaltyEnabled, canPick);
    drawHeuristicButton(pinHeuristicBtn, trainPinEnabled, canPick);
    drawHeuristicButton(scoreHeuristicBtn, trainScoreEnabled, canPick);
  }

  void drawHeuristicButton(Button btn, boolean selected, boolean enabled) {
    pushStyle();
    rectMode(CORNER);
    noStroke();
    if (!enabled)           fill(50);
    else if (selected)      fill(90, 160, 100);
    else if (btn.hover)     fill(70, 120, 160);
    else                    fill(45, 85, 130);
    rect(btn.x, btn.y, btn.w, btn.h, 4);
    fill(enabled ? 255 : 140);
    textAlign(CENTER, CENTER);
    textSize(11);
    text(btn.label, btn.x + btn.w * 0.5, btn.y + btn.h * 0.5);
    popStyle();
  }

  void drawAiPanel() {
    float left  = ICE_W + 16;
    float right = ICE_W + SIDEBAR_W - 16;
    float barW  = right - left;
    float barX  = left;

    pushStyle();
    textAlign(LEFT, TOP);
    fill(150);
    textSize(10);
    text("AI", left, TRAIN_BAR_Y - 14);

    noStroke();
    fill(50);
    rect(barX, TRAIN_BAR_Y, barW, TRAIN_BAR_H, 3);
    float frac = trainBarFraction();
    fill(80, 150, 210);
    rect(barX, TRAIN_BAR_Y, barW * frac, TRAIN_BAR_H, 3);
    float hx = barX + barW * frac;
    fill(220);
    ellipse(hx, TRAIN_BAR_Y + TRAIN_BAR_H * 0.5, 12, 12);

    textAlign(RIGHT, TOP);
    fill(180);
    textSize(9);
    text(formatTrainCount(trainComparisons) + " jamf.", right, TRAIN_BAR_Y - 14);

    fill(185);
    textAlign(LEFT, TOP);
    textSize(9);
    text("Shots/pred: " + shotsPerPrediction, left, 124);
    text("Expert/pred: " + expertShotsPerPrediction, left, 140);

    boolean canTuneCounts = appMode != AppMode.RECORD;
    simShotsMinusBtn.enabled = canTuneCounts;
    simShotsPlusBtn.enabled = canTuneCounts;
    expertShotsMinusBtn.enabled = canTuneCounts;
    expertShotsPlusBtn.enabled = canTuneCounts;
    simShotsMinusBtn.draw();
    simShotsPlusBtn.draw();
    expertShotsMinusBtn.draw();
    expertShotsPlusBtn.draw();

    fill(200);
    textAlign(LEFT, TOP);
    textSize(9);
    if (appMode == AppMode.TRAINING) {
      text("Traning: " + trainingDone + " / " + trainingTarget
           + "  (" + heuristicLabel() + ")", left, AI_STATUS_Y);
    } else if (trainingStatus.length() > 0) {
      text(trainingStatus + "  (" + heuristicLabel() + ")", left, AI_STATUS_Y);
    } else if (appMode == AppMode.TEST) {
      text("Test: Rod " + testHumanWins + "  Gul " + testAiWins
           + "  (" + testGamesPlayed + " sim)", left, AI_STATUS_Y);
    } else if (appMode == AppMode.RECORD) {
      int n = expertShots != null ? expertShots.count() : 0;
      text("Dataset: " + n + " rader  |  " + expertShots.csvPath, left, AI_STATUS_Y);
    } else {
      text("Heuristik: " + heuristicLabel() + "  |  " + trainingDone + " jamf.", left, AI_STATUS_Y);
    }
    popStyle();
  }

  String heuristicLabel() {
    String label = "";
    if (trainPenaltyEnabled) label += (label.length() > 0 ? "+" : "") + "Straff";
    if (trainPinEnabled)     label += (label.length() > 0 ? "+" : "") + "Pin";
    if (trainScoreEnabled)   label += (label.length() > 0 ? "+" : "") + "Score";
    return label.length() > 0 ? label : "Pin";
  }

  int enabledHeuristicCount() {
    int n = 0;
    if (trainPenaltyEnabled) n++;
    if (trainPinEnabled)     n++;
    if (trainScoreEnabled)   n++;
    return n;
  }

  void togglePenaltyHeuristic() {
    if (trainPenaltyEnabled && enabledHeuristicCount() <= 1) return;
    trainPenaltyEnabled = !trainPenaltyEnabled;
  }

  void togglePinHeuristic() {
    if (trainPinEnabled && enabledHeuristicCount() <= 1) return;
    trainPinEnabled = !trainPinEnabled;
  }

  void toggleScoreHeuristic() {
    if (trainScoreEnabled && enabledHeuristicCount() <= 1) return;
    trainScoreEnabled = !trainScoreEnabled;
  }

  String formatTrainCount(int n) {
    if (n >= 1000000) return (n / 1000000) + "M";
    if (n >= 1000)    return (n / 1000) + "k";
    return str(n);
  }

  void adjustShotsPerPrediction(int delta) {
    shotsPerPrediction = constrain(shotsPerPrediction + delta, 1, 500);
  }

  void adjustExpertShotsPerPrediction(int delta) {
    expertShotsPerPrediction = constrain(expertShotsPerPrediction + delta, 0, 500);
  }

  float trainBarX() { return ICE_W + 16; }
  float trainBarW() { return SIDEBAR_W - 32; }

  float trainBarFraction() {
    float lo = log((float) TRAIN_MIN) / log(10);
    float hi = log((float) TRAIN_MAX) / log(10);
    float v  = log((float) trainComparisons) / log(10);
    return constrain((v - lo) / (hi - lo), 0, 1);
  }

  void setTrainComparisonsFromMouse(float mx) {
    float f = constrain((mx - trainBarX()) / trainBarW(), 0, 1);
    float lo = log((float) TRAIN_MIN) / log(10);
    float hi = log((float) TRAIN_MAX) / log(10);
    trainComparisons = round(pow(10, lo + f * (hi - lo)));
  }

  boolean trainBarHit(float mx, float my) {
    return mx >= trainBarX() && mx <= trainBarX() + trainBarW()
        && my >= TRAIN_BAR_Y - 4 && my <= TRAIN_BAR_Y + TRAIN_BAR_H + 4;
  }

  // ----- stats panel: header + turn + status + stones-left ----
  void drawRecordPanel() {
    float left = ICE_W + 16;
    pushStyle();
    textAlign(LEFT, TOP);
    fill(200);
    textSize(10);
    text("Dra stenar p\u00e5 isen f\u00f6r att \u00e4ndra layout", left, BAR_Y - 28);
    text("N\u00e4sta / Ny state = ny layout + AI-f\u00f6rslag", left, BAR_Y - 16);
    popStyle();
  }

  void drawStatsPanel() {
    float left  = ICE_W + 16;
    float right = ICE_W + SIDEBAR_W - 16;

    pushStyle();
    textAlign(LEFT, TOP);
    fill(235);
    textSize(15);
    text(recordMode() ? "Expert-skott" : "Skott", left, STATS_TOP);

    fill(170);
    textSize(11);
    text("Tur",       left, STATS_TOP + 26);
    text("Status",    left, STATS_TOP + 44);

    textAlign(RIGHT, TOP);
    if (recordMode()) {
      fill(color(230, 210, 80));
      text("GUL", right, STATS_TOP + 26);
      fill(220);
      text(recordSession != null && recordSession.simulating
           ? "SIMULERAR" : "AIMING", right, STATS_TOP + 44);
    } else {
      fill(game.currentTeam == TEAM_RED ? color(230, 80, 80) : color(230, 210, 80));
      text(game.teamLabel(game.currentTeam), right, STATS_TOP + 26);
      fill(220);
      text(game.stateLabel(),                right, STATS_TOP + 44);
    }

    // Two rows of stone-count dots.
    drawTeamRow(left, STATS_TOP + 64, TEAM_RED);
    drawTeamRow(left, STATS_TOP + 82, TEAM_YELLOW);

    // Separator above AI panel.
    stroke(60);
    strokeWeight(1);
    line(ICE_W + 12, STATS_BOTTOM, ICE_W + SIDEBAR_W - 12, STATS_BOTTOM);
    popStyle();
  }

  // One team's row: label + 4 dots (filled = remaining, hollow = thrown).
  void drawTeamRow(float xLeft, float y, int team) {
    color teamColor = team == TEAM_RED ? color(230, 80, 80) : color(230, 210, 80);
    int   remaining = recordMode()
      ? (team == TEAM_YELLOW ? 1 : 0)
      : game.stonesRemaining(team);

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
    if (appMode == AppMode.TRAINING) return;
    if (recordMode()) {
      if (recordSession != null && recordSession.canShoot()) {
        recordSession.shoot(intendedShot());
      }
      return;
    }
    if (appMode == AppMode.TEST) {
      if (!aiTestSimulating) runAiTestSim();
      return;
    }
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

  void setRndStones(){
    RandomState rs = new RandomState();
    rs.randomize(game.stones, STONES_PER_TEAM);
    game.stonesThrown = TOTAL_STONES - 1;
    game.currentTeam = TEAM_YELLOW;
  }

  // ---- Mouse wiring (called from main sketch) ---------------
  void onMousePressed(float mx, float my) {
    if (recordMode()) {
      if (expertExitBtn.enabled && expertExitBtn.hits(mx, my)) {
        stopRecordMode();
        return;
      }
      if (expertNextBtn.enabled && expertNextBtn.hits(mx, my)) {
        recordNextShot();
        return;
      }
      if (expertNewBtn.enabled && expertNewBtn.hits(mx, my)) {
        recordNewState();
        return;
      }
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
      return;
    }

    if (simShotsMinusBtn.enabled && simShotsMinusBtn.hits(mx, my)) {
      adjustShotsPerPrediction(-1);
      return;
    }
    if (simShotsPlusBtn.enabled && simShotsPlusBtn.hits(mx, my)) {
      adjustShotsPerPrediction(1);
      return;
    }
    if (expertShotsMinusBtn.enabled && expertShotsMinusBtn.hits(mx, my)) {
      adjustExpertShotsPerPrediction(-1);
      return;
    }
    if (expertShotsPlusBtn.enabled && expertShotsPlusBtn.hits(mx, my)) {
      adjustExpertShotsPerPrediction(1);
      return;
    }

    if (trainBtn.enabled && trainBtn.hits(mx, my)) {
      if (trainingActive) cancelTraining();
      else startTraining(trainComparisons);
      return;
    }
    if (resetModelBtn.enabled && resetModelBtn.hits(mx, my)) {
      resetTrainingModel();
      return;
    }
    if (appMode != AppMode.TRAINING && appMode != AppMode.TEST
        && appMode != AppMode.RECORD) {
      if (penaltyHeuristicBtn.hits(mx, my)) {
        togglePenaltyHeuristic();
        return;
      }
      if (pinHeuristicBtn.hits(mx, my)) {
        togglePinHeuristic();
        return;
      }
      if (scoreHeuristicBtn.hits(mx, my)) {
        toggleScoreHeuristic();
        return;
      }
    }
    if (testBtn.enabled && testBtn.hits(mx, my)) {
      if (appMode == AppMode.TEST) stopAiTest();
      else startAiTest();
      return;
    }
    if (trainBarHit(mx, my) && appMode != AppMode.TRAINING && appMode != AppMode.RECORD) {
      draggingTrainBar = true;
      setTrainComparisonsFromMouse(mx);
      return;
    }

    if (shootBtn.enabled && shootBtn.hits(mx, my)) {
      triggerAction();
      return;
    }

    if (rndStateBtn.enabled && rndStateBtn.hits(mx, my)){
      setRndStones();
      return;
    }

    if (datasetBtn.enabled && datasetBtn.hits(mx, my)) {
      startRecordMode();
      return;
    }

    activeSlider = null;
    if (appMode == AppMode.TRAINING || appMode == AppMode.TEST) return;
    if      (curlSlider .trackHit(mx, my)) activeSlider = curlSlider;
    else if (speedSlider.trackHit(mx, my)) activeSlider = speedSlider;
    else if (angleSlider.trackHit(mx, my)) activeSlider = angleSlider;
    if (activeSlider != null) {
      activeSlider.dragging = true;
      activeSlider.setFromMouseY(my);
    }
  }

  void onMouseDragged(float mx, float my) {
    if (draggingTrainBar) {
      setTrainComparisonsFromMouse(mx);
      return;
    }
    if (activeSlider != null) activeSlider.setFromMouseY(my);
  }

  void onMouseReleased(float mx, float my) {
    draggingTrainBar = false;
    if (activeSlider != null) activeSlider.dragging = false;
    activeSlider = null;
  }

  void onMouseMoved(float mx, float my) {
    shootBtn.hover = shootBtn.hits(mx, my);
    trainBtn.hover = trainBtn.hits(mx, my);
    testBtn.hover  = testBtn.hits(mx, my);
    resetModelBtn.hover = resetModelBtn.hits(mx, my);
    datasetBtn.hover = datasetBtn.hits(mx, my);
    expertNextBtn.hover = expertNextBtn.hits(mx, my);
    expertNewBtn.hover  = expertNewBtn.hits(mx, my);
    expertExitBtn.hover = expertExitBtn.hits(mx, my);
    simShotsMinusBtn.hover = simShotsMinusBtn.hits(mx, my);
    simShotsPlusBtn.hover = simShotsPlusBtn.hits(mx, my);
    expertShotsMinusBtn.hover = expertShotsMinusBtn.hits(mx, my);
    expertShotsPlusBtn.hover = expertShotsPlusBtn.hits(mx, my);
    penaltyHeuristicBtn.hover = penaltyHeuristicBtn.hits(mx, my);
    pinHeuristicBtn.hover = pinHeuristicBtn.hits(mx, my);
    scoreHeuristicBtn.hover = scoreHeuristicBtn.hits(mx, my);
  }
}
