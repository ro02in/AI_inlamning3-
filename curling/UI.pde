// =============================================================
// UI - sliders + Skjut button drawn in the right sidebar.
// =============================================================

class Slider {
  String  label;
  float   minV, maxV, value;
  float   cx, y, h;
  float   hitHalfW;
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
    noStroke();
    fill(220);
    textAlign(CENTER, BOTTOM);
    textSize(12);
    text(label, cx, y - 8);
    stroke(80);
    strokeWeight(2);
    line(cx, y, cx, y + h);
    stroke(60);
    strokeWeight(1);
    line(cx - 8, y,     cx + 8, y);
    line(cx - 8, y + h, cx + 8, y + h);
    if (minV < 0 && maxV > 0) {
      float zy = y + h * (maxV / (maxV - minV));
      line(cx - 8, zy, cx + 8, zy);
    }
    float hy = y + h * (1 - fraction());
    noStroke();
    fill(dragging ? color(255) : color(200));
    rect(cx - 14, hy - 5, 28, 10, 3);
    fill(230);
    textAlign(CENTER, TOP);
    textSize(12);
    text(nf(value, 0, decimals), cx, y + h + 8);
    popStyle();
  }
}

// =============================================================
// TimingBar
// =============================================================
class TimingBar {
  String  label;
  float   x, y, w, h;
  float   oscPhase;
  float   oscHz;
  float   greenFrac;
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

  float indicatorFraction() {
    float p = locked ? lockedPhase : oscPhase;
    return p < 0.5 ? p * 2.0 : 2.0 - p * 2.0;
  }

  float signedError() {
    float pos       = indicatorFraction();
    float deviation = pos - 0.5;
    float magnitude = abs(deviation) * 2.0;
    if (magnitude <= greenFrac) return 0;
    float em = (magnitude - greenFrac) / (1.0 - greenFrac);
    return deviation > 0 ? em : -em;
  }

  boolean hits(float mx, float my) {
    return mx >= x && mx <= x + w
        && my >= y - 6 && my <= y + h + 6;
  }

  void draw() {
    pushStyle();
    rectMode(CORNER);
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
    noStroke();
    fill(150, 40, 40);
    rect(x, y, w, h, 4);
    float gw = w * greenFrac;
    float gx = x + (w - gw) * 0.5;
    fill(50, 170, 70);
    rect(gx, y, gw, h, 4);
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
  Button     resetGameBtn;
  Button     startPlayerBtn;
  Button     rndStateBtn;
  Button     trainBtn;
  Button     testBtn;
  Button     resetModelBtn;
  Button     saveModelBtn;
  Button     loadModelBtn;
  Slider     activeSlider;

  int        trainComparisons = 100;
  boolean    draggingTrainBar = false;
  final int  TRAIN_MIN = 10;
  final int  TRAIN_MAX = 1000000;
  final float TRAIN_BAR_Y = 114;
  final float TRAIN_BAR_H = 10;
  final float AI_STATUS_Y = 134;

  final int PHASE_ANGLE = 0;
  final int PHASE_SPEED = 1;
  int lockPhase;

  final float STATS_TOP    =  16;
  final float STATS_BOTTOM = 108;
  final float GAME_BTN_TOP = 154;
  final float SLIDER_TOP   = 204;
  final float SLIDER_H     = 469;
  final float BAR_Y        = 708;
  final float BAR_H        =  20;
  final float BTN_TOP      = 736;
  final float TRAIN_BTN_TOP    = 794;
  final float EXTRA_BTN_TOP    = 840;

  final float LOCK_BASE_HZ            = 0.7;
  final float LOCK_CURL_HZ_MULT       = 1.1;
  final float LOCK_GREEN_FRAC_BASE    = 0.45;
  final float LOCK_GREEN_FRAC_MIN     = 0.18;
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
    angleBar = new TimingBar("L\u00e5s vinkel", barX, BAR_Y, barW, BAR_H);
    speedBar = new TimingBar("L\u00e5s fart",   barX, BAR_Y, barW, BAR_H);

    float btnW    = SIDEBAR_W - 40;
    float halfW   = (btnW - 6) * 0.5f;
    float thirdW  = (btnW - 6 * 2) / 3.0f;
    float quarterW = (btnW - 6 * 3) / 4.0f;
    float halfX   = ICE_W + (SIDEBAR_W - btnW) * 0.5f;

    shootBtn = new Button("L\u00e5s vinkel", halfX, BTN_TOP, btnW, 52);
    resetGameBtn = new Button("Ny match", halfX, GAME_BTN_TOP, halfW, 28);
    startPlayerBtn = new Button("Start: Rod", halfX + halfW + 6, GAME_BTN_TOP, halfW, 28);

    trainBtn = new Button("Trana",    halfX,          TRAIN_BTN_TOP, halfW, 40);
    testBtn  = new Button("Testa AI", halfX + halfW + 6, TRAIN_BTN_TOP, halfW, 40);

    resetModelBtn = new Button("Ny modell",
                               halfX,
                               EXTRA_BTN_TOP,
                               quarterW, 36);
    saveModelBtn  = new Button("Spara",
                               halfX + quarterW + 6,
                               EXTRA_BTN_TOP,
                               quarterW, 36);
    loadModelBtn  = new Button("Ladda",
                               halfX + (quarterW + 6) * 2,
                               EXTRA_BTN_TOP,
                               quarterW, 36);
    rndStateBtn   = new Button("Rand",
                               halfX + (quarterW + 6) * 3,
                               EXTRA_BTN_TOP,
                               quarterW, 36);

    lockPhase = PHASE_ANGLE;
  }

  TimingBar activeBar() {
    return (lockPhase == PHASE_ANGLE) ? angleBar : speedBar;
  }

  void update(float dt) {
    float absCurl = abs(curlSlider.value);
    float hz      = LOCK_BASE_HZ + absCurl * LOCK_CURL_HZ_MULT;
    float greenF  = lerp(LOCK_GREEN_FRAC_BASE, LOCK_GREEN_FRAC_MIN, absCurl);
    TimingBar bar = activeBar();
    bar.configure(hz, greenF);
    bar.update(dt);
  }

  void onTurnStart() {
    angleBar.reset();
    speedBar.reset();
    lockPhase = PHASE_ANGLE;
  }

  final static float SPEED_MAX = 40.0;

  Shot intendedShot() {
    return new Shot(curlSlider.value,
                    speedSlider.value * SPEED_MAX,
                    radians(angleSlider.value));
  }

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
    boolean showLockBars    = controlsEnabled;
    curlSlider.draw();
    speedSlider.draw();
    angleSlider.draw();

    if (showLockBars) activeBar().draw();

    resetGameBtn.label = "Ny match";
    resetGameBtn.enabled = true;
    resetGameBtn.draw();
    startPlayerBtn.label = game.startingTeam == TEAM_RED ? "Start: Rod" : "Start: Gul";
    startPlayerBtn.enabled = true;
    startPlayerBtn.draw();

    if (appMode == AppMode.TEST) {
      shootBtn.label   = aiTestSimulating ? "..." : "Nasta test";
      shootBtn.enabled = controlsEnabled && !aiTestSimulating;
    } else if (game.state == GameState.ENDED) {
      shootBtn.label   = "Ny match";
      shootBtn.enabled = controlsEnabled;
    } else if (game.state == GameState.AIMING) {
      shootBtn.label   = (lockPhase == PHASE_ANGLE) ? "L\u00e5s vinkel" : "L\u00e5s fart";
      shootBtn.enabled = controlsEnabled && game.currentTeam == TEAM_RED;
    } else {
      shootBtn.label   = "...";
      shootBtn.enabled = false;
    }
    shootBtn.draw();

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

    resetModelBtn.label   = "Ny modell";
    resetModelBtn.enabled = controlsEnabled && appMode != AppMode.TEST;
    resetModelBtn.draw();

    saveModelBtn.enabled = controlsEnabled && appMode != AppMode.TEST
                        && appMode != AppMode.TRAINING;
    loadModelBtn.enabled = saveModelBtn.enabled;
    saveModelBtn.draw();
    loadModelBtn.draw();

    rndStateBtn.enabled = controlsEnabled && appMode == AppMode.PLAY;
    rndStateBtn.draw();
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
    text("AI  (gradient ensemble)", left, TRAIN_BAR_Y - 14);

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
    text(formatTrainCount(trainComparisons) + " steg", right, TRAIN_BAR_Y - 14);

    fill(185);
    textAlign(LEFT, TOP);
    textSize(9);
    int depthCap = curriculumDepthCap();
    text("Djup: " + depthCap + "/" + TOTAL_STONES
         + "  Shots/round: " + SHOTS_PER_ROUND, left, AI_STATUS_Y);

    fill(200);
    textAlign(LEFT, TOP);
    textSize(9);
    if (appMode == AppMode.TRAINING) {
      text("Traning: " + trainingDone + " / " + trainingTarget, left, AI_STATUS_Y + 14);
    } else if (trainingStatus.length() > 0) {
      text(trainingStatus, left, AI_STATUS_Y + 14);
    } else if (appMode == AppMode.TEST) {
      text("Test: Rod " + testHumanWins + "  Gul " + testAiWins
           + "  (" + testGamesPlayed + " sim)", left, AI_STATUS_Y + 14);
    } else {
      text(trainingDone + " steg klara", left, AI_STATUS_Y + 14);
    }
    popStyle();
  }

  String formatTrainCount(int n) {
    if (n >= 1000000) return (n / 1000000) + "M";
    if (n >= 1000)    return (n / 1000) + "k";
    return str(n);
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
    float f  = constrain((mx - trainBarX()) / trainBarW(), 0, 1);
    float lo = log((float) TRAIN_MIN) / log(10);
    float hi = log((float) TRAIN_MAX) / log(10);
    trainComparisons = round(pow(10, lo + f * (hi - lo)));
  }

  boolean trainBarHit(float mx, float my) {
    return mx >= trainBarX() && mx <= trainBarX() + trainBarW()
        && my >= TRAIN_BAR_Y - 4 && my <= TRAIN_BAR_Y + TRAIN_BAR_H + 4;
  }

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
    text("Tur",    left, STATS_TOP + 26);
    text("Status", left, STATS_TOP + 44);

    textAlign(RIGHT, TOP);
    fill(game.currentTeam == TEAM_RED ? color(230, 80, 80) : color(230, 210, 80));
    text(game.teamLabel(game.currentTeam), right, STATS_TOP + 26);
    fill(220);
    text(game.stateLabel(), right, STATS_TOP + 44);

    drawTeamRow(left, STATS_TOP + 64, TEAM_RED);
    drawTeamRow(left, STATS_TOP + 82, TEAM_YELLOW);

    stroke(60);
    strokeWeight(1);
    line(ICE_W + 12, STATS_BOTTOM, ICE_W + SIDEBAR_W - 12, STATS_BOTTOM);
    popStyle();
  }

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

  void triggerAction() {
    if (appMode == AppMode.TRAINING) return;
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
      }
    }
  }

  void resetGameNow() {
    if (trainingActive) {
      trainingActive = false;
      trainingPreview.reset();
    }
    if (appMode == AppMode.TEST) stopAiTest();
    appMode = AppMode.PLAY;
    game.reset();
  }

  void toggleStartingPlayer() {
    if (trainingActive) {
      trainingActive = false;
      trainingPreview.reset();
    }
    if (appMode == AppMode.TEST) stopAiTest();
    appMode = AppMode.PLAY;
    game.toggleStartingTeam();
  }

  void setRndStones() {
    RandomState rs = new RandomState();
    rs.randomize(game.stones, STONES_PER_TEAM);
    game.stonesThrown = TOTAL_STONES - 1;
    game.currentTeam  = TEAM_YELLOW;
  }

  void onMousePressed(float mx, float my) {
    if (resetGameBtn.enabled && resetGameBtn.hits(mx, my)) { resetGameNow(); return; }
    if (startPlayerBtn.enabled && startPlayerBtn.hits(mx, my)) { toggleStartingPlayer(); return; }

    if (trainBtn.enabled      && trainBtn.hits(mx, my))      {
      if (trainingActive) cancelTraining(); else startTraining(trainComparisons); return;
    }
    if (resetModelBtn.enabled && resetModelBtn.hits(mx, my)) { resetTrainingModel(); return; }
    if (saveModelBtn.enabled  && saveModelBtn.hits(mx, my))  { promptSaveModel();    return; }
    if (loadModelBtn.enabled  && loadModelBtn.hits(mx, my))  { promptLoadModel();    return; }

    if (testBtn.enabled && testBtn.hits(mx, my)) {
      if (appMode == AppMode.TEST) stopAiTest(); else startAiTest(); return;
    }
    if (trainBarHit(mx, my) && appMode != AppMode.TRAINING) {
      draggingTrainBar = true;
      setTrainComparisonsFromMouse(mx);
      return;
    }
    if (shootBtn.enabled && shootBtn.hits(mx, my)) { triggerAction(); return; }
    if (rndStateBtn.enabled && rndStateBtn.hits(mx, my)) { setRndStones(); return; }
    activeSlider = null;
    if (appMode == AppMode.TRAINING || appMode == AppMode.TEST) return;
    if      (curlSlider .trackHit(mx, my)) activeSlider = curlSlider;
    else if (speedSlider.trackHit(mx, my)) activeSlider = speedSlider;
    else if (angleSlider.trackHit(mx, my)) activeSlider = angleSlider;
    if (activeSlider != null) { activeSlider.dragging = true; activeSlider.setFromMouseY(my); }
  }

  void onMouseDragged(float mx, float my) {
    if (draggingTrainBar) { setTrainComparisonsFromMouse(mx); return; }
    if (activeSlider != null) activeSlider.setFromMouseY(my);
  }

  void onMouseReleased(float mx, float my) {
    draggingTrainBar = false;
    if (activeSlider != null) activeSlider.dragging = false;
    activeSlider = null;
  }

  void onMouseMoved(float mx, float my) {
    shootBtn.hover      = shootBtn.hits(mx, my);
    resetGameBtn.hover  = resetGameBtn.hits(mx, my);
    startPlayerBtn.hover = startPlayerBtn.hits(mx, my);
    trainBtn.hover      = trainBtn.hits(mx, my);
    testBtn.hover       = testBtn.hits(mx, my);
    resetModelBtn.hover = resetModelBtn.hits(mx, my);
    saveModelBtn.hover  = saveModelBtn.hits(mx, my);
    loadModelBtn.hover  = loadModelBtn.hits(mx, my);
    rndStateBtn.hover   = rndStateBtn.hits(mx, my);
  }
}
