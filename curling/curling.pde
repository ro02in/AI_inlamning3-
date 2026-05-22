// =============================================================
// Curling - main sketch
// =============================================================
// World coordinates are in FEET (see Sheet.pde). +y runs from the
// hack toward the house and hog (top of the ice view). Stones
// travel with positive vy when angle = 0.
//
// The ice is drawn procedurally from regulation dimensions; the
// global Sheet handles feet-to-screen mapping with a uniform scale
// so stones stay circular.
// =============================================================

// ----- Viewport (pixels) ---------------------------------------
// Ice panel on the left; UI sidebar on the right.
final int ICE_W     = 540;
final int ICE_H     = 960;
final int SIDEBAR_W = 220;
final int WIN_W     = ICE_W + SIDEBAR_W;
final int WIN_H     = ICE_H;

// ----- Assets ------------------------------------------------
PImage stoneRedImg;
PImage stoneYellowImg;

// ----- World / sheet ------------------------------------------
Sheet sheet;

// ----- Game world -------------------------------------------
Game    game;
House   house;
Physics physics;
UI      ui;

// Fixed simulation timestep (seconds). Drives Physics.step.
final float DT = 1.0 / 60.0;

// Toggle with 'd' to show tee and hack markers.
boolean DEBUG = false;

// ----- AI training / test ------------------------------------
enum AppMode { PLAY, TRAINING, TEST }

AppMode appMode = AppMode.PLAY;
boolean trainPenaltyEnabled = false;
boolean trainPinEnabled     = true;
boolean trainScoreEnabled   = false;
PolicySearchTraining trainer;
NeuralPolicy aiPolicy;
ScoreHeuristic scoreHeuristic;
CloseToButtonHeuristic pinHeuristic;
PenaltyHeuristic penaltyHeuristic;
TrainingPreview    trainingPreview;

int  trainingTarget = 100;
int  trainingDone   = 0;
boolean trainingActive = false;
String trainingStatus = "";

int testGamesPlayed = 0;
int testHumanWins     = 0;
int testAiWins        = 0;
int testBlankEnds     = 0;

ArrayList<Stone> aiTestStones;
RandomState      aiTestRandom;
boolean          aiTestSimulating = false;
ScoreResult      aiTestScore;
Shot             aiTestLastShot;

// ----- Setup / draw ------------------------------------------
void settings() {
  size(WIN_W, WIN_H);
}

void setup() {
  frameRate(60);
  imageMode(CENTER);
  textAlign(LEFT, TOP);

  stoneRedImg    = loadImage("curling-stone-red.png");
  stoneYellowImg = loadImage("curling-stone-yellow.png");

  sheet   = new Sheet();
  physics = new Physics();
  house   = new House();
  ui      = new UI();
  game    = new Game();

  trainer       = new PolicySearchTraining();
  scoreHeuristic   = new ScoreHeuristic();
  pinHeuristic     = new CloseToButtonHeuristic();
  penaltyHeuristic = new PenaltyHeuristic();
  trainingPreview  = new TrainingPreview();
  aiPolicy         = trainer.current.copy();
}

ArrayList<Heuristic> activeTrainingHeuristics() {
  ArrayList<Heuristic> heuristics = new ArrayList<Heuristic>();
  if (trainPenaltyEnabled) heuristics.add(penaltyHeuristic);
  if (trainPinEnabled)     heuristics.add(pinHeuristic);
  if (trainScoreEnabled)   heuristics.add(scoreHeuristic);
  if (heuristics.isEmpty()) heuristics.add(pinHeuristic); // safety: at least one
  return heuristics;
}

void draw() {
  background(20);

  if (appMode == AppMode.TRAINING) {
    if (trainingActive) {
      trainer.comparePolicies(activeTrainingHeuristics());
      trainingDone++;
      trainingPreview.recordSnapshot(trainingDone, trainer.current);
      if (trainingDone >= trainingTarget) {
        trainingActive = false;
        appMode = AppMode.PLAY;
        aiPolicy = trainer.current.copy();
        trainingStatus = "Klar! (" + trainingDone + " jamf.)";
        trainingPreview.reset();
      }
    }
    trainingPreview.update(DT);
  } else if (appMode == AppMode.TEST) {
    updateAiTest();
    ui.update(DT);
  } else {
    physics.step(game.stones, DT);
    game.update();
    maybeAiShoot();
    ui.update(DT);
  }

  sheet.drawSheet();
  if (appMode == AppMode.TEST && aiTestStones != null) {
    if (aiTestLastShot != null) drawShotPreview(aiTestLastShot, color(235, 205, 60));
    for (Stone s : aiTestStones) s.draw();
    drawAiTestOverlay();
  } else if (appMode == AppMode.TRAINING) {
    trainingPreview.drawStones();
  } else {
    if (game.state == GameState.AIMING && game.currentTeam == TEAM_RED) {
      drawAimPreview(ui.intendedShot());
    }
    for (Stone s : game.stones) s.draw();
    if (appMode != AppMode.TRAINING) drawEndOverlay();
  }
  if (DEBUG) drawDebugMarkers();

  drawSidebarBackdrop();
  ui.draw();
  if (appMode == AppMode.TRAINING) trainingPreview.drawOverlay(trainingDone, trainingTarget);
}
void startTraining(int comparisons) {
  if (trainingActive) return;
  trainingTarget = trainingDone + comparisons;
  trainingActive = true;
  trainingStatus = "";
  trainingPreview.reset();
  appMode        = AppMode.TRAINING;
}

void cancelTraining() {
  if (!trainingActive) return;
  trainingActive = false;
  appMode        = AppMode.PLAY;
  aiPolicy       = trainer.current.copy();
  trainingStatus = "Avbruten (" + trainingDone + " jamf.)";
  trainingPreview.reset();
}

void resetTrainingModel() {
  if (trainingActive) return;
  trainer.reset();
  aiPolicy       = trainer.current.copy();
  trainingDone   = 0;
  trainingTarget = 0;
  trainingStatus = "Ny modell";
}

void startAiTest() {
  appMode = AppMode.TEST;
  aiPolicy = trainer.current.copy();
  testGamesPlayed = 0;
  testHumanWins     = 0;
  testAiWins        = 0;
  testBlankEnds     = 0;
  if (aiTestRandom == null) aiTestRandom = new RandomState();
  runAiTestSim();
}

void stopAiTest() {
  appMode = AppMode.PLAY;
  aiTestSimulating = false;
  aiTestStones = null;
  aiTestScore = null;
  aiTestLastShot = null;
}

void runAiTestSim() {
  aiTestStones = new ArrayList<Stone>();
  aiTestRandom.randomize(aiTestStones, STONES_PER_TEAM);
  float[] state = aiPolicy.convertState(aiTestStones, 1, TEAM_RED);
  aiTestLastShot = aiPolicy.predict(state);
  new PolicyDiagnostics().logTestShot(aiPolicy, aiTestStones, 1, TEAM_RED);

  PVector h = sheet.hackWorld();
  Stone fired = new Stone(h.x, h.y, TEAM_YELLOW);
  fired.curl = constrain(aiTestLastShot.curl, -1, 1);
  fired.vel.set(sin(aiTestLastShot.angle) * aiTestLastShot.speed,
                cos(aiTestLastShot.angle) * aiTestLastShot.speed);
  aiTestStones.add(fired);

  aiTestSimulating = true;
  aiTestScore = null;
}

void updateAiTest() {
  if (!aiTestSimulating || aiTestStones == null) return;

  physics.step(aiTestStones, DT);
  boolean anyMoving = false;
  for (Stone s : aiTestStones) {
    if (s.isMoving()) { anyMoving = true; break; }
  }
  if (!anyMoving) {
    aiTestSimulating = false;
    aiTestScore = house.scoreEnd(aiTestStones);
    testGamesPlayed++;
    if (aiTestScore.team == TEAM_YELLOW)      testAiWins++;
    else if (aiTestScore.team == TEAM_RED)    testHumanWins++;
    else testBlankEnds++;
  }
}

void drawAiTestOverlay() {
  pushStyle();
  if (aiTestSimulating) {
    fill(0, 140);
    noStroke();
    rect(0, ICE_H - 36, ICE_W, 36);
    fill(240);
    textAlign(CENTER, CENTER);
    textSize(14);
    text("Simulerar...", ICE_W * 0.5, ICE_H - 18);
    popStyle();
    return;
  }
  if (aiTestScore == null) { popStyle(); return; }

  fill(0, 200);
  noStroke();
  rect(0, ICE_H * 0.5 - 55, ICE_W, 110);
  textAlign(CENTER, CENTER);
  if (aiTestScore.team < 0) {
    fill(230);
    textSize(24);
    text("Oavgjord", ICE_W * 0.5, ICE_H * 0.5 - 10);
  } else {
    int t = aiTestScore.team;
    color c = t == TEAM_RED ? color(230, 80, 80) : color(230, 210, 80);
    String n = t == TEAM_RED ? "Rod" : "Gul (AI)";
    fill(c);
    textSize(28);
    text(n + " vinner", ICE_W * 0.5, ICE_H * 0.5 - 12);
    fill(240);
    textSize(18);
    text("med " + aiTestScore.points + " poang", ICE_W * 0.5, ICE_H * 0.5 + 16);
  }
  fill(200);
  textSize(13);
  text("Serie: Rod " + testHumanWins + " - " + testAiWins + " Gul  ("
       + testGamesPlayed + ")", ICE_W * 0.5, ICE_H * 0.5 + 42);
  popStyle();
}

// ----- Mouse / keyboard input --------------------------------
void mousePressed()  { ui.onMousePressed(mouseX, mouseY); }
void mouseDragged()  { ui.onMouseDragged(mouseX, mouseY); }
void mouseReleased() { ui.onMouseReleased(mouseX, mouseY); }
void mouseMoved()    { ui.onMouseMoved(mouseX, mouseY); }

void keyPressed() {
  if (key == ' ') {
    ui.triggerAction();
  } else if (key == 'd' || key == 'D') {
    DEBUG = !DEBUG;
  } else if (key == 'r' || key == 'R') {
    game.reset();
  }
}

void maybeAiShoot() {
  if (appMode != AppMode.PLAY) return;
  if (game.state != GameState.AIMING || game.currentTeam != TEAM_YELLOW) return;

  int lastTeam = TEAM_RED;
  if (game.stones.size() > 0) {
    lastTeam = game.stones.get(game.stones.size() - 1).team;
  }

  float[] state = aiPolicy.convertState(
    game.stones,
    game.stonesRemaining(TEAM_YELLOW),
    lastTeam
  );
  game.fire(aiPolicy.predict(state));
}

// ----- Aim preview: forward-simulated trajectory in team color
void drawAimPreview(Shot shot) {
  color teamColor = (game.currentTeam == TEAM_RED)
      ? color(235, 70, 70)
      : color(235, 205, 60);
  drawShotPreview(shot, teamColor);
}

void drawShotPreview(Shot shot, color teamColor) {
  ArrayList<PVector> path = physics.predictPath(sheet.hackWorld(), shot, 800, DT);

  pushStyle();
  noFill();
  stroke(teamColor, 240);
  strokeWeight(3);
  beginShape();
  for (PVector p : path) {
    PVector s = worldToScreen(p);
    vertex(s.x, s.y);
  }
  endShape();
  popStyle();
}

void drawEndOverlay() {
  if (game.state != GameState.ENDED) return;
  pushStyle();
  fill(0, 200);
  noStroke();
  rect(0, ICE_H * 0.5 - 60, ICE_W, 120);

  textAlign(CENTER, CENTER);
  if (game.winningTeam < 0) {
    fill(230);
    textSize(28);
    text("Oavgjord (blank end)", ICE_W * 0.5, ICE_H * 0.5);
  } else {
    int   t  = game.winningTeam;
    color c  = t == TEAM_RED ? color(230, 80, 80) : color(230, 210, 80);
    String n = t == TEAM_RED ? "Rod" : "Gul";
    fill(c);
    textSize(34);
    text(n + " vinner", ICE_W * 0.5, ICE_H * 0.5 - 14);
    fill(240);
    textSize(20);
    text("med " + game.winningPoints + " poang", ICE_W * 0.5, ICE_H * 0.5 + 18);
  }
  popStyle();
}

// Solid backdrop behind the sidebar.
void drawSidebarBackdrop() {
  pushStyle();
  noStroke();
  fill(28);
  rect(ICE_W, 0, SIDEBAR_W, ICE_H);
  stroke(60);
  line(ICE_W, 0, ICE_W, ICE_H);
  popStyle();
}

void drawDebugMarkers() {
  PVector b = worldToScreen(house.BUTTON);
  PVector h = worldToScreen(sheet.hackWorld());

  pushStyle();
  noFill();
  strokeWeight(1);

  stroke(0, 255, 0, 200);
  ellipse(b.x, b.y, 10, 10);
  line(b.x - 8, b.y, b.x + 8, b.y);
  line(b.x, b.y - 8, b.x, b.y + 8);

  stroke(255, 200, 0, 200);
  ellipse(h.x, h.y, 10, 10);
  line(h.x - 8, h.y, h.x + 8, h.y);
  line(h.x, h.y - 8, h.x, h.y + 8);

  fill(0, 255, 0, 220);
  noStroke();
  textSize(11);
  text("tee", b.x + 8, b.y - 14);
  fill(255, 200, 0, 220);
  text("hack", h.x + 8, h.y - 14);
  popStyle();
}

// ----- Coordinate helpers ------------------------------------
PVector worldToScreen(float wx, float wy) {
  return sheet.worldToScreen(new PVector(wx, wy));
}

PVector worldToScreen(PVector w) {
  return sheet.worldToScreen(w);
}

float worldToScreen(float wLen) {
  return sheet.worldLenToScreen(wLen);
}
