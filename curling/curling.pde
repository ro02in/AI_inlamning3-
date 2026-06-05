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

// Toggle with 'd' to show hack debug marker.
boolean DEBUG = false;

// ----- AI training / test ------------------------------------
enum AppMode { PLAY, TRAINING, TEST }

AppMode appMode = AppMode.PLAY;

// ----- Backend training constants. Tuneable. --------
final int   SHOTS_PER_ROUND    = 50;    // shots sampled per expert per update
final int   CURRICULUM_STEP    = 500;   // training steps before adding another stone depth
final int   PLAY_CANDIDATE_EXPERTS = 3; // current expert shots tested during NN play
final int   NEXT_UTILITY_EXPERTS   = 3; // next-shot experts used for expected utility
final int   SIMS_PER_EXPERT_SHOT  = 3;  // repeated simulations per expert shot
final int   SIMS_PER_NEXT_EXPERT  = 3;  // repeated simulations per next-shot expert
final float SELECTOR_TEMP      = 1.0f;  // temperature for reward->target softmax
final float SELECTOR_LR        = 0.001f;// selector network learning rate
final float SELECTOR_TYPE_REWARD_BLEND = 0.30f; // finalScore + blend * type heuristic

GradientEnsemble gradientEnsemble;
ShotTypeSelector shotSelector;
ModelStorage       modelStorage;

int  trainingTarget = 100;
int  trainingDone   = 0;
boolean trainingActive = false;
String trainingStatus = "";

// Active AI type for Yellow: 1 = MC AI (random sampling), 2 = NN (neural network)
int activeAiType = 2;

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

  gradientEnsemble  = new GradientEnsemble();
  shotSelector      = new ShotTypeSelector(gradientEnsemble);
  modelStorage      = new ModelStorage();

}

// Compute the curriculum depth cap from training progress.
int curriculumDepthCap() {
  return min(TOTAL_STONES, 1 + trainingDone / CURRICULUM_STEP);
}

// Dispatch inference using the selector to choose one expert.
Shot bestShotActive(ArrayList<Stone> layout, int stonesLeft, int lastTeam,
                    int remainingShotsAfterCurrent,
                    boolean logScores) {
  return gradientEnsemble.bestShot(layout, stonesLeft, lastTeam,
                                   remainingShotsAfterCurrent,
                                   shotSelector, logScores, false);
}

void resetActiveTrainer() {
  gradientEnsemble.reset();
  shotSelector.reset();
  trainingDone = 0;
  trainingStatus = "Ny modell";
}

void draw() {
  background(20);

  if (appMode == AppMode.TRAINING) {
    if (trainingActive) {
      int depthCap = curriculumDepthCap();
      gradientEnsemble.trainAll(depthCap, SHOTS_PER_ROUND);
      shotSelector.updateStep(depthCap);
      trainingDone++;
      if (trainingDone >= trainingTarget) {
        trainingActive = false;
        appMode = AppMode.PLAY;
        trainingStatus = "Klar! (" + trainingDone + " steg)";
      }
    }
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
}
void startTraining(int comparisons) {
  if (trainingActive) return;
  trainingTarget = trainingDone + comparisons;
  trainingActive = true;
  trainingStatus = "";
  appMode        = AppMode.TRAINING;
}

void cancelTraining() {
  if (!trainingActive) return;
  trainingActive = false;
  appMode        = AppMode.PLAY;
  trainingStatus = "Avbruten (" + trainingDone + " jamf.)";
}

void resetTrainingModel() {
  if (trainingActive) return;
  gradientEnsemble.reset();
  shotSelector.reset();
  trainingDone   = 0;
  trainingTarget = 0;
  trainingStatus = "Ny modell";
}

void promptSaveModel() {
  if (trainingActive || appMode == AppMode.TRAINING) return;
  selectOutput("Spara modell", "onSaveModelSelected",
               new File(modelStorage.defaultSavePath()));
}

void onSaveModelSelected(File selection) {
  if (selection == null) return;
  String path = selection.getAbsolutePath();
  if (!path.toLowerCase().endsWith(".curlmodel")) {
    path += ".curlmodel";
  }
  modelStorage.saveActiveModel(path);
  trainingStatus = modelStorage.lastMessage;
}

void promptLoadModel() {
  if (trainingActive || appMode == AppMode.TRAINING) return;
  selectInput("Ladda modell", "onLoadModelSelected", modelStorage.modelDir());
}

void onLoadModelSelected(File selection) {
  if (selection == null) return;
  if (modelStorage.loadActiveModel(selection.getAbsolutePath())) {
    trainingStatus = modelStorage.lastMessage;
  } else {
    trainingStatus = modelStorage.lastMessage;
  }
}

void startAiTest() {
  appMode = AppMode.TEST;
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

  aiTestLastShot = bestShotActive(aiTestStones, 1, TEAM_RED, 0, true);

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
void mousePressed() {
  ui.onMousePressed(mouseX, mouseY);
}

void mouseDragged() {
  ui.onMouseDragged(mouseX, mouseY);
}

void mouseReleased() {
  ui.onMouseReleased();
}
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

ArrayList<Stone> copyLayout(ArrayList<Stone> layout) {
  ArrayList<Stone> copy = new ArrayList<Stone>();
  for (Stone s : layout) {
    Stone c = new Stone(s.pos.x, s.pos.y, s.team);
    c.hogPassed = s.hogPassed;
    copy.add(c);
  }
  return copy;
}

Shot mcBestShot(ArrayList<Stone> layout) {
  int N = 80;
  Shot best = null;
  float bestScore = Float.NEGATIVE_INFINITY;
  ScoreResult before = house.scoreEnd(layout);

  for (int i = 0; i < N; i++) {
    float curl  = random(-1, 1);
    float speed = random(0.35, 0.95) * UI.SPEED_MAX;
    float angle = random(radians(-12), radians(12));

    ArrayList<Stone> simStones = copyLayout(layout);
    PVector h = sheet.hackWorld();
    Stone fired = new Stone(h.x, h.y, TEAM_YELLOW);
    fired.curl = constrain(curl, -1, 1);
    fired.vel.set(sin(angle) * speed, cos(angle) * speed);
    simStones.add(fired);

    for (int step = 0; step < 100000; step++) {
      physics.step(simStones, DT);
      boolean anyMoving = false;
      for (Stone s : simStones) if (s.isMoving()) { anyMoving = true; break; }
      if (!anyMoving) break;
    }

    ScoreResult after = house.scoreEnd(simStones);
    float score = after.diffFrom(before);
    if (score > bestScore) {
      bestScore = score;
      best = new Shot(curl, speed, angle);
    }
  }
  return best != null ? best : new Shot(0, 0.6 * UI.SPEED_MAX, 0);
}

void maybeAiShoot() {
  if (appMode != AppMode.PLAY) return;
  if (game.state != GameState.AIMING || game.currentTeam != TEAM_YELLOW) return;

  int lastTeam = TEAM_RED;
  if (game.stones.size() > 0) {
    lastTeam = game.stones.get(game.stones.size() - 1).team;
  }

  Shot shot;
  if (activeAiType == 1) {
    shot = mcBestShot(game.stones);
  } else {
    int remainingShotsAfterCurrent = max(0, TOTAL_STONES - (game.stonesThrown + 1));
    shot = bestShotActive(game.stones, game.stonesRemaining(TEAM_YELLOW),
                          lastTeam, remainingShotsAfterCurrent, true);
  }
  game.fire(shot);
}

// ----- Aim preview: forward-simulated trajectory in team color
void drawAimPreview(Shot shot) {
  int team = game.currentTeam;
  color teamColor = (team == TEAM_RED)
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
  PVector h = worldToScreen(sheet.hackWorld());

  pushStyle();
  noFill();
  strokeWeight(1);

  stroke(255, 200, 0, 200);
  ellipse(h.x, h.y, 10, 10);
  line(h.x - 8, h.y, h.x + 8, h.y);
  line(h.x, h.y - 8, h.x, h.y + 8);

  fill(255, 200, 0, 220);
  noStroke();
  textSize(11);
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
