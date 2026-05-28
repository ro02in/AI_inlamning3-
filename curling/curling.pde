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
enum AppMode { PLAY, TRAINING, TEST, RECORD }

AppMode appMode = AppMode.PLAY;
boolean trainPenaltyEnabled = false;
boolean trainPinEnabled     = true;
boolean trainScoreEnabled   = false;

// Model-type and algorithm selectors (toggled from UI).
boolean useEnsemble  = true;   // true = ExpertEnsemble; false = single NeuralPolicy
boolean useGradients = false;  // true = PolicyGradient; false = PolicySearch

ExpertEnsemble ensemble;
GradientEnsemble gradientEnsemble;
// Single-model policy search trainer (useEnsemble=false, useGradients=false).
PolicySearchTraining singleTrainer;
// Policy gradient trainer (useGradients=true, useEnsemble=false).
PolicyGradientTraining pgTrainer;

ScoreHeuristic scoreHeuristic;
CloseToButtonHeuristic pinHeuristic;
PenaltyHeuristic penaltyHeuristic;
ExpertShotHeuristic expertHeuristic;
TrainingPreview    trainingPreview;
PolicyDiagnostics  policyDiagnostics;
ModelStorage       modelStorage;

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

ExpertShotDataset expertShots;
RecordSession     recordSession;
String            recordStatus = "";

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

  ensemble         = new ExpertEnsemble();
  gradientEnsemble = new GradientEnsemble();
  singleTrainer    = new PolicySearchTraining();
  pgTrainer        = new PolicyGradientTraining();
  scoreHeuristic   = new ScoreHeuristic();
  pinHeuristic     = new CloseToButtonHeuristic();
  penaltyHeuristic = new PenaltyHeuristic();
  expertHeuristic  = new ExpertShotHeuristic();
  trainingPreview  = new TrainingPreview();
  policyDiagnostics = new PolicyDiagnostics();
  modelStorage      = new ModelStorage();
  expertShots      = new ExpertShotDataset();
}

// Returns a representative NeuralPolicy from whichever model is currently active.
NeuralPolicy activePolicy() {
  if (useGradients) {
    return useEnsemble ? gradientEnsemble.anyPolicy() : pgTrainer.policy;
  }
  if (useEnsemble) return ensemble.anyPolicy();
  return singleTrainer.current;
}

// Dispatch inference: pick the best shot from the active model.
Shot bestShotActive(ArrayList<Stone> layout, int stonesLeft, int lastTeam,
                    Heuristic scorer, boolean logScores) {
  if (useGradients) {
    if (useEnsemble) {
      return gradientEnsemble.bestShot(layout, stonesLeft, lastTeam, scorer, logScores);
    }
    float[] state = pgTrainer.policy.convertState(layout, stonesLeft, lastTeam);
    Shot shot = pgTrainer.policy.predictMean(state);
    if (logScores) {
      policyDiagnostics.logGradientInference(pgTrainer.policy, layout,
                                             stonesLeft, lastTeam, scorer);
    }
    return shot;
  }
  if (useEnsemble) {
    return ensemble.bestShot(layout, stonesLeft, lastTeam, scorer, logScores);
  }
  // Single policy-search model: predict directly (no ensemble competition).
  float[] state = singleTrainer.current.convertState(layout, stonesLeft, lastTeam);
  return singleTrainer.current.predict(state);
}

// Reset whichever trainer is currently selected.
void resetActiveTrainer() {
  if (useGradients) {
    if (useEnsemble) gradientEnsemble.reset();
    else pgTrainer.reset();
  } else if (useEnsemble) {
    ensemble.reset();
  } else {
    singleTrainer.reset();
  }
}

// Switch use-ensemble flag and reset the newly selected trainer.
void setUseEnsemble(boolean val) {
  if (useEnsemble == val) return;
  useEnsemble = val;
  trainingDone = 0;
  trainingStatus = "Ny modell";
}

// Switch algorithm flag and reset the newly selected trainer.
void setUseGradients(boolean val) {
  if (useGradients == val) return;
  useGradients = val;
  trainingDone = 0;
  trainingStatus = "Ny modell";
}

void applyRecordPolicySliders() {
  if (recordSession == null) return;
  NeuralPolicy p = activePolicy();
  float[] state = p.convertState(recordSession.layoutSnapshot, 1, TEAM_RED);
  ui.setSlidersFromShot(p.predict(state));
}

void recordNextShot() {
  if (recordSession == null || !recordSession.canSave()) return;
  recordSession.saveCurrentShot(ui.intendedShot());
  applyRecordPolicySliders();
}

void recordNewState() {
  if (recordSession == null) return;
  recordSession.newRandomLayout();
  applyRecordPolicySliders();
}

void startRecordMode() {
  if (trainingActive) return;
  if (appMode == AppMode.TEST) stopAiTest();
  appMode = AppMode.RECORD;
  recordSession = new RecordSession(expertShots);
  recordStatus = "Expert-lage";
  ui.onRecordEnter();
  applyRecordPolicySliders();
}

void stopRecordMode() {
  if (appMode != AppMode.RECORD) return;
  appMode = AppMode.PLAY;
  recordSession = null;
  recordStatus = "";
  ui.onRecordExit();
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
      ArrayList<Heuristic> heuristics = activeTrainingHeuristics();
      if (useGradients) {
        if (useEnsemble) {
          gradientEnsemble.trainAll(heuristics, ui.shotsPerPrediction);
        } else {
          pgTrainer.shotsPerUpdate = ui.shotsPerPrediction;
          pgTrainer.updateStep(heuristics);
        }
      } else if (useEnsemble) {
        ensemble.trainAll(heuristics, ui.shotsPerPrediction,
                          expertHeuristic, ui.expertShotsPerPrediction, expertShots);
      } else {
        singleTrainer.comparePolicies(heuristics, ui.shotsPerPrediction,
                                      expertHeuristic, ui.expertShotsPerPrediction, expertShots);
      }
      trainingDone++;
      trainingPreview.recordSnapshot(trainingDone, activePolicy());
      if (trainingDone >= trainingTarget) {
        trainingActive = false;
        appMode = AppMode.PLAY;
        trainingStatus = "Klar! (" + trainingDone + " steg)";
        trainingPreview.reset();
      }
    }
    trainingPreview.update(DT);
  } else if (appMode == AppMode.TEST) {
    updateAiTest();
    ui.update(DT);
  } else if (appMode == AppMode.RECORD) {
    recordSession.update(DT);
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
  } else if (appMode == AppMode.RECORD) {
    if (recordSession != null) {
      drawAimPreview(ui.intendedShot());
      for (Stone s : recordSession.stones) s.draw();
      drawRecordDragHighlight();
      drawRecordOverlay();
    }
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
  trainingStatus = "Avbruten (" + trainingDone + " jamf.)";
  trainingPreview.reset();
}

void resetTrainingModel() {
  if (trainingActive) return;
  resetActiveTrainer();
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
  selectInput("Ladda modell", "onLoadModelSelected");
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

  Heuristic scorer = activeTrainingHeuristics().isEmpty()
                   ? pinHeuristic
                   : activeTrainingHeuristics().get(0);
  aiTestLastShot = bestShotActive(aiTestStones, 1, TEAM_RED, scorer, true);

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

void drawRecordDragHighlight() {
  if (recordSession == null || recordSession.dragIndex < 0) return;
  Stone s = recordSession.layoutSnapshot.get(recordSession.dragIndex);
  PVector sc = worldToScreen(s.pos);
  float d = worldToScreen(STONE_RADIUS * 2.4f);
  pushStyle();
  noFill();
  stroke(255, 220, 80, 230);
  strokeWeight(3);
  ellipse(sc.x, sc.y, d, d);
  popStyle();
}

void drawRecordOverlay() {
  pushStyle();
  fill(0, 160);
  noStroke();
  rect(0, 8, ICE_W, 52);
  fill(240);
  textAlign(LEFT, TOP);
  textSize(13);
  text("Expert — dra stenar, Gul skjuter", 12, 14);
  fill(200);
  textSize(11);
  String msg = recordSession != null ? recordSession.status : "";
  if (recordStatus.length() > 0) msg = recordStatus + "  |  " + msg;
  text(msg, 12, 32);
  if (expertShots != null) {
    textAlign(RIGHT, TOP);
    fill(180);
    text(expertShots.count() + " sparade i " + expertShots.csvPath,
         ICE_W - 12, 14);
  }
  popStyle();
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
  if (handleRecordMousePressed(mouseX, mouseY)) return;
  ui.onMousePressed(mouseX, mouseY);
}

void mouseDragged() {
  if (handleRecordMouseDragged(mouseX, mouseY)) return;
  ui.onMouseDragged(mouseX, mouseY);
}

void mouseReleased() {
  if (handleRecordMouseReleased()) return;
  ui.onMouseReleased(mouseX, mouseY);
}

boolean handleRecordMousePressed(float mx, float my) {
  if (appMode != AppMode.RECORD || recordSession == null) return false;
  if (mx >= ICE_W) return false;
  if (recordSession.onMousePressed(mx, my)) return true;
  return false;
}

boolean handleRecordMouseDragged(float mx, float my) {
  if (appMode != AppMode.RECORD || recordSession == null) return false;
  return recordSession.onMouseDragged(mx, my);
}

boolean handleRecordMouseReleased() {
  if (appMode != AppMode.RECORD || recordSession == null) return false;
  if (recordSession.onMouseReleased()) {
    applyRecordPolicySliders();
    return true;
  }
  return false;
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

void maybeAiShoot() {
  if (appMode != AppMode.PLAY) return;
  if (game.state != GameState.AIMING || game.currentTeam != TEAM_YELLOW) return;

  int lastTeam = TEAM_RED;
  if (game.stones.size() > 0) {
    lastTeam = game.stones.get(game.stones.size() - 1).team;
  }

  Heuristic scorer = activeTrainingHeuristics().isEmpty()
                   ? pinHeuristic
                   : activeTrainingHeuristics().get(0);
  Shot shot = bestShotActive(game.stones, game.stonesRemaining(TEAM_YELLOW),
                             lastTeam, scorer, false);
  game.fire(shot);
}

// ----- Aim preview: forward-simulated trajectory in team color
void drawAimPreview(Shot shot) {
  int team = (appMode == AppMode.RECORD) ? TEAM_YELLOW : game.currentTeam;
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
