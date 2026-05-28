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

MonteCarloAI aiPolicy;

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
  aiPolicy = new MonteCarloAI();
}

void draw() {
  background(20);
  physics.step(game.stones, DT);
  game.update();
  maybeAiShoot();
  ui.update(DT);

  sheet.drawSheet();
  if (game.state == GameState.AIMING) drawAimPreview(ui.intendedShot());
  for (Stone s : game.stones) s.draw();
  if (DEBUG) drawDebugMarkers();

  drawSidebarBackdrop();
  ui.draw();
  drawEndOverlay();
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
  if (game.state != GameState.AIMING || game.currentTeam != TEAM_YELLOW) return;

  game.fire(aiPolicy.chooseBestShot(game.stones));
}

// ----- Aim preview: forward-simulated trajectory in team color
void drawAimPreview(Shot shot) {
  ArrayList<PVector> path = physics.predictPath(sheet.hackWorld(), shot, 800, DT);

  color teamColor = (game.currentTeam == TEAM_RED)
      ? color(235, 70, 70)
      : color(235, 205, 60);

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
