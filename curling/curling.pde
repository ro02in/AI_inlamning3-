// =============================================================
// Curling - main sketch
// =============================================================
// World coordinates use the map image's pixel space directly:
//   - Image is 1080 wide x 1920 tall (portrait).
//   - House (button) center in world coords: (540, 360).
//   - Hack / release point: (540, 1820).
//   - Stones travel from the hack toward the house (negative y).
//
// Rendering uses WORLD_TO_SCREEN to scale world units to the
// on-screen window. All physics happens in world-space; only the
// draw() side multiplies by WORLD_TO_SCREEN.
// =============================================================

// ----- World constants ---------------------------------------
final int   WORLD_W = 1080;
final int   WORLD_H = 1920;

// World->screen scale for the ice viewport.
final float WORLD_TO_SCREEN = 0.5;

// On-screen layout: ice viewport on the left, UI sidebar on the right.
final int   ICE_W     = (int) (WORLD_W * WORLD_TO_SCREEN);  // 540
final int   ICE_H     = (int) (WORLD_H * WORLD_TO_SCREEN);  // 960
final int   SIDEBAR_W = 220;
final int   WIN_W     = ICE_W + SIDEBAR_W;                  // 760
final int   WIN_H     = ICE_H;                              // 960

// ----- Assets ------------------------------------------------
PImage mapImg;
PImage stoneRedImg;
PImage stoneYellowImg;

// ----- Game world -------------------------------------------
Game    game;
House   house;
Physics physics;
UI      ui;

// Fixed simulation timestep (seconds). Drives Physics.step.
final float DT = 1.0 / 60.0;

// Toggle with 'd' to show alignment crosses for the button and hack.
boolean DEBUG = false;

// ----- Setup / draw ------------------------------------------
void settings() {
  size(WIN_W, WIN_H);
}

void setup() {
  frameRate(60);
  imageMode(CENTER);
  textAlign(LEFT, TOP);

  mapImg         = loadImage("curling-map.png");
  stoneRedImg    = loadImage("curling-stone-red.png");
  stoneYellowImg = loadImage("curling-stone-yellow.png");

  physics = new Physics();
  house   = new House();
  ui      = new UI();        // before Game so Game.reset can call ui.onTurnStart
  game    = new Game();
}

void draw() {
  background(20);
  physics.step(game.stones, DT);
  game.update();
  ui.update(DT);

  drawMap();
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

// ----- Aim preview: forward-simulated trajectory in team color
void drawAimPreview(Shot shot) {
  ArrayList<PVector> path = physics.predictPath(game.hack, shot, 600, DT);

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

// Draw the map fitted to the ice viewport (left side of the window).
void drawMap() {
  pushStyle();
  imageMode(CORNER);
  image(mapImg, 0, 0, ICE_W, ICE_H);
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

// Small overlay so we can visually confirm the button location.
void drawDebugMarkers() {
  PVector b = worldToScreen(540, 360);   // button
  PVector h = worldToScreen(540, 1820);  // hack

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
  text("button (540, 360)", b.x + 8, b.y - 14);
  fill(255, 200, 0, 220);
  text("hack (540, 1820)", h.x + 8, h.y - 14);
  popStyle();
}

// ----- Coordinate helpers ------------------------------------
PVector worldToScreen(float wx, float wy) {
  return new PVector(wx * WORLD_TO_SCREEN, wy * WORLD_TO_SCREEN);
}

PVector worldToScreen(PVector w) {
  return worldToScreen(w.x, w.y);
}

float worldToScreen(float wLen) {
  return wLen * WORLD_TO_SCREEN;
}
