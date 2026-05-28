// =============================================================
// ExpertShotDataset - CSV storage for human-labeled layouts + shots.
// Used by RECORD mode and (later) training heuristics.
// =============================================================

class ExpertShotEntry {
  float[][] slots;   // [TOTAL_STONES][4]  x,y,team,exists
  float   stonesLeft;
  int     lastTeam;
  float   curl;
  float   speed;
  float   angle;     // radians

  ExpertShotEntry() {
    slots = new float[TOTAL_STONES][4];
  }

  ArrayList<Stone> toLayout() {
    ArrayList<Stone> layout = new ArrayList<Stone>();
    for (int slot = 0; slot < TOTAL_STONES; slot++) {
      if (slots[slot][3] > 0.5) {
        Stone s = new Stone(slots[slot][0], slots[slot][1], (int) slots[slot][2]);
        s.hogPassed = true;
        layout.add(s);
      }
    }
    return layout;
  }

  String toCsvLine() {
    StringBuilder sb = new StringBuilder();
    for (int slot = 0; slot < TOTAL_STONES; slot++) {
      for (int f = 0; f < 4; f++) {
        if (sb.length() > 0) sb.append(',');
        sb.append(nf(slots[slot][f], 0, 4));
      }
    }
    sb.append(',').append(nf(stonesLeft, 0, 4));
    sb.append(',').append(lastTeam);
    sb.append(',').append(nf(curl, 0, 4));
    sb.append(',').append(nf(speed, 0, 4));
    sb.append(',').append(nf(angle, 0, 6));
    return sb.toString();
  }
}

class ExpertShotDataset {
  final String csvPath = "expert_shots.csv";
  final String csvHeader =
    "s0_x,s0_y,s0_team,s0_exists,"
  + "s1_x,s1_y,s1_team,s1_exists,"
  + "s2_x,s2_y,s2_team,s2_exists,"
  + "s3_x,s3_y,s3_team,s3_exists,"
  + "s4_x,s4_y,s4_team,s4_exists,"
  + "s5_x,s5_y,s5_team,s5_exists,"
  + "stones_left,last_team,curl,speed,angle_rad";

  ArrayList<ExpertShotEntry> entries = new ArrayList<ExpertShotEntry>();
  String lastSaveMessage = "";

  void fillSlotsFromLayout(ArrayList<Stone> layout, float[][] slots) {
    for (int slot = 0; slot < TOTAL_STONES; slot++) {
      if (slot < layout.size()) {
        Stone s = layout.get(slot);
        slots[slot][0] = s.pos.x;
        slots[slot][1] = s.pos.y;
        slots[slot][2] = s.team;
        slots[slot][3] = 1;
      } else {
        slots[slot][0] = 0;
        slots[slot][1] = 0;
        int throwTeam = (slot % 2 == 0) ? TEAM_RED : TEAM_YELLOW;
        slots[slot][2] = throwTeam;
        slots[slot][3] = -1;
      }
    }
  }

  ExpertShotEntry buildEntry(ArrayList<Stone> layout, float stonesLeft,
                             int lastTeam, Shot shot) {
    ExpertShotEntry e = new ExpertShotEntry();
    fillSlotsFromLayout(layout, e.slots);
    e.stonesLeft = stonesLeft;
    e.lastTeam   = lastTeam;
    e.curl       = shot.curl;
    e.speed      = shot.speed;
    e.angle      = shot.angle;
    return e;
  }

  ExpertShotEntry parseCsvLine(String line) {
    String[] p = split(line, ',');
    int need = TOTAL_STONES * 4 + 5;
    if (p.length < need) return null;
    ExpertShotEntry e = new ExpertShotEntry();
    int i = 0;
    for (int slot = 0; slot < TOTAL_STONES; slot++) {
      for (int f = 0; f < 4; f++) {
        e.slots[slot][f] = parseFloat(p[i++]);
      }
    }
    e.stonesLeft = parseFloat(p[i++]);
    e.lastTeam   = (int) parseFloat(p[i++]);
    e.curl       = parseFloat(p[i++]);
    e.speed      = parseFloat(p[i++]);
    e.angle      = parseFloat(p[i++]);
    return e;
  }

  void load() {
    entries.clear();
    String[] lines = loadTableLines();
    if (lines == null) return;
    for (String line : lines) {
      line = trim(line);
      if (line.length() == 0) continue;
      if (line.startsWith("s0_x")) continue;
      ExpertShotEntry e = parseCsvLine(line);
      if (e != null) entries.add(e);
    }
  }

  String[] loadTableLines() {
    String path = dataPath(csvPath);
    File f = new File(path);
    if (!f.exists()) return null;
    return loadStrings(path);
  }

  void append(ExpertShotEntry entry) {
    entries.add(entry);
    String path = dataPath(csvPath);
    String[] lines = loadStrings(path);
    String row = entry.toCsvLine();
    if (lines == null) {
      lines = new String[] { csvHeader, row };
    } else {
      String[] extended = new String[lines.length + 1];
      arrayCopy(lines, 0, extended, 0, lines.length);
      extended[lines.length] = row;
      lines = extended;
    }
    saveStrings(path, lines);
    lastSaveMessage = "Sparat #" + entries.size();
  }

  ExpertShotEntry randomEntry() {
    if (entries.isEmpty()) return null;
    return entries.get((int) random(entries.size()));
  }

  ArrayList<Stone> randomLayout() {
    ExpertShotEntry e = randomEntry();
    if (e == null) return null;
    return e.toLayout();
  }

  int count() {
    return entries.size();
  }
}

// Self-contained record session: random layouts, direct fire, CSV export.
class RecordSession {
  final float recordStonesLeft = 1;
  final int   recordLastTeam   = TEAM_RED;

  ExpertShotDataset dataset;
  RandomState       randomState = new RandomState();
  ArrayList<Stone>  stones = new ArrayList<Stone>();
  ArrayList<Stone>  layoutSnapshot = new ArrayList<Stone>();
  boolean           simulating = false;
  int               dragIndex = -1;
  String            status = "";

  final float placeMinX;
  final float placeMaxX;
  final float placeMinY;
  final float placeMaxY;
  final float pickRadiusScreen;

  RecordSession(ExpertShotDataset dataset) {
    this.dataset = dataset;
    float r = STONE_RADIUS;
    placeMinX = r;
    placeMaxX = sheet.SHEET_WIDTH_FT - r;
    placeMinY = sheet.hogY + r;
    placeMaxY = sheet.backFarY - r;
    pickRadiusScreen = sheet.worldLenToScreen(r * 2.2f);
    dataset.load();
    newRandomLayout();
  }

  boolean canEditLayout() {
    return !simulating;
  }

  int findStoneAtScreen(float mx, float my) {
    if (mx < 0 || mx > ICE_W || my < 0 || my > ICE_H) return -1;
    int best = -1;
    float bestDist = pickRadiusScreen;
    for (int i = 0; i < layoutSnapshot.size(); i++) {
      PVector sc = worldToScreen(layoutSnapshot.get(i).pos);
      float d = dist(mx, my, sc.x, sc.y);
      if (d <= bestDist) {
        bestDist = d;
        best = i;
      }
    }
    return best;
  }

  boolean onMousePressed(float mx, float my) {
    if (!canEditLayout()) return false;
    int idx = findStoneAtScreen(mx, my);
    if (idx < 0) return false;
    dragIndex = idx;
    status = "Drar sten — släpp för att uppdatera AI-f\u00f6rslag";
    return true;
  }

  boolean onMouseDragged(float mx, float my) {
    if (dragIndex < 0) return false;
    PVector w = sheet.screenToWorld(mx, my);
    Stone s = layoutSnapshot.get(dragIndex);
    s.pos.x = constrain(w.x, placeMinX, placeMaxX);
    s.pos.y = constrain(w.y, placeMinY, placeMaxY);
    stones = copyStones(layoutSnapshot);
    return true;
  }

  boolean onMouseReleased() {
    if (dragIndex < 0) return false;
    dragIndex = -1;
    stones = copyStones(layoutSnapshot);
    status = "Layout uppdaterad";
    return true;
  }

  ArrayList<Stone> copyStones(ArrayList<Stone> src) {
    ArrayList<Stone> copy = new ArrayList<Stone>();
    for (Stone s : src) {
      Stone c = new Stone(s.pos.x, s.pos.y, s.team);
      c.hogPassed = s.hogPassed;
      copy.add(c);
    }
    return copy;
  }

  void newRandomLayout() {
    layoutSnapshot = new ArrayList<Stone>();
    randomState.randomize(layoutSnapshot, STONES_PER_TEAM);
    restoreLayout();
    status = "Ny layout (" + layoutSnapshot.size() + " stenar)";
  }

  void restoreLayout() {
    stones = copyStones(layoutSnapshot);
    simulating = false;
    dragIndex = -1;
  }

  void shoot(Shot shot) {
    if (simulating) return;
    stones = copyStones(layoutSnapshot);
    PVector h = sheet.hackWorld();
    Stone fired = new Stone(h.x, h.y, TEAM_YELLOW);
    fired.curl = constrain(shot.curl, -1, 1);
    fired.vel.set(sin(shot.angle) * shot.speed, cos(shot.angle) * shot.speed);
    stones.add(fired);
    simulating = true;
    status = "Simulerar...";
  }

  void update(float dt) {
    if (!simulating) return;
    physics.step(stones, dt);
    boolean moving = false;
    for (Stone s : stones) {
      if (s.isMoving()) { moving = true; break; }
    }
    if (!moving) {
      simulating = false;
      restoreLayout();
      status = "Klar — justera, tryck N\u00e4sta eller Ny state";
    }
  }

  boolean saveCurrentShot(Shot intended) {
    ExpertShotEntry row = dataset.buildEntry(
      layoutSnapshot, recordStonesLeft, recordLastTeam, intended);
    dataset.append(row);
    status = dataset.lastSaveMessage;
    newRandomLayout();
    return true;
  }

  boolean canShoot() {
    return !simulating;
  }

  boolean canSave() {
    return !simulating;
  }
}
