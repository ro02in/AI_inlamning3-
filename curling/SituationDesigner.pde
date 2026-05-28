// UI for placing TOTAL_STONES-1 stones and saving them to data/situations/
class SituationDesigner {
  static final int MAX_STONES = TOTAL_STONES - 1;

  SituationLibrary library;
  ArrayList<Stone> stones = new ArrayList<Stone>();
  int    placeTeam      = TEAM_RED;
  int    selectedIndex  = -1;
  int    dragIndex      = -1;
  int    browseIndex    = -1;
  String editFilename   = "";
  String editName       = "ny situation";

  Button exitBtn, saveBtn, newBtn, prevBtn, nextBtn;
  Button teamBtn, deleteBtn, clearBtn;

  final float PANEL_TOP = 178;
  final float LIBRARY_TOP = PANEL_TOP + 234;
  final float LIBRARY_ROW_H = 14;
  final float LIBRARY_LEFT = ICE_W + 12;
  final float LIBRARY_WIDTH = SIDEBAR_W - 24;

  int hoverLibraryIndex = -1;

  SituationDesigner(SituationLibrary library) {
    this.library = library;
    float bx = ICE_W + 12;
    float bw = SIDEBAR_W - 24;
    float bh = 32;
    float y  = PANEL_TOP;

    exitBtn   = new Button("Klar",     bx, y, bw, bh); y += bh + 6;
    saveBtn   = new Button("Spara",     bx, y, bw, bh); y += bh + 6;
    newBtn    = new Button("Ny",        bx, y, bw * 0.48f, bh);
    prevBtn   = new Button("<",         bx + bw * 0.52f, y, bw * 0.22f, bh);
    nextBtn   = new Button(">",         bx + bw * 0.76f, y, bw * 0.22f, bh); y += bh + 6;
    teamBtn   = new Button("Lag: Rod",  bx, y, bw, bh); y += bh + 6;
    deleteBtn = new Button("Ta bort",   bx, y, bw * 0.48f, bh);
    clearBtn  = new Button("Rensa",     bx + bw * 0.52f, y, bw * 0.48f, bh);
  }

  void enter() {
    stones.clear();
    selectedIndex = -1;
    dragIndex     = -1;
    browseIndex   = -1;
    editFilename  = nextDefaultFilename();
    editName      = "ny situation";
    library.reloadFromDisk();
  }

  void exit() {
    stones.clear();
    selectedIndex = -1;
    dragIndex     = -1;
  }

  String nextDefaultFilename() {
    int n = 1;
    while (true) {
      String fn = "situation_" + (n < 10 ? "0" : "") + n + ".sit";
      boolean taken = false;
      for (SavedSituation s : library.situations) {
        if (s.filename.equals(fn)) { taken = true; break; }
      }
      if (!taken) return fn;
      n++;
    }
  }

  void loadBrowse(int index) {
    if (index < 0 || index >= library.count()) return;
    SavedSituation sit = library.situations.get(index);
    SavedSituation fresh = library.loadFile(sit.filename);
    if (fresh == null) return;

    browseIndex = index;
    stones.clear();
    for (Stone s : fresh.stones) {
      Stone c = new Stone(s.pos.x, s.pos.y, s.team);
      c.hogPassed = s.hogPassed;
      stones.add(c);
    }
    editFilename = fresh.filename;
    editName     = fresh.name;
    selectedIndex = -1;
  }

  void drawIce() {
    for (int i = 0; i < stones.size(); i++) {
      if (i == selectedIndex) {
        pushStyle();
        PVector s = worldToScreen(stones.get(i).pos);
        noFill();
        stroke(255, 220, 80);
        strokeWeight(2);
        ellipse(s.x, s.y, worldToScreen(STONE_RADIUS * 2.6f), worldToScreen(STONE_RADIUS * 2.6f));
        popStyle();
      }
      stones.get(i).draw();
    }

    pushStyle();
    fill(0, 150);
    noStroke();
    rect(0, 0, ICE_W, 44);
    fill(240);
    textAlign(LEFT, TOP);
    textSize(13);
    text("Situationsdesigner  (" + stones.size() + "/" + MAX_STONES + " stenar)", 12, 8);
    textSize(11);
    fill(200);
    text("Klicka isen: placera  |  dra sten  |  hogerklikk: ta bort", 12, 26);
    popStyle();
  }

  void drawSidebar() {
    pushStyle();
    float left = ICE_W + 16;
    fill(150);
    textAlign(LEFT, TOP);
    textSize(10);
    text("SITUATION", left, 100);

    textSize(11);
    fill(200);
    text(editName, left, 140);
    text(editFilename, left, 154);
    text("Sparade: " + library.count(), left, 168);
    popStyle();

    exitBtn.draw();
    saveBtn.draw();
    newBtn.draw();
    prevBtn.draw();
    nextBtn.draw();
    teamBtn.label = placeTeam == TEAM_RED ? "Lag: Rod" : "Lag: Gul";
    teamBtn.draw();
    deleteBtn.draw();
    clearBtn.draw();

    float y = PANEL_TOP + 220;
    pushStyle();
    fill(180);
    textAlign(LEFT, TOP);
    textSize(10);
    text("Bibliotek (klicka for att redigera):", ICE_W + 16, y);
    y = LIBRARY_TOP;
    textSize(11);
    for (int i = 0; i < library.count(); i++) {
      SavedSituation s = library.situations.get(i);
      boolean active = i == browseIndex;
      boolean hover  = i == hoverLibraryIndex;
      if (active) {
        fill(40, 90, 55);
        noStroke();
        rect(LIBRARY_LEFT, y - 1, LIBRARY_WIDTH, LIBRARY_ROW_H, 3);
        fill(170, 240, 185);
      } else if (hover) {
        fill(55, 55, 65);
        noStroke();
        rect(LIBRARY_LEFT, y - 1, LIBRARY_WIDTH, LIBRARY_ROW_H, 3);
        fill(220);
      } else {
        fill(160);
      }
      text((i + 1) + ". " + s.name, ICE_W + 16, y);
      y += LIBRARY_ROW_H;
      if (y > ICE_H - 20) {
        fill(160);
        text("...", ICE_W + 16, y);
        break;
      }
    }
    popStyle();
  }

  void updateHover(float mx, float my) {
    exitBtn.hover = exitBtn.hits(mx, my);
    saveBtn.hover = saveBtn.hits(mx, my);
    newBtn.hover  = newBtn.hits(mx, my);
    prevBtn.hover = prevBtn.hits(mx, my);
    nextBtn.hover = nextBtn.hits(mx, my);
    teamBtn.hover = teamBtn.hits(mx, my);
    deleteBtn.hover = deleteBtn.hits(mx, my);
    clearBtn.hover = clearBtn.hits(mx, my);
    hoverLibraryIndex = libraryIndexAt(mx, my);
  }

  void onMousePressed(float mx, float my) {
    if (mx >= ICE_W) {
      handleSidebarClick(mx, my);
      return;
    }
    if (mouseButton == RIGHT) {
      int hit = stoneAt(mx, my);
      if (hit >= 0) removeStone(hit);
      return;
    }
    int hit = stoneAt(mx, my);
    if (hit >= 0) {
      selectedIndex = hit;
      dragIndex     = hit;
      return;
    }
    if (stones.size() < MAX_STONES) {
      PVector w = sheet.screenToWorld(mx, my);
      if (validPlacement(w, -1)) {
        Stone s = new Stone(w.x, w.y, placeTeam);
        s.hogPassed = true;
        stones.add(s);
        selectedIndex = stones.size() - 1;
      }
    }
  }

  void onMouseDragged(float mx, float my) {
    if (dragIndex < 0 || dragIndex >= stones.size()) return;
    if (mx >= ICE_W) return;
    PVector w = sheet.screenToWorld(mx, my);
    if (validPlacement(w, dragIndex)) {
      stones.get(dragIndex).pos.set(w);
    }
  }

  void onMouseReleased(float mx, float my) {
    dragIndex = -1;
  }

  void handleSidebarClick(float mx, float my) {
    if (exitBtn.hits(mx, my))   { stopSituationDesigner(); return; }
    if (saveBtn.hits(mx, my))   { saveCurrent(); return; }
    if (newBtn.hits(mx, my))    { newSituation(); return; }
    if (prevBtn.hits(mx, my))   { browse(-1); return; }
    if (nextBtn.hits(mx, my))   { browse(+1); return; }
    if (teamBtn.hits(mx, my))   { placeTeam = placeTeam == TEAM_RED ? TEAM_YELLOW : TEAM_RED; return; }
    if (deleteBtn.hits(mx, my)) { if (selectedIndex >= 0) removeStone(selectedIndex); return; }
    if (clearBtn.hits(mx, my))  { stones.clear(); selectedIndex = -1; return; }

    int libHit = libraryIndexAt(mx, my);
    if (libHit >= 0) {
      String fn = library.situations.get(libHit).filename;
      library.reloadFromDisk();
      for (int i = 0; i < library.count(); i++) {
        if (library.situations.get(i).filename.equals(fn)) {
          loadBrowse(i);
          break;
        }
      }
    }
  }

  int libraryIndexAt(float mx, float my) {
    if (mx < LIBRARY_LEFT || mx > LIBRARY_LEFT + LIBRARY_WIDTH) return -1;
    if (my < LIBRARY_TOP || my > ICE_H - 20) return -1;

    int index = (int) ((my - LIBRARY_TOP) / LIBRARY_ROW_H);
    if (index < 0 || index >= library.count()) return -1;
    return index;
  }

  void saveCurrent() {
    if (stones.isEmpty()) return;
    if (editFilename.length() == 0) editFilename = nextDefaultFilename();
    library.save(editFilename, editName, stones);
    browseIndex = max(0, library.count() - 1);
    for (int i = 0; i < library.count(); i++) {
      if (library.situations.get(i).filename.equals(editFilename)) {
        browseIndex = i;
        break;
      }
    }
  }

  void newSituation() {
    stones.clear();
    selectedIndex = -1;
    browseIndex   = -1;
    editFilename  = nextDefaultFilename();
    editName      = editFilename.replace(".sit", "").replace("_", " ");
  }

  void browse(int delta) {
    if (library.count() == 0) return;
    if (browseIndex < 0) browseIndex = 0;
    else browseIndex = (browseIndex + delta + library.count()) % library.count();
    loadBrowse(browseIndex);
  }

  void removeStone(int index) {
    stones.remove(index);
    if (selectedIndex >= stones.size()) selectedIndex = stones.size() - 1;
    if (selectedIndex == index) selectedIndex = -1;
    dragIndex = -1;
  }

  int stoneAt(float mx, float my) {
    PVector w = sheet.screenToWorld(mx, my);
    float hitR = STONE_RADIUS * 1.2f;
    for (int i = stones.size() - 1; i >= 0; i--) {
      if (PVector.dist(w, stones.get(i).pos) <= hitR) return i;
    }
    return -1;
  }

  boolean validPlacement(PVector p, int ignoreIndex) {
    float r = STONE_RADIUS;
    if (p.x - r < 0 || p.x + r > sheet.SHEET_WIDTH_FT) return false;
    if (p.y + r < sheet.hogY || p.y - r > sheet.backFarY) return false;
    for (int i = 0; i < stones.size(); i++) {
      if (i == ignoreIndex) continue;
      if (PVector.dist(p, stones.get(i).pos) < 2 * r) return false;
    }
    return true;
  }
}
