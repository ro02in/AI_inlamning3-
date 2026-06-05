// Oliwer Carpman, Rafal Galinski, Robin Karim
// =============================================================
// Game - turn management + state machine.
// =============================================================
//
//   AIMING     : waiting for the current team to set up a shot
//   SIMULATING : a stone has been released, physics is running
//   ENDED      : all 8 stones have come to rest, end is scored
//
// Red throws first; teams alternate every shot until 8 total.
// =============================================================

static final int STONES_PER_TEAM = 4;
static final int TOTAL_STONES    = STONES_PER_TEAM * 2;

enum GameState { AIMING, SIMULATING, ENDED }

class Shot {
  float curl;    // -1..+1
  float speed;   // world units / s
  float angle;   // radians, 0 = straight up the sheet, +CW

  Shot(float curl, float speed, float angle) {
    this.curl  = curl;
    this.speed = speed;
    this.angle = angle;
  }
}

class Game {
  GameState        state;
  ArrayList<Stone> stones;
  int              currentTeam;   // TEAM_RED or TEAM_YELLOW
  int              startingTeam = TEAM_RED;
  int              stonesThrown;  // 0..TOTAL_STONES

  // Result of scoring (set when state becomes ENDED).
  int winningTeam = -1;
  int winningPoints = 0;

  Game() {
    reset();
  }

  void reset() {
    stones        = new ArrayList<Stone>();
    state         = GameState.AIMING;
    currentTeam   = startingTeam;
    stonesThrown  = 0;
    winningTeam   = -1;
    winningPoints = 0;
    if (ui != null) ui.onTurnStart();
  }

  void toggleStartingTeam() {
    startingTeam = (startingTeam == TEAM_RED) ? TEAM_YELLOW : TEAM_RED;
    reset();
  }

  void changeStartingTeam() {
    startingTeam = (startingTeam == TEAM_RED) ? TEAM_YELLOW : TEAM_RED;
  }

  int teamForThrowIndex(int throwIndex) {
    if (throwIndex % 2 == 0) return startingTeam;
    return startingTeam == TEAM_RED ? TEAM_YELLOW : TEAM_RED;
  }

  // Number of stones each team has left to throw.
  int stonesRemaining(int team) {
    int thrownByTeam = 0;
    for (int i = 0; i < stonesThrown; i++) {
      int t = teamForThrowIndex(i);
      if (t == team) thrownByTeam++;
    }
    return STONES_PER_TEAM - thrownByTeam;
  }

  // ---------------------------------------------------------
  // fire: release a stone with the given Shot.
  // Only valid in AIMING; transitions to SIMULATING.
  // ---------------------------------------------------------
  void fire(Shot shot) {
    if (state != GameState.AIMING) return;
    if (stonesThrown >= TOTAL_STONES) return;

    PVector h = sheet.hackWorld();
    Stone s = new Stone(h.x, h.y, currentTeam);
    s.curl = constrain(shot.curl, -1, 1);

    // angle = 0 means straight toward the house (+y in world coords).
    // Positive angle rotates clockwise (toward +x).
    float vx = sin(shot.angle) * shot.speed;
    float vy = cos(shot.angle) * shot.speed;
    s.vel.set(vx, vy);

    stones.add(s);
    stonesThrown++;
    state = GameState.SIMULATING;
  }

  // ---------------------------------------------------------
  // update: called every frame after Physics.step.
  // Handles the AIMING <-> SIMULATING <-> ENDED transitions.
  // ---------------------------------------------------------
  void update() {
    if (state != GameState.SIMULATING) return;
    if (anyMoving()) return;

    if (stonesThrown >= TOTAL_STONES) {
      ScoreResult r = house.scoreEnd(stones);
      winningTeam   = r.team;
      winningPoints = r.points;
      state = GameState.ENDED;
    } else {
      currentTeam = (currentTeam == TEAM_RED) ? TEAM_YELLOW : TEAM_RED;
      state = GameState.AIMING;
      if (ui != null) ui.onTurnStart();
    }
  }

  boolean anyMoving() {
    for (Stone s : stones) if (s.isMoving()) return true;
    return false;
  }

  String stateLabel() {
    switch (state) {
      case AIMING:     return "AIMING";
      case SIMULATING: return "SIMULATING";
      case ENDED:      return "ENDED";
      default:         return "?";
    }
  }

  String teamLabel(int team) {
    return team == TEAM_RED ? "RED" : "YELLOW";
  }
}
