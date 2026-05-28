// Loads and saves fixed training layouts from data/situations/*.sit
import java.util.Arrays;

class SavedSituation {
  String name;
  String filename;
  ArrayList<Stone> stones;

  SavedSituation(String name, String filename, ArrayList<Stone> stones) {
    this.name     = name;
    this.filename = filename;
    this.stones   = stones;
  }
}

class SituationLibrary {
  static final String FOLDER = "situations";

  ArrayList<SavedSituation> situations = new ArrayList<SavedSituation>();

  void reloadFromDisk() {
    situations.clear();
    File dir = new File(dataPath(FOLDER));
    if (!dir.exists() || !dir.isDirectory()) return;

    String[] names = dir.list();
    if (names == null) return;
    Arrays.sort(names);

    for (String filename : names) {
      if (!filename.endsWith(".sit")) continue;
      SavedSituation s = loadFile(filename);
      if (s != null) situations.add(s);
    }
  }

  SavedSituation loadFile(String filename) {
    String[] lines = loadStrings(FOLDER + "/" + filename);
    if (lines == null) return null;

    ArrayList<Stone> stones = new ArrayList<Stone>();
    String name = filename.replace(".sit", "");

    for (String line : lines) {
      line = line.trim();
      if (line.length() == 0 || line.startsWith("#")) {
        if (line.startsWith("# name:")) name = line.substring(7).trim();
        continue;
      }
      String[] parts = splitTokens(line);
      if (parts.length < 3) continue;
      int team = parts[0].equalsIgnoreCase("RED") || parts[0].equalsIgnoreCase("ROD")
                 ? TEAM_RED : TEAM_YELLOW;
      float x = parseCoord(parts[1]);
      float y = parseCoord(parts[2]);
      if (Float.isNaN(x) || Float.isNaN(y)) continue;
      Stone st = new Stone(x, y, team);
      st.hogPassed = true;
      stones.add(st);
    }
    if (stones.isEmpty()) return null;
    return new SavedSituation(name, filename, stones);
  }

  float parseCoord(String s) {
    return parseFloat(s.replace(',', '.'));
  }

  String formatCoord(float v) {
    return nf(v, 1, 2).replace(',', '.');
  }

  boolean save(String filename, String displayName, ArrayList<Stone> stones) {
    if (stones.isEmpty()) return false;
    if (!filename.endsWith(".sit")) filename += ".sit";

    String[] lines = new String[stones.size() + 1];
    lines[0] = "# name: " + displayName;
    for (int i = 0; i < stones.size(); i++) {
      Stone s = stones.get(i);
      String team = s.team == TEAM_RED ? "RED" : "YELLOW";
      lines[i + 1] = team + " " + formatCoord(s.pos.x) + " " + formatCoord(s.pos.y);
    }
    File dir = new File(dataPath(FOLDER));
    if (!dir.exists()) dir.mkdirs();
    saveStrings(FOLDER + "/" + filename, lines);
    reloadFromDisk();
    return true;
  }

  int count() { return situations.size(); }

  ArrayList<Stone> layoutCopy(int index) {
    ArrayList<Stone> copy = new ArrayList<Stone>();
    if (index < 0 || index >= situations.size()) return copy;
    for (Stone s : situations.get(index).stones) {
      Stone c = new Stone(s.pos.x, s.pos.y, s.team);
      c.hogPassed = s.hogPassed;
      copy.add(c);
    }
    return copy;
  }

  ArrayList<Stone> layoutCopy(SavedSituation sit) {
    ArrayList<Stone> copy = new ArrayList<Stone>();
    for (Stone s : sit.stones) {
      Stone c = new Stone(s.pos.x, s.pos.y, s.team);
      c.hogPassed = s.hogPassed;
      copy.add(c);
    }
    return copy;
  }
}
