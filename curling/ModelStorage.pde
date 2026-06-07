// Oliwer Carpman, Rafal Galinski, Robin Karim
// Save / load deterministic NeuralPolicy weights + ShotTypeSelector.
// File format: text .curlmodel with header + @policy blocks + @selector block.
class ModelStorage {
    static final String FORMAT_VERSION = "curling_model_simple_v1";
    static final String DEFAULT_DIR    = "data/models";

    String lastMessage = "";

    boolean saveActiveModel(String path) {
        ArrayList<String> lines = new ArrayList<String>();
        lines.add(FORMAT_VERSION);
        lines.add("training_done=" + trainingDone);

        StringBuilder body = new StringBuilder();
        // Serialize all 8 gradient-ensemble experts.
        for (int i = 0; i < gradientEnsemble.count; i++) {
            PolicyGradientTraining t = gradientEnsemble.trainers[i];
            t.policy.appendSave(body, gradientEnsemble.names[i], t.updateCount);
        }
        lines.add("policy_count=" + gradientEnsemble.count);

        // Serialize the selector network.
        shotSelector.appendSave(body);

        lines.add("");
        String[] bodyLines = split(body.toString(), "\n");
        for (String l : bodyLines) {
            if (l.length() > 0) lines.add(l);
        }

        String[] out = lines.toArray(new String[lines.size()]);
        ensureParentDir(path);
        saveStrings(path, out);
        lastMessage = "Sparad (" + gradientEnsemble.count + " experts) → " + path;
        println(lastMessage);
        return true;
    }

    boolean loadActiveModel(String path) {
        String[] lines = loadStrings(path);
        if (lines == null || lines.length == 0) {
            lastMessage = "Kunde inte lasa: " + path;
            return false;
        }
        if (!trim(lines[0]).equals(FORMAT_VERSION)) {
            lastMessage = "Fel modellformat. Skapa en ny modell med den forenklade AI:n.";
            println(lastMessage);
            return false;
        }

        int fileTrainingDone = trainingDone;
        for (String raw : lines) {
            String line = trim(raw);
            if (line.length() == 0 || line.startsWith("#")) continue;
            if (line.startsWith("@")) break;
            int eq = line.indexOf('=');
            if (eq <= 0) continue;
            String key = line.substring(0, eq);
            String val = line.substring(eq + 1);
            if (key.equals("training_done")) fileTrainingDone = parseInt(val);
        }
        trainingDone = fileTrainingDone;

        // Parse @policy and @selector blocks.
        ArrayList<LoadedPolicy> loaded = new ArrayList<LoadedPolicy>();
        int idx = 0;
        while (idx < lines.length) {
            String line = trim(lines[idx]);
            if (line.startsWith("@policy ")) {
                LoadedPolicy lp = new LoadedPolicy();
                lp.name   = line.substring(8);
                lp.policy = new NeuralPolicy();
                idx = lp.policy.loadFromLines(lines, idx);
                lp.updateCount = lp.policy.lastLoadedUpdateCount;
                loaded.add(lp);
            } else if (line.equals("@selector")) {
                idx = shotSelector.loadFromLines(lines, idx);
            } else {
                idx++;
            }
        }

        if (loaded.isEmpty()) {
            lastMessage = "Ingen policy hittades i " + path;
            return false;
        }

        // Apply policies to gradient ensemble experts.
        int n = min(gradientEnsemble.count, loaded.size());
        for (int i = 0; i < n; i++) {
            gradientEnsemble.trainers[i].setPolicy(loaded.get(i).policy,
                                                loaded.get(i).updateCount);
        }

        lastMessage = "Laddad (" + loaded.size() + " experts, "
            + trainingDone + " steg) <- " + path;
        println(lastMessage);
        return true;
    }

    void ensureParentDir(String path) {
        File f = new File(path);
        File parent = f.getParentFile();
        if (parent != null && !parent.exists()) parent.mkdirs();
    }

    File modelDir() {
        File dir = new File(sketchPath(DEFAULT_DIR));
        if (!dir.exists()) dir.mkdirs();
        return dir;
    }

    String defaultSavePath() {
        modelDir();
        return sketchPath(DEFAULT_DIR + "/model_"
            + year() + nf(month(), 2) + nf(day(), 2)
            + "_" + nf(hour(), 2) + nf(minute(), 2) + nf(second(), 2)
            + ".curlmodel");
    }

    class LoadedPolicy {
        String      name;
        NeuralPolicy policy;
        int         updateCount = -1;
    }
}
