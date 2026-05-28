// =============================================================
// Sheet - standard curling geometry (feet) + drawing.
// =============================================================
// World Y (feet): +y is up-ice toward the target (top of screen).
// hackY = 0 at delivery. Order along play: hack -> long ice ->
// hog line -> target tee/house -> far back line.
//
// The target tee is NOT at TEE_TO_HACK from the hack; that distance
// only relates tee and hack at the same end. Here hogY is chosen
// first, then teeY = hogY + HOG_FROM_TEE_FT (hog is 21 ft toward
// hack from the tee in real curling, so tee is 21 ft up-ice from hog).
//
// Screen Y: sy = iceOffsetY + (yMax - y) * pxPerFt (larger y higher).
// x runs 0 .. SHEET_WIDTH_FT.
// =============================================================

class Sheet {

  // ----- Key dimensions (feet) --------------------------------
  final float SHEET_WIDTH_FT = 14.5;

  // Same-end tee–hack spacing (reference only; not used for target tee).
  final float BACK_LINE_FROM_TEE_FT = 6.0;
  final float HACK_FROM_BACK_LINE_FT = 4.0;
  final float TEE_TO_HACK_FT = BACK_LINE_FROM_TEE_FT + HACK_FROM_BACK_LINE_FT;

  // Hog is 21 ft toward the hack from the target tee line.
  final float HOG_FROM_TEE_FT = 21.0;

  final float HOUSE_12_R_FT = 6.0;
  final float HOUSE_8_R_FT  = 4.0;
  final float HOUSE_4_R_FT  = 2.0;
  final float BUTTON_R_FT   = 0.5;

  // Long ice from hack to the hog line you must cross (tunable).
  final float HACK_TO_HOG_FT = 45.0;

  // Margins inside the world span (feet). Keep FAR_EXTENT small so the
  // far back line sits just below the top of the ice panel; that raises
  // pxPerFt (letterbox uses min(width, height) scale) and widens the sheet.
  final float HACK_MARGIN_FT = 3.0;
  final float FAR_EXTENT_FT  = 3.0;

  // ----- World y (feet), delivery toward +y --------------------
  final float hackY = 0.0;
  final float hogY  = hackY + HACK_TO_HOG_FT;
  final float teeY  = hogY + HOG_FROM_TEE_FT;
  // Far (house-side) back line: 6 ft past tee toward end boards.
  final float backFarY = teeY + BACK_LINE_FROM_TEE_FT;
  // Optional: back line 4 ft in front of hack (same-end detail).
  final float hackSideBackLineY = hackY + HACK_FROM_BACK_LINE_FT;

  final float yMin = hackY - HACK_MARGIN_FT;
  final float yMax = backFarY + FAR_EXTENT_FT;
  final float worldSpanY = yMax - yMin;

  final float centerX = SHEET_WIDTH_FT * 0.5;

  float pxPerFt;
  float iceOffsetX;
  float iceOffsetY;

  Sheet() {
    float sx = ICE_W / SHEET_WIDTH_FT;
    float sy = ICE_H / worldSpanY;
    pxPerFt = min(sx, sy);
    iceOffsetX = (ICE_W  - SHEET_WIDTH_FT * pxPerFt) * 0.5;
    iceOffsetY = (ICE_H  - worldSpanY     * pxPerFt) * 0.5;
  }

  PVector hackWorld() {
    return new PVector(centerX, hackY);
  }

  PVector teeWorld() {
    return new PVector(centerX, teeY);
  }

  PVector worldToScreen(PVector w) {
    return new PVector(worldToScreenX(w.x), worldToScreenY(w.y));
  }

  float worldToScreenX(float xFt) {
    return iceOffsetX + xFt * pxPerFt;
  }

  float worldToScreenY(float yFt) {
    return iceOffsetY + (yMax - yFt) * pxPerFt;
  }

  float worldLenToScreen(float ft) {
    return ft * pxPerFt;
  }

  float screenToWorldX(float sx) {
    return (sx - iceOffsetX) / pxPerFt;
  }

  float screenToWorldY(float sy) {
    return yMax - (sy - iceOffsetY) / pxPerFt;
  }

  PVector screenToWorld(float sx, float sy) {
    return new PVector(screenToWorldX(sx), screenToWorldY(sy));
  }

  void drawSheet() {
    pushStyle();
    rectMode(CORNER);
    noStroke();

    fill(232, 242, 252);
    rect(0, 0, ICE_W, ICE_H);

    float ix0 = iceOffsetX;
    float iy0 = iceOffsetY;
    float iw  = SHEET_WIDTH_FT * pxPerFt;
    float ih  = worldSpanY * pxPerFt;

    fill(228, 238, 248);
    rect(ix0, iy0, iw, ih);

    stroke(90, 110, 130);
    strokeWeight(max(2, pxPerFt * 0.15));
    line(ix0, iy0, ix0, iy0 + ih);
    line(ix0 + iw, iy0, ix0 + iw, iy0 + ih);

    float cx = worldToScreenX(centerX);
    stroke(160, 175, 195);
    strokeWeight(max(1, pxPerFt * 0.04));
    float dash = pxPerFt * 1.2;
    for (float y = iy0; y < iy0 + ih; y += dash * 2) {
      line(cx, y, cx, min(y + dash, iy0 + ih));
    }

    float xL = worldToScreenX(0);
    float xR = worldToScreenX(SHEET_WIDTH_FT);

    // Near-end back line (4 ft in front of hack) — subtle.
    float nb = worldToScreenY(hackSideBackLineY);
    stroke(130, 145, 165, 160);
    strokeWeight(max(1, pxPerFt * 0.05));
    line(xL, nb, xR, nb);

    // Hog line (must clear to score).
    float hy = worldToScreenY(hogY);
    stroke(210, 120, 40);
    strokeWeight(max(3, pxPerFt * 0.14));
    line(xL, hy, xR, hy);

    // Single target house at teeY.
    float tcx = worldToScreenX(centerX);
    float tcy = worldToScreenY(teeY);
    noFill();
    strokeWeight(max(2, pxPerFt * 0.12));
    stroke(55, 95, 180);
    ellipse(tcx, tcy, worldLenToScreen(HOUSE_12_R_FT * 2), worldLenToScreen(HOUSE_12_R_FT * 2));
    stroke(210, 218, 228);
    ellipse(tcx, tcy, worldLenToScreen(HOUSE_8_R_FT * 2), worldLenToScreen(HOUSE_8_R_FT * 2));
    stroke(200, 70, 70);
    ellipse(tcx, tcy, worldLenToScreen(HOUSE_4_R_FT * 2), worldLenToScreen(HOUSE_4_R_FT * 2));

    stroke(70, 85, 105);
    strokeWeight(max(1.5, pxPerFt * 0.06));
    line(xL, tcy, xR, tcy);

    float bkf = worldToScreenY(backFarY);
    stroke(120, 60, 60);
    strokeWeight(max(2, pxPerFt * 0.08));
    line(xL, bkf, xR, bkf);

    float hx = worldToScreenX(centerX);
    float hyk = worldToScreenY(hackY);
    noStroke();
    fill(80, 95, 115);
    float hw = pxPerFt * 1.1;
    float hh = pxPerFt * 0.35;
    rect(hx - hw * 2.2, hyk - hh * 0.5, hw * 2, hh, 3);
    rect(hx + hw * 0.2, hyk - hh * 0.5, hw * 2, hh, 3);

    fill(40, 55, 75, 200);
    textAlign(RIGHT, BOTTOM);
    textSize(max(9, pxPerFt * 0.55));
    float lx = ix0 + iw - 4;
    text("Hog", lx, hy - 2);
    text("Back", lx, bkf - 2);
    text("Tee", lx, tcy - 2);
    textAlign(CENTER, BOTTOM);
    fill(40, 55, 75, 200);
    textSize(max(9, pxPerFt * 0.55));
    text("Hack", hx, hyk - hh - 4);

    popStyle();
  }
}
