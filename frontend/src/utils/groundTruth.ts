// Ground truth validation for touch typing correctness
// Ports KEY_TO_EXPECTED_HAND and KEY_TO_EXPECTED_FINGER from detect_keyboard_live.py

export const YOLO_LABEL_FIX: Record<string, string> = { y: "z", z: "y" };

export const KEY_TO_EXPECTED_HAND: Record<string, string> = {
  q: "left",  w: "left",  e: "left",  r: "left",  t: "left",
  a: "left",  s: "left",  d: "left",  f: "left",  g: "left",
  z: "left",  x: "left",  c: "left",  v: "left",  b: "left",
  y: "right", u: "right", i: "right", o: "right", p: "right",
  h: "right", j: "right", k: "right", l: "right",
  n: "right", m: "right",
};

export const KEY_TO_EXPECTED_FINGER: Record<string, string> = {
  q: "pinky",  a: "pinky",  z: "pinky",
  w: "ring",   s: "ring",   x: "ring",
  e: "middle", d: "middle", c: "middle",
  r: "index",  t: "index",  f: "index",  g: "index",  v: "index",  b: "index",
  y: "index",  u: "index",  h: "index",  j: "index",  n: "index",  m: "index",
  i: "middle", k: "middle",
  o: "ring",   l: "ring",
  p: "pinky",
};

/**
 * Validate a keystroke against ground truth touch typing rules.
 * Returns "Correct" only if SVM predicts correct AND hand/finger match ground truth.
 */
export function validateKeystroke(
  svmPrediction: number,
  hand: string,
  finger: string,
  key: string
): "Correct" | "Incorrect" {
  const k = key.toLowerCase();
  const expectedHand = KEY_TO_EXPECTED_HAND[k];
  const expectedFinger = KEY_TO_EXPECTED_FINGER[k];

  const handCorrect = !expectedHand || hand === expectedHand;
  const fingerCorrect = !expectedFinger || finger === expectedFinger;

  if (svmPrediction === 1 && handCorrect && fingerCorrect) {
    return "Correct";
  }
  return "Incorrect";
}
