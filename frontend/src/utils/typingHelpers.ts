/**
 * Typing helper utilities for PlaySession
 */

/**
 * Provides finger placement tip for a given character
 */
export const getCorrectionTip = (char: string): string => {
  const map: Record<string, string> = {
    A: "Use your left pinky",
    S: "Use your left ring",
    D: "Use your left middle",
    F: "Use your left index",
    G: "Use your left index",
    H: "Use your right index",
    J: "Use your right index",
    K: "Use your right middle",
    L: "Use your right ring",
    ";": "Use your right pinky",
    Q: "Use your left pinky",
    W: "Use your left ring",
    E: "Use your left middle",
    R: "Use your left index",
    T: "Use your left index",
    Y: "Use your right index",
    U: "Use your right index",
    I: "Use your right middle",
    O: "Use your right ring",
    P: "Use your right pinky",
    Z: "Use your left pinky",
    X: "Use your left ring",
    C: "Use your left middle",
    V: "Use your left index",
    B: "Use your left index",
    N: "Use your right index",
    M: "Use your right index",
  };
  return map[char] ? `${map[char]} for "${char}"` : "Check your finger placement";
};

/**
 * Determines the color for key feedback based on correctness
 */
export const getKeyColor = (
  pressedKey: string,
  expectedKey: string,
  mlLabel: string
): "green" | "red" => {
  const correctKey = pressedKey === expectedKey;
  const correctFinger = mlLabel === "Correct";

  // Only green if BOTH key and finger are correct
  // Otherwise red for any kind of mistake
  if (correctKey && correctFinger) {
    return "green"; // Fully correct
  } else {
    return "red"; // Any error (wrong key OR wrong finger)
  }
};

