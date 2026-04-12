/**
 * Cloud OCR Layout Detector via Backend Proxy
 */

const OCR_PROXY_URL = '/api/ocr/parse';

export async function checkIsQwerty(
  video: HTMLVideoElement,
  keyPositions: Record<string, number[]>
): Promise<{ isQwerty: boolean; detectedLayout: string; reason?: string }> {
  // Check the full QWERTY sequence
  const keysToCheck = ['q', 'w', 'e', 'r', 't', 'y'];
  
  const canvas = document.createElement('canvas');
  const ctx = canvas.getContext('2d');
  if (!ctx) throw new Error("Could not get canvas context");

  let matchCount = 0;
  const results: Record<string, string> = {};

  try {
    for (const key of keysToCheck) {
      if (!keyPositions[key]) continue;

      let [x1, y1, x2, y2] = keyPositions[key];
      let width = x2 - x1;
      let height = y2 - y1;

      // Add padding for better context
      const padX = width * 0.15;
      const padY = height * 0.15;
      x1 = Math.max(0, x1 - padX);
      y1 = Math.max(0, y1 - padY);
      width += padX * 2;
      height += padY * 2;

      // Upscale for cloud OCR
      const scaleFactor = 6; 
      canvas.width = width * scaleFactor;
      canvas.height = height * scaleFactor;
      
      ctx.imageSmoothingEnabled = true;
      ctx.imageSmoothingQuality = 'high';
      
      // Setup background
      ctx.fillStyle = 'white';
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      
      // CRITICAL: We re-enable the invert logic here. 
      // Even cloud OCRs perform significantly better on isolated White-on-Black crops if they are pre-inverted.
      ctx.filter = 'grayscale(1) contrast(3) brightness(1.2) invert(1)';
      ctx.drawImage(video, x1, y1, width, height, 0, 0, canvas.width, canvas.height);

      // Convert canvas to base64
      const base64Image = canvas.toDataURL('image/jpeg', 0.85);

      // Call Backend Proxy
      const response = await fetch(OCR_PROXY_URL, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ base64Image })
      });

      const data = await response.json();
      
      if (data.IsErroredOnProcessing) {
        console.error("Cloud OCR error:", data.ErrorMessage);
        continue;
      }

      const detectedText = data.ParsedResults?.[0]?.ParsedText || "";
      const cleanText = detectedText.toUpperCase().replace(/[^A-Z]/g, '').trim();
      results[key] = cleanText;

      if (fuzzyMatch(key, cleanText)) {
        matchCount++;
      }
    }

    // Logic for layout determination
    const isColemak = results['e']?.includes('F') || results['r']?.includes('P');
    const isAzerty = results['q']?.includes('A');

    if (isColemak || isAzerty) {
      return { isQwerty: false, detectedLayout: isColemak ? 'Colemak' : 'AZERTY' };
    }

    // Pass if at least 3 out of 6 keys match (50% accuracy requirement for cloud OCR)
    const isQwerty = matchCount >= 3;

    return { 
      isQwerty, 
      detectedLayout: isQwerty ? 'QWERTY' : 'unknown' 
    };

  } catch (error) {
    console.error("Cloud OCR Error:", error);
    // Fallback to true if network/API fails to avoid blocking the user
    return { isQwerty: true, detectedLayout: 'QWERTY' };
  }
}

/**
 * Fuzzy matching remains similar to Tesseract logic but more strict due to higher cloud accuracy
 */
function fuzzyMatch(expected: string, detected: string): boolean {
  const e = expected.toUpperCase();
  if (detected.includes(e)) return true;
  
  // Minimal fuzzy list for Cloud OCR which is less prone to character confusion
  const misreads: Record<string, string[]> = {
    'Q': ['O', '0'],
    'W': ['V', 'U'],
    'E': ['3'], 
    'R': ['B', 'K'],
    'T': ['7', '1'],
    'Y': ['V', 'X']
  };

  const alternatives = misreads[e] || [];
  return alternatives.some(alt => detected.includes(alt));
}
