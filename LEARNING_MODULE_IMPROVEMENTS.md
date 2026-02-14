# Learning Module Improvements - Implementation Prompt

## Overview
Implement the following enhancements to the student learning modules to improve readability, user experience, and ensure proper learning progression with 100% accuracy enforcement.

---

## 1. Add Background to Learning Module Cards for Text Readability

### Problem
Text in the learning modules (Learn.tsx) is difficult to read due to the dynamic video background.

### Solution
Wrap all module cards in `Learn.tsx` with a semi-transparent background overlay to ensure text readability.

### Implementation Details

**File: `frontend/src/pages/student/Learn.tsx`**

- Add a background overlay to the entire modules grid section
- Apply a semi-transparent dark background (e.g., `bg-black/40` or `bg-black/50`) with backdrop blur
- Ensure PixelCard components have sufficient contrast against the video background
- Consider adding a subtle border or shadow to enhance card visibility

**Example approach:**
```tsx
{/* Modules Grid - wrapped with readable background */}
<div className="bg-black/50 backdrop-blur-sm rounded-lg p-6 border-2 border-yellow-300/30">
  <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
    {dynamicModules.map((module) => (
      // ... existing module cards
    ))}
  </div>
</div>
```

---

## 2. Per-Student Learning Module Progress Persistence

### Problem
Currently, learning module progress is stored in `localStorage` only, which is not tied to individual student accounts. Progress is lost when switching devices or browsers.

### Solution
Implement backend persistence for learning module progress tied to each student's account.

### Implementation Details

#### Backend Changes

**File: `backend/models/User.ts` (or equivalent User model)**

Add a new field to the User schema for students:
```typescript
learningProgress: {
  type: Map,
  of: {
    completed: Boolean,
    accuracy: Number,
    lastAttemptDate: Date,
    attempts: Number
  },
  default: new Map()
}
```

**New API Endpoints:**

Create new routes in `backend/routes/student.ts` (or create a new `learn.ts` route file):

1. **GET `/api/student/learning-progress`**
   - Returns the student's learning module progress
   - Response format:
   ```json
   {
     "1": { "completed": true, "accuracy": 95, "lastAttemptDate": "2026-02-14T10:00:00Z", "attempts": 2 },
     "2": { "completed": false, "accuracy": 0, "lastAttemptDate": null, "attempts": 0 },
     ...
   }
   ```

2. **POST `/api/student/learning-progress`**
   - Updates progress for a specific module
   - Request body:
   ```json
   {
     "moduleId": 1,
     "completed": true,
     "accuracy": 95
   }
   ```

3. **PUT `/api/student/learning-progress/reset`**
   - Resets all learning progress for the student
   - Requires confirmation

#### Frontend Changes

**File: `frontend/src/pages/student/Learn.tsx`**

- Replace localStorage-only approach with API calls
- Fetch progress from backend on component mount
- Maintain localStorage as a cache/fallback for offline scenarios
- Show loading state while fetching progress

**File: `frontend/src/pages/student/LearnSession.tsx`**

- Update the `handleFinish()` function to save progress to backend
- Include accuracy percentage in the saved progress
- Show success/error toast on save completion

**Example implementation:**
```typescript
// In Learn.tsx
useEffect(() => {
  const fetchProgress = async () => {
    const token = localStorage.getItem("token");
    try {
      const res = await fetch(`${BASE_URL}/api/student/learning-progress`, {
        headers: { Authorization: `Bearer ${token}` }
      });
      const data = await res.json();
      if (res.ok) {
        setProgress(data);
        // Also update localStorage as cache
        localStorage.setItem("typingModuleProgress", JSON.stringify(data));
      }
    } catch (error) {
      // Fallback to localStorage if API fails
      const saved = localStorage.getItem("typingModuleProgress");
      if (saved) setProgress(JSON.parse(saved));
    }
  };
  fetchProgress();
}, []);

// In LearnSession.tsx - handleFinish()
const handleFinish = async () => {
  const token = localStorage.getItem("token");
  try {
    await fetch(`${BASE_URL}/api/student/learning-progress`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${token}`
      },
      body: JSON.stringify({
        moduleId,
        completed: true,
        accuracy: accuracyPercent
      })
    });
    toast.success("Progress saved!");
  } catch (error) {
    toast.error("Failed to save progress");
  }
  
  // Update localStorage as well
  const saved = JSON.parse(localStorage.getItem("typingModuleProgress") || "{}");
  saved[moduleId] = { completed: true, accuracy: accuracyPercent };
  localStorage.setItem("typingModuleProgress", JSON.stringify(saved));
  
  navigate("/student/learn");
};
```

---

## 3. Add Tooltip for Locked Modules

### Problem
Users don't understand why modules are locked or what they need to do to unlock them.

### Solution
Add a tooltip that appears when hovering over locked modules, explaining that previous modules must be completed first.

### Implementation Details

**File: `frontend/src/pages/student/Learn.tsx`**

Install or use existing tooltip component (shadcn/ui has a Tooltip component):
```bash
npx shadcn-ui@latest add tooltip
```

Wrap locked module cards with a Tooltip:

```tsx
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";

// In the modules grid:
{dynamicModules.map((module) => (
  <TooltipProvider key={module.id}>
    <Tooltip>
      <TooltipTrigger asChild>
        <div>
          <PixelCard
            variant={module.unlocked ? "yellow" : "default"}
            className={`relative transition-all duration-200 ${
              module.unlocked
                ? "cursor-pointer hover:brightness-110"
                : "opacity-50 cursor-not-allowed"
            }`}
          >
            {/* ... existing card content ... */}
          </PixelCard>
        </div>
      </TooltipTrigger>
      {!module.unlocked && (
        <TooltipContent className="font-pixel text-xs bg-black/90 text-yellow-300 border-2 border-yellow-400">
          <p>🔒 Complete Module {module.id - 1} first to unlock this module</p>
        </TooltipContent>
      )}
    </Tooltip>
  </TooltipProvider>
))}
```

---

## 4. Enforce 100% Correct Key and Finger Usage - No Progression on Errors

### Problem
Currently, the learning module advances to the next character even if the user presses the wrong key or uses incorrect finger placement. This doesn't ensure proper learning.

### Solution
Modify the detection logic to prevent progression unless the key press is 100% correct (correct key AND correct finger).

### Implementation Details

**File: `frontend/src/pages/student/LearnSession.tsx`**

Modify the SSE detection handler in the `case "detection":` block:

**Current behavior (lines 187-262):**
- Advances character index regardless of correctness
- Only tracks correct/incorrect for statistics

**New behavior:**
- Only advance `currentCharIndex` if BOTH conditions are met:
  1. Correct key pressed (`key === expectedChar`)
  2. Correct finger used (`data.ml_label === "Correct"`)
- If incorrect, show error feedback but DO NOT advance
- Add visual/audio feedback for errors
- Optionally: Add a "retry counter" to show how many attempts on current character

**Modified detection logic:**

```typescript
case "detection": {
  if (!data.key) return;
  const key = String(data.key).toUpperCase();
  if (!/^[A-Z]$/.test(key)) return;
  if (isCalibratingRef.current) return;

  const currentDrills = drillsRef.current;
  const dIdx = currentDrillIndexRef.current;
  const cIdx = currentCharIndexRef.current;

  if (dIdx >= currentDrills.length) return;
  const currentDrill = currentDrills[dIdx];
  if (!currentDrill || cIdx >= currentDrill.length) return;

  const expectedChar = currentDrill[cIdx].toUpperCase();

  // Duplicate prevention
  const signature = `${key}-${dIdx}-${cIdx}`;
  if (lastEventRef.current === signature) return;
  lastEventRef.current = signature;

  const mlCorrect = data.ml_label === "Correct";
  const keyCorrect = key === expectedChar;
  
  // ✅ NEW: Only proceed if BOTH key AND finger are correct
  const isFullyCorrect = mlCorrect && keyCorrect;

  // Visual feedback on VirtualKeyboard
  const color = getKeyColor(key, expectedChar, data.ml_label);
  setActiveKeys({ [key]: color });
  setTimeout(() => setActiveKeys({}), 300);

  if (isFullyCorrect) {
    // ✅ CORRECT: Update counts and advance
    correctCountRef.current++;
    setCorrectCount(correctCountRef.current);
    setLastTip(null);
    
    // Update char feedback for drill display
    setCharFeedback((prev) => ({
      ...prev,
      [cIdx]: "correct",
    }));

    // Advance character pointer
    const nextCharIndex = cIdx + 1;
    if (nextCharIndex >= currentDrill.length) {
      // Drill complete — advance to next
      const nextDrillIndex = dIdx + 1;
      if (nextDrillIndex >= currentDrills.length) {
        setModuleComplete(true);
      } else {
        currentDrillIndexRef.current = nextDrillIndex;
        currentCharIndexRef.current = 0;
        setCurrentDrillIndex(nextDrillIndex);
        setCurrentCharIndex(0);
        setCharFeedback({});
        setLastTip(null);

        // Update expected keys for new drill
        fetch(`${BASE_URL}/api/detect/set-expected`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            words: currentDrills.slice(nextDrillIndex),
            startIndex: 0,
          }),
        });
      }
    } else {
      currentCharIndexRef.current = nextCharIndex;
      setCurrentCharIndex(nextCharIndex);
    }
  } else {
    // ❌ INCORRECT: Do NOT advance, show feedback
    incorrectCountRef.current++;
    setIncorrectCount(incorrectCountRef.current);
    
    // Show specific error message
    if (!keyCorrect && !mlCorrect) {
      setLastTip(`Wrong key AND wrong finger! Expected: ${expectedChar}. ${getCorrectionTip(expectedChar)}`);
    } else if (!keyCorrect) {
      setLastTip(`Wrong key! Expected: ${expectedChar}`);
    } else if (!mlCorrect) {
      setLastTip(`Wrong finger! ${getCorrectionTip(expectedChar)}`);
    }
    
    // Optional: Flash the incorrect character in red
    setCharFeedback((prev) => ({
      ...prev,
      [cIdx]: "incorrect",
    }));
    
    // Clear the incorrect feedback after a short delay
    setTimeout(() => {
      setCharFeedback((prev) => {
        const updated = { ...prev };
        delete updated[cIdx];
        return updated;
      });
    }, 500);
  }
  break;
}
```

**Additional enhancements:**

1. **Add retry counter (optional):**
```typescript
const [retryCount, setRetryCount] = useState<Record<string, number>>({});

// In the incorrect block:
const retryKey = `${dIdx}-${cIdx}`;
setRetryCount(prev => ({
  ...prev,
  [retryKey]: (prev[retryKey] || 0) + 1
}));

// Display retry count in UI
{retryCount[`${currentDrillIndex}-${currentCharIndex}`] > 0 && (
  <p className="font-pixel text-xs text-orange-400">
    Attempts: {retryCount[`${currentDrillIndex}-${currentCharIndex}`] + 1}
  </p>
)}
```

2. **Add audio feedback (optional):**
```typescript
// Create audio elements
const correctSound = new Audio('/sounds/correct.mp3');
const incorrectSound = new Audio('/sounds/incorrect.mp3');

// Play on correct/incorrect
if (isFullyCorrect) {
  correctSound.play().catch(() => {});
} else {
  incorrectSound.play().catch(() => {});
}
```

3. **Enhanced visual feedback:**
```tsx
{/* In the drill display section, add shake animation on error */}
<PixelCard 
  className={`w-full max-w-md bg-black/60 border-2 border-yellow-300 backdrop-blur-sm ${
    charFeedback[currentCharIndex] === "incorrect" ? "animate-shake" : ""
  }`}
>
  {/* ... drill content ... */}
</PixelCard>

{/* Add to your CSS/Tailwind config: */}
// tailwind.config.ts
{
  theme: {
    extend: {
      keyframes: {
        shake: {
          '0%, 100%': { transform: 'translateX(0)' },
          '10%, 30%, 50%, 70%, 90%': { transform: 'translateX(-5px)' },
          '20%, 40%, 60%, 80%': { transform: 'translateX(5px)' },
        }
      },
      animation: {
        shake: 'shake 0.5s ease-in-out',
      }
    }
  }
}
```

---

## Testing Checklist

After implementing these changes, verify:

- [ ] Module cards in Learn.tsx have readable text against video background
- [ ] Learning progress is saved to backend and persists across sessions
- [ ] Progress is tied to individual student accounts
- [ ] Locked modules show tooltip on hover explaining unlock requirements
- [ ] Incorrect key presses do NOT advance to next character
- [ ] Incorrect finger usage does NOT advance to next character
- [ ] Clear error messages appear for wrong key vs wrong finger
- [ ] Visual feedback (colors, animations) clearly indicates errors
- [ ] Module completion only occurs when all characters are typed correctly
- [ ] Accuracy percentage reflects the new strict validation
- [ ] Progress reset functionality works with backend persistence

---

## Additional Considerations

1. **Migration:** Existing localStorage progress should be migrated to backend on first login after update
2. **Offline support:** Consider implementing a sync mechanism when connection is restored
3. **Performance:** Ensure the strict validation doesn't cause lag or missed inputs
4. **Accessibility:** Ensure tooltips are keyboard-accessible and screen-reader friendly
5. **User feedback:** Consider adding a "strict mode" toggle in settings if users want to practice with relaxed rules

---

## Files to Modify

### Frontend
- `frontend/src/pages/student/Learn.tsx` - Background overlay, tooltips, progress fetching
- `frontend/src/pages/student/LearnSession.tsx` - Strict validation logic, progress saving
- `frontend/src/components/ui/tooltip.tsx` - Add if not exists

### Backend
- `backend/models/User.ts` - Add learningProgress field
- `backend/routes/student.ts` or `backend/routes/learn.ts` - Add progress endpoints
- `backend/controllers/studentController.ts` - Add progress management logic

### Configuration
- `frontend/tailwind.config.ts` - Add shake animation (optional)
- `frontend/package.json` - Ensure tooltip dependencies are included
