# Claude Code Prompt: Graded/Activity Hover Tooltip Implementation

## Objective
Implement a hover tooltip on the "Active Activity" banner in the Student Dashboard that displays contextual messages based on the student's enrollment status and session state.

## Context
- **File to modify**: `frontend/src/pages/student/Dashboard.tsx`
- **Current implementation**: Lines 124-144 show an "Active Activity" banner that appears when `activeEvalName` exists and `evalRemaining > 0`
- **Student data available**: `classrooms` array (lines 20, 37) contains all classrooms the student has joined

## Requirements

### Tooltip Logic (Priority Order)
The tooltip should check conditions in this order and display the appropriate message:

1. **Student Not Enrolled in Any Classroom**
   - **Condition**: `classrooms.length === 0`
   - **Tooltip Message**: "You are not enrolled in any classroom. Please join a classroom to participate in activities."
   - **Tooltip Style**: Warning/informational (yellow/amber theme)

2. **Student Enrolled but No Active Session**
   - **Condition**: `classrooms.length > 0 && (!activeEvalName || evalRemaining <= 0)`
   - **Tooltip Message**: "Your teacher has not started a new session yet. Check back later or contact your teacher."
   - **Tooltip Style**: Informational (blue/neutral theme)

3. **Active Session Available**
   - **Condition**: `activeEvalName && evalRemaining > 0`
   - **Tooltip Message**: "Active evaluation in progress! Click 'Go to Activity' to participate."
   - **Tooltip Style**: Success/action (green/accent theme)

### Implementation Details

#### 1. Tooltip Component Selection
Choose one of these approaches:
- **Option A**: Use an existing tooltip library (e.g., Radix UI Tooltip, if already in the project)
- **Option B**: Create a custom tooltip component using CSS and React state
- **Recommended**: Check if the project uses shadcn/ui components (common with PixelCard/PixelButton pattern), and use their Tooltip component

#### 2. Tooltip Placement
- Position the tooltip on the **"Active Activity" banner** section (lines 124-144)
- Add a small info icon (HelpCircle from lucide-react is already imported on line 7) next to the "Active Activity" text
- Tooltip should appear on hover over the info icon or the entire banner

#### 3. Visual Design Requirements
- **Pixel art aesthetic**: Match the existing PixelCard and PixelButton styling
- **Font**: Use `font-pixel` class (consistent with existing design)
- **Animation**: Subtle fade-in/fade-out (200-300ms)
- **Z-index**: Ensure tooltip appears above other elements
- **Responsive**: Should work on both desktop and mobile (consider touch events)

#### 4. Code Structure

```typescript
// Suggested helper function to add
const getActivityTooltipContent = () => {
  if (classrooms.length === 0) {
    return {
      message: "You are not enrolled in any classroom. Please join a classroom to participate in activities.",
      variant: "warning" // yellow/amber
    };
  }
  
  if (!activeEvalName || evalRemaining <= 0) {
    return {
      message: "Your teacher has not started a new session yet. Check back later or contact your teacher.",
      variant: "info" // blue/neutral
    };
  }
  
  return {
    message: "Active evaluation in progress! Click 'Go to Activity' to participate.",
    variant: "success" // green/accent
  };
};
```

#### 5. Banner Modification
Update the "Active Activity" banner section (lines 124-144) to:
- Always show the banner (not just when `activeEvalName && evalRemaining > 0`)
- Change the banner appearance based on state:
  - **No enrollment**: Red/warning variant with disabled button
  - **Enrolled, no session**: Purple/neutral variant with disabled button
  - **Active session**: Current red variant with active button
- Add the tooltip trigger (info icon) next to the title

#### 6. Example JSX Structure

```tsx
{/* Active Activity / Graded Banner - Always Visible */}
<PixelCard 
  variant={
    classrooms.length === 0 ? "red" : 
    (!activeEvalName || evalRemaining <= 0) ? "purple" : 
    "red"
  } 
  className="text-white mb-6"
>
  <div className="flex items-center justify-between">
    <div className="flex items-center gap-3">
      <Clock size={20} />
      <div>
        <div className="flex items-center gap-2">
          <p className="font-pixel text-sm">
            {activeEvalName && evalRemaining > 0 
              ? `Active Activity — ${activeEvalName}`
              : "Graded Activity Status"
            }
          </p>
          {/* Tooltip Trigger */}
          <TooltipProvider>
            <Tooltip>
              <TooltipTrigger>
                <HelpCircle size={14} className="opacity-70 hover:opacity-100 transition-opacity" />
              </TooltipTrigger>
              <TooltipContent>
                <p className="font-pixel text-xs">
                  {getActivityTooltipContent().message}
                </p>
              </TooltipContent>
            </Tooltip>
          </TooltipProvider>
        </div>
        <p className="font-pixel text-[10px] opacity-80">
          {activeEvalName && evalRemaining > 0
            ? `${Math.ceil(evalRemaining / 60)} minutes remaining`
            : classrooms.length === 0
            ? "Join a classroom to get started"
            : "Waiting for teacher to start session"
          }
        </p>
      </div>
    </div>
    <PixelButton 
      variant="accent" 
      size="sm" 
      onClick={() => navigate("/student/play")}
      disabled={!activeEvalName || evalRemaining <= 0}
    >
      {activeEvalName && evalRemaining > 0 ? "Go to Activity" : "Not Available"}
    </PixelButton>
  </div>
</PixelCard>
```

## Technical Considerations

### 1. Dependencies
Check if these are already installed:
```bash
# If using shadcn/ui
npx shadcn-ui@latest add tooltip

# If using Radix UI directly
npm install @radix-ui/react-tooltip
```

### 2. Accessibility
- Add `aria-label` to the info icon
- Ensure tooltip is keyboard accessible (focus state)
- Use semantic HTML

### 3. Performance
- Memoize the tooltip content calculation if needed
- Avoid re-rendering on every hover

### 4. Testing Scenarios
Test these states:
1. New student (no classrooms joined)
2. Student in classroom but no active evaluation
3. Student with active evaluation
4. Transition between states (e.g., evaluation expires)

## Additional Enhancements (Optional)

1. **Animation**: Add a subtle pulse animation to the info icon to draw attention
2. **Click to dismiss**: Allow users to click the tooltip to keep it open
3. **Link to join classroom**: In the "not enrolled" state, add a link/button to join a classroom
4. **Countdown timer**: Show real-time countdown in the tooltip when session is active
5. **History**: Show when the last session was (if available from backend)

## Files to Modify

1. **Primary**: `frontend/src/pages/student/Dashboard.tsx`
2. **Possible new file**: `frontend/src/components/ActivityTooltip.tsx` (if creating custom component)
3. **Possible new file**: `frontend/src/components/ui/tooltip.tsx` (if using shadcn/ui)

## Expected Outcome

After implementation:
- ✅ Tooltip appears on hover over the info icon in the Activity banner
- ✅ Correct message displays based on enrollment and session status
- ✅ Banner is always visible (not just when session is active)
- ✅ Visual design matches the pixel art aesthetic
- ✅ Tooltip is accessible and responsive
- ✅ Code is clean, maintainable, and follows existing patterns

## Notes for Claude

- Preserve all existing functionality (don't break the current active evaluation display)
- Match the existing code style (TypeScript, React hooks, Tailwind CSS)
- Use existing imports where possible (lucide-react icons, existing components)
- Maintain the pixel art aesthetic throughout
- Add appropriate TypeScript types for new functions/components
- Consider edge cases (e.g., what if classrooms data is still loading?)
