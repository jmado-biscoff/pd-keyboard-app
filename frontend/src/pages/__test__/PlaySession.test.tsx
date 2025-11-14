// frontend/src/pages/__test__/PlaySession.test.tsx
import { describe, it, expect, vi, beforeAll, beforeEach, afterEach } from "vitest";
import { render, screen, fireEvent, act, cleanup } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";

// 🧼 Clean up DOM after each test
afterEach(() => cleanup());

// 🔧 Stub backend URL for environment
beforeAll(() => {
  vi.stubEnv("VITE_API_URL", "http://localhost:5000/api/auth");
});

// ✅ Declare navigateMock BEFORE mocking react-router-dom (Vitest hoists mocks)
const navigateMock = vi.fn();

vi.mock("react-router-dom", async () => {
  const actual = await vi.importActual<typeof import("react-router-dom")>("react-router-dom");
  return { ...actual, useNavigate: () => navigateMock };
});

// ── Lightweight component mocks ────────────────
vi.mock("../../components/Logo.tsx", () => ({ Logo: () => <div>Logo</div> }));
vi.mock("../../components/PixelButton.tsx", () => ({
  PixelButton: (props: any) => <button {...props}>{props.children}</button>,
}));
vi.mock("../../components/PixelCard.tsx", () => ({
  PixelCard: (props: any) => <div {...props} />,
}));

// Utility to render PlaySession
const renderPlaySession = async () => {
  const { default: PlaySession } = await import("../student/PlaySession.tsx");
  return render(
    <MemoryRouter initialEntries={["/play?type=practice&level=1"]}>
      <PlaySession />
    </MemoryRouter>
  );
};

const TYPING_PAYLOAD = { data: ["AB"] };

// Mock fetch sequence for backend endpoints
function mockFetchSequence() {
  const fetchMock = vi.fn((input: RequestInfo | URL, init?: RequestInit) => {
    const url = String(input);
    if (url.includes("/api/typing/level/")) {
      return Promise.resolve(new Response(JSON.stringify(TYPING_PAYLOAD), { status: 200 }));
    }
    return Promise.resolve(new Response(JSON.stringify({}), { status: 200 }));
  });
  // @ts-expect-error override global fetch
  global.fetch = fetchMock;
  return fetchMock;
}

beforeEach(() => {
  vi.useFakeTimers();
  mockFetchSequence();
});

afterEach(() => {
  vi.clearAllMocks();
  vi.useRealTimers();
});

// ───────────────────────────────────────────────
// ✅ Test 1: keyboard shows + highlights pressed key
// ───────────────────────────────────────────────
describe("PlaySession - Keyboard UI", () => {
  it("renders all layout keys and highlights pressed key (A)", async () => {
    await renderPlaySession();
    await act(async () => {}); // let effects run
    const input = screen.getByPlaceholderText("Type here...");

    // Simulate typing 'a' to trigger onKeyDown highlight logic
    await act(async () => fireEvent.keyDown(input, { key: "a", code: "KeyA" }));

    // There are two "A"s (prompt + key). Pick the keyboard *key* by its class.
    const allAs = screen.getAllByText("A");
    const keyA = allAs.find(
      (el) => el.tagName === "DIV" && el.classList.contains("pixel-border")
    ) as HTMLElement | undefined;

    expect(keyA).toBeTruthy();
    expect(keyA!.classList.contains("bg-green-500")).toBe(true);
    expect(keyA!.classList.contains("text-white")).toBe(true);
  });
});

// ───────────────────────────────────────────────
// ✅ Test 2: typed characters appear in the input field
// ───────────────────────────────────────────────
describe("PlaySession - Typing Input", () => {
  it("updates the input value correctly", async () => {
    await renderPlaySession();
    await act(async () => {});
    const input = screen.getByPlaceholderText("Type here...");

    await act(async () => {
      fireEvent.change(input, { target: { value: "A" } });
      fireEvent.change(input, { target: { value: "AB" } });
    });

    expect((input as HTMLInputElement).value).toBe("AB");
  });
});

// ───────────────────────────────────────────────
// ✅ Test 3: Back to Dashboard button navigation
// ───────────────────────────────────────────────
describe("PlaySession - Back to Dashboard button", () => {
  it("calls navigate('/student/play') when clicked", async () => {
    const { default: PlaySession } = await import("../student/PlaySession.tsx");

    render(
      <MemoryRouter initialEntries={["/play?type=practice&level=1"]}>
        <PlaySession />
      </MemoryRouter>
    );

    const buttons = screen.getAllByRole("button");
    const backBtn = buttons[0]; // first button is the back button
    fireEvent.click(backBtn);

    expect(navigateMock).toHaveBeenCalledWith("/student/play");
  });
});
