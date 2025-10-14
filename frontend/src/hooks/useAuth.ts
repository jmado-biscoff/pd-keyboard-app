import { useEffect, useState } from "react";

/**
 * useAuth Hook
 * Centralized authentication management for your frontend.
 * Handles token, userType, and userName stored in localStorage.
 */
export function useAuth() {
  const [token, setToken] = useState<string | null>(null);
  const [userType, setUserType] = useState<string | null>(null);
  const [userName, setUserName] = useState<string | null>(null);

  // Load auth data from localStorage on mount
  useEffect(() => {
    const storedToken = localStorage.getItem("token");
    const storedType = localStorage.getItem("userType");
    const storedName = localStorage.getItem("userName");

    if (storedToken) setToken(storedToken);
    if (storedType) setUserType(storedType);
    if (storedName) setUserName(storedName);
  }, []);

  // Save changes to localStorage whenever auth state updates
  const saveAuth = (token: string, type: string, name: string) => {
    localStorage.setItem("token", token);
    localStorage.setItem("userType", type);
    localStorage.setItem("userName", name);

    setToken(token);
    setUserType(type);
    setUserName(name);
  };

  // Clear auth state and localStorage (for logout)
  const clearAuth = () => {
    localStorage.removeItem("token");
    localStorage.removeItem("userType");
    localStorage.removeItem("userName");

    setToken(null);
    setUserType(null);
    setUserName(null);
  };

  return {
    token,
    userType,
    userName,
    isAuthenticated: !!token,
    saveAuth,
    clearAuth,
  };
}
