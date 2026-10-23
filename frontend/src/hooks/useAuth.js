import { useEffect, useState } from "react";
import { supabase } from "../lib/supabase";

// Who is logged in? `session` is null when nobody is.
// `checking` is true while we look for a saved login on page load.
export function useAuth() {
  const [session, setSession] = useState(null);
  const [checking, setChecking] = useState(true);

  useEffect(() => {
    // A login from last time is kept by Supabase, so the customer stays in.
    supabase.auth.getSession().then(({ data }) => {
      setSession(data.session);
      setChecking(false);
    });

    // Keep `session` up to date on login, logout and token refresh.
    const { data } = supabase.auth.onAuthStateChange((_event, newSession) => {
      setSession(newSession);
    });
    return () => data.subscription.unsubscribe();
  }, []);

  // Returns an error message for the form, or null if it worked.
  async function logIn(email, password) {
    const { error } = await supabase.auth.signInWithPassword({ email, password });
    if (!error) return null;
    // Show the real reason for anything else, e.g. "Email logins are disabled".
    return error.code === "invalid_credentials" ? "Wrong email or password." : error.message;
  }

  async function logOut() {
    await supabase.auth.signOut();
  }

  return { session, checking, email: session?.user?.email, logIn, logOut };
}
