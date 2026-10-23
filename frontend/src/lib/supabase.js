import { createClient } from "@supabase/supabase-js";

// One Supabase client for the whole app. It logs the customer in and keeps
// their token (the "ticket" we send to the backend). The values come from
// frontend/.env; the publishable key is safe to have in the browser.
export const supabase = createClient(
  import.meta.env.VITE_SUPABASE_URL,
  import.meta.env.VITE_SUPABASE_PUBLISHABLE_KEY,
);
