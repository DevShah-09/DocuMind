/**
 * supabase.js — Supabase Client Singleton
 * =========================================
 * Initialises the Supabase JS client using environment variables.
 * Import `supabase` anywhere in the app — the same instance is reused.
 *
 * Auth state (sessions, tokens) is persisted automatically to localStorage
 * by the Supabase SDK and refreshed before expiry.
 */
import { createClient } from '@supabase/supabase-js';

const supabaseUrl = process.env.REACT_APP_SUPABASE_URL;
const supabaseAnonKey = process.env.REACT_APP_SUPABASE_ANON_KEY;

if (!supabaseUrl || !supabaseAnonKey) {
  console.error(
    '[Supabase] Missing REACT_APP_SUPABASE_URL or REACT_APP_SUPABASE_ANON_KEY in .env'
  );
}

export const supabase = createClient(supabaseUrl, supabaseAnonKey, {
  auth: {
    autoRefreshToken: true,       // Refresh access token before it expires
    persistSession: true,         // Persist session across page reloads (localStorage)
    detectSessionInUrl: true,     // Handle OAuth redirects automatically
  },
});
