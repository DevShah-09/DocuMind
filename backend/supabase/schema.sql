-- ================================================================
-- DocuMind AI — Supabase PostgreSQL Schema + Row Level Security
-- ================================================================
-- Run this in the Supabase SQL Editor (Dashboard → SQL Editor)
-- in your project.
--
-- Prerequisites:
--   • Supabase Auth is enabled (users are managed by auth.users)
--   • This schema uses auth.uid() to enforce per-user isolation
-- ================================================================


-- ── Extensions ────────────────────────────────────────────────────────────────
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";


-- ── 1. users table ────────────────────────────────────────────────────────────
-- Mirrors auth.users for application-level queries.
-- Populated automatically via a Supabase trigger on first sign-in.

CREATE TABLE IF NOT EXISTS public.users (
    id          UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    email       TEXT UNIQUE NOT NULL,
    full_name   TEXT,
    avatar_url  TEXT,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

COMMENT ON TABLE public.users IS
    'Application user profiles — mirrors auth.users, populated on first sign-in.';

-- RLS on users
ALTER TABLE public.users ENABLE ROW LEVEL SECURITY;

-- Users can only read/update their own profile
CREATE POLICY "users: select own row"
    ON public.users FOR SELECT
    USING (auth.uid() = id);

CREATE POLICY "users: update own row"
    ON public.users FOR UPDATE
    USING (auth.uid() = id);


-- ── 2. documents table ────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS public.documents (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id         UUID NOT NULL REFERENCES public.users(id) ON DELETE CASCADE,
    filename        TEXT NOT NULL,
    category        TEXT,
    cloudflare_url  TEXT,           -- R2 object path (never a public URL)
    encrypted_key   TEXT,           -- Reserved: key metadata (NOT the raw key)
    file_size_bytes BIGINT,
    content_type    TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

COMMENT ON TABLE public.documents IS
    'Metadata for user-uploaded documents. Files are stored encrypted in R2.';
COMMENT ON COLUMN public.documents.cloudflare_url IS
    'R2 bucket path used to generate signed URLs — not a public link.';
COMMENT ON COLUMN public.documents.encrypted_key IS
    'Reserved for future key ID reference. Raw encryption keys are NEVER stored.';

-- Index for fast user document lookups
CREATE INDEX IF NOT EXISTS idx_documents_user_id ON public.documents(user_id);
CREATE INDEX IF NOT EXISTS idx_documents_created_at ON public.documents(created_at DESC);

-- RLS on documents
ALTER TABLE public.documents ENABLE ROW LEVEL SECURITY;

CREATE POLICY "documents: select own"
    ON public.documents FOR SELECT
    USING (auth.uid() = user_id);

CREATE POLICY "documents: insert own"
    ON public.documents FOR INSERT
    WITH CHECK (auth.uid() = user_id);

CREATE POLICY "documents: update own"
    ON public.documents FOR UPDATE
    USING (auth.uid() = user_id);

CREATE POLICY "documents: delete own"
    ON public.documents FOR DELETE
    USING (auth.uid() = user_id);


-- ── 3. upload_history table ───────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS public.upload_history (
    id          UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id     UUID NOT NULL REFERENCES public.users(id) ON DELETE CASCADE,
    document_id UUID REFERENCES public.documents(id) ON DELETE SET NULL,
    filename    TEXT NOT NULL,
    status      TEXT NOT NULL DEFAULT 'pending',   -- pending | success | failed
    error_msg   TEXT,
    uploaded_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

COMMENT ON TABLE public.upload_history IS
    'Audit log of all upload attempts per user.';

CREATE INDEX IF NOT EXISTS idx_upload_history_user_id ON public.upload_history(user_id);

ALTER TABLE public.upload_history ENABLE ROW LEVEL SECURITY;

CREATE POLICY "upload_history: select own"
    ON public.upload_history FOR SELECT
    USING (auth.uid() = user_id);

CREATE POLICY "upload_history: insert own"
    ON public.upload_history FOR INSERT
    WITH CHECK (auth.uid() = user_id);


-- ── 4. query_history table ────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS public.query_history (
    id          UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id     UUID NOT NULL REFERENCES public.users(id) ON DELETE CASCADE,
    query_text  TEXT NOT NULL,
    result_count INT DEFAULT 0,
    queried_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

COMMENT ON TABLE public.query_history IS
    'Log of semantic search queries per user for history view.';

CREATE INDEX IF NOT EXISTS idx_query_history_user_id ON public.query_history(user_id);
CREATE INDEX IF NOT EXISTS idx_query_history_queried_at ON public.query_history(queried_at DESC);

ALTER TABLE public.query_history ENABLE ROW LEVEL SECURITY;

CREATE POLICY "query_history: select own"
    ON public.query_history FOR SELECT
    USING (auth.uid() = user_id);

CREATE POLICY "query_history: insert own"
    ON public.query_history FOR INSERT
    WITH CHECK (auth.uid() = user_id);


-- ── 5. Auto-populate users on first sign-in ───────────────────────────────────
-- This trigger fires after Supabase creates a new auth.users record
-- and copies basic profile info into public.users.

CREATE OR REPLACE FUNCTION public.handle_new_user()
RETURNS TRIGGER
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public
AS $$
BEGIN
    INSERT INTO public.users (id, email, full_name, avatar_url)
    VALUES (
        NEW.id,
        NEW.email,
        NEW.raw_user_meta_data ->> 'full_name',
        NEW.raw_user_meta_data ->> 'avatar_url'
    )
    ON CONFLICT (id) DO UPDATE
        SET email      = EXCLUDED.email,
            full_name  = EXCLUDED.full_name,
            avatar_url = EXCLUDED.avatar_url,
            updated_at = now();
    RETURN NEW;
END;
$$;

-- Attach trigger to auth.users
DROP TRIGGER IF EXISTS on_auth_user_created ON auth.users;
CREATE TRIGGER on_auth_user_created
    AFTER INSERT OR UPDATE ON auth.users
    FOR EACH ROW EXECUTE FUNCTION public.handle_new_user();


-- ── Done ───────────────────────────────────────────────────────────────────────
-- After running this script:
--  1. Go to Supabase Dashboard → Authentication → Providers
--  2. Enable Google (paste your Google OAuth client ID + secret)
--  3. Enable Email OTP (set "OTP expiry" to 600 seconds recommended)
--  4. Add your frontend URL to Supabase → Authentication → URL Configuration
--     → Redirect URLs: http://localhost:3000/**
