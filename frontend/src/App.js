import { useState, useCallback, useEffect, useRef } from 'react';
import '@/App.css';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { Toaster } from '@/components/ui/sonner';

// Auth
import { AuthProvider, useAuth } from '@/contexts/AuthContext';
import ProtectedRoute from '@/components/ProtectedRoute';
import Login from '@/pages/Login';

// App shell
import Sidebar from '@/components/Sidebar';

// Protected pages
import QueryInterface from '@/pages/QueryInterface';
import UploadDocuments from '@/pages/UploadDocuments';
import MyCollections from '@/pages/MyCollections';
import QueryHistory from '@/pages/QueryHistory';
import Settings from '@/pages/Settings';

const colorOptions = ['blue', 'green', 'purple', 'orange', 'pink', 'cyan'];

/**
 * AppShell — rendered only for authenticated users.
 * Wraps the sidebar + main content layout.
 * Collections + upload queue state live here so they survive page navigation.
 */
function AppShell({ darkMode, toggleDarkMode }) {
  // ── Persisted upload queue ──────────────────────────────────────────────────
  const [uploadQueue, setUploadQueue] = useState([]);

  // ── Key to force QueryInterface remount on "New Query" ──────────────────────
  const [newQueryKey, setNewQueryKey] = useState(0);

  const handleNewQuery = () => {
    sessionStorage.removeItem('qi_messages');
    sessionStorage.removeItem('qi_results');
    sessionStorage.removeItem('qi_query');
    sessionStorage.removeItem('qi_session_id');
    sessionStorage.removeItem('qi_last_searched_query');
    setNewQueryKey((k) => k + 1);
  };

  // ── Persisted collections ───────────────────────────────────────────────────
  const [collections, setCollections] = useState([]);
  const [collectionsLoading, setCollectionsLoading] = useState(false);
  const hasFetchedRef = useRef(false);
  const { getAccessToken, session } = useAuth();

  const fetchCollections = useCallback(async () => {
    setCollectionsLoading(true);
    try {
      const token = getAccessToken();
      if (!token) { setCollectionsLoading(false); return; }

      const response = await fetch('http://localhost:8000/collections/summary', {
        headers: { 'Authorization': `Bearer ${token}` }
      });

      if (!response.ok) { setCollectionsLoading(false); return; }

      const data = await response.json();
      if (!Array.isArray(data)) { setCollectionsLoading(false); return; }

      setCollections(
        data.map((item, index) => ({
          id: index,
          name: item.category_name,
          documents: item.count,
          docs: item.documents || [],
          color: colorOptions[index % colorOptions.length],
        }))
      );
    } catch (err) {
      console.error('Failed to fetch collections:', err);
    } finally {
      setCollectionsLoading(false);
    }
  }, [getAccessToken]);

  // Fetch once when the session first becomes available
  useEffect(() => {
    if (!session) return;
    if (hasFetchedRef.current) return;
    hasFetchedRef.current = true;
    fetchCollections();
  }, [session, fetchCollections]);

  return (
    <div className="flex h-screen overflow-hidden">
      <Sidebar onNewQuery={handleNewQuery} />
      <main className="flex-1 overflow-auto">
        <Routes>
          <Route path="/" element={<Navigate to="/query" replace />} />
          <Route path="/query" element={<QueryInterface key={newQueryKey} />} />
          <Route path="/upload" element={<UploadDocuments queue={uploadQueue} setQueue={setUploadQueue} />} />
          <Route
            path="/collections"
            element={
              <MyCollections
                collections={collections}
                setCollections={setCollections}
                collectionsLoading={collectionsLoading}
                fetchCollections={fetchCollections}
              />
            }
          />
          <Route path="/history" element={<QueryHistory />} />
          <Route
            path="/settings"
            element={
              <Settings darkMode={darkMode} toggleDarkMode={toggleDarkMode} />
            }
          />
        </Routes>
      </main>
    </div>
  );
}

function App() {
  const [darkMode, setDarkMode] = useState(false);

  const toggleDarkMode = () => {
    setDarkMode(!darkMode);
    document.documentElement.classList.toggle('dark');
  };

  return (
    <div className={darkMode ? 'dark' : ''}>
      <div className="min-h-screen bg-slate-50 dark:bg-slate-950 transition-colors duration-300">
        <BrowserRouter>
          {/*
            AuthProvider must be inside BrowserRouter so that
            useNavigate() works inside AuthContext.
          */}
          <AuthProvider>
            <Routes>
              {/* ── Public routes ─────────────────────────────── */}
              <Route path="/login" element={<Login />} />

              {/* ── Protected routes ──────────────────────────── */}
              <Route
                path="/*"
                element={
                  <ProtectedRoute>
                    <AppShell
                      darkMode={darkMode}
                      toggleDarkMode={toggleDarkMode}
                    />
                  </ProtectedRoute>
                }
              />
            </Routes>
          </AuthProvider>

          <Toaster position="top-right" />
        </BrowserRouter>
      </div>
    </div>
  );
}

export default App;
