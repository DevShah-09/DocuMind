import { useState, useCallback, useEffect } from 'react';
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
import NotFound from '@/pages/NotFound';

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
    // Clear session storage chat history
    sessionStorage.removeItem('qi_messages');
    sessionStorage.removeItem('qi_results');
    sessionStorage.removeItem('qi_query');
    sessionStorage.removeItem('qi_last_searched_query');
    sessionStorage.removeItem('qi_session_id');
    // Force QueryInterface to remount with clean state
    setNewQueryKey((prev) => prev + 1);
  };

  // ── Collections cache at shell level (persists across tab navigations) ──────
  const [collections, setCollections] = useState([]);
  const [collectionsLoading, setCollectionsLoading] = useState(false);
  const { getAccessToken, session } = useAuth();

  const fetchCollections = useCallback(async () => {
    const token = getAccessToken();
    if (!token) return;

    setCollectionsLoading(true);
    try {
      const response = await fetch('http://localhost:8000/documents/categories', {
        headers: { Authorization: `Bearer ${token}` },
      });

      if (!response.ok) {
        throw new Error(`Failed to fetch collections: ${response.status}`);
      }

      const data = await response.json();
      const categories = Array.isArray(data) ? data : (data.categories || []);

      // Update persistent cache
      localStorage.setItem('documind:collections_cache', JSON.stringify(categories));

      setCollections(
        categories.map((item, index) => ({
          id: item.category || item.category_name || `col_${index}`,
          name: item.display_name || item.category_name || item.name || 'Collection',
          documents: typeof item.count === 'number' ? item.count : (item.documents?.length || 0),
          docs: item.documents || [],
          color: colorOptions[index % colorOptions.length],
        }))
      );
    } catch (err) {
      console.error('Failed to fetch collections:', err);
      // Restore from cache on error
      try {
        const cached = localStorage.getItem('documind:collections_cache');
        if (cached) {
          const cats = JSON.parse(cached);
          setCollections(cats.map((item, index) => ({
            id: item.category || item.category_name || `col_${index}`,
            name: item.display_name || item.category_name || item.name || 'Collection',
            documents: typeof item.count === 'number' ? item.count : (item.documents?.length || 0),
            docs: item.documents || [],
            color: colorOptions[index % colorOptions.length],
          })));
        }
      } catch { /* ignore */ }
    } finally {
      setCollectionsLoading(false);
    }
  }, [getAccessToken]);

  // Re-fetch whenever session becomes available (login / token refresh)
  useEffect(() => {
    if (!session) return;
    fetchCollections();
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [session]);

  // Event-driven collections refresh (in-tab CustomEvent and cross-tab BroadcastChannel)
  useEffect(() => {
    const handleUpdate = () => {
      fetchCollections();
    };

    window.addEventListener('documind:collections_updated', handleUpdate);

    let channel = null;
    try {
      channel = new BroadcastChannel('documind_events');
      channel.onmessage = (event) => {
        if (event.data?.type === 'COLLECTIONS_UPDATED') {
          fetchCollections();
        }
      };
    } catch (_) {}

    return () => {
      window.removeEventListener('documind:collections_updated', handleUpdate);
      if (channel) channel.close();
    };
  }, [fetchCollections]);

  return (
    <div className="flex h-screen overflow-hidden">
      <Sidebar
        onNewQuery={handleNewQuery}
        darkMode={darkMode}
        toggleDarkMode={toggleDarkMode}
      />
      <main className="flex-1 overflow-auto">
        <Routes>
          <Route path="/" element={<Navigate to="/query" replace />} />
          <Route path="/query" element={<QueryInterface key={newQueryKey} />} />
          <Route
            path="/upload"
            element={
              <UploadDocuments
                queue={uploadQueue}
                setQueue={setUploadQueue}
                onUploadComplete={fetchCollections}
              />
            }
          />
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
          {/* Custom 404 Route */}
          <Route path="*" element={<NotFound />} />
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
              <Route path="/404" element={<NotFound />} />

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
