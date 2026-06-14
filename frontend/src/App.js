import { useState } from 'react';
import '@/App.css';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { Toaster } from '@/components/ui/sonner';

// Auth
import { AuthProvider } from '@/contexts/AuthContext';
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

/**
 * AppShell — rendered only for authenticated users.
 * Wraps the sidebar + main content layout.
 */
function AppShell({ darkMode, toggleDarkMode }) {
  return (
    <div className="flex h-screen overflow-hidden">
      <Sidebar />
      <main className="flex-1 overflow-auto">
        <Routes>
          <Route path="/" element={<Navigate to="/query" replace />} />
          <Route path="/query" element={<QueryInterface />} />
          <Route path="/upload" element={<UploadDocuments />} />
          <Route path="/collections" element={<MyCollections />} />
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
