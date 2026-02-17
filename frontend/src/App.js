import { useState } from 'react';
import '@/App.css';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { Toaster } from '@/components/ui/sonner';
import Sidebar from '@/components/Sidebar';
import Dashboard from '@/pages/Dashboard';
import QueryInterface from '@/pages/QueryInterface';
import UploadDocuments from '@/pages/UploadDocuments';
import MyCollections from '@/pages/MyCollections';
import QueryHistory from '@/pages/QueryHistory';
import Settings from '@/pages/Settings';

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
          <div className="flex h-screen overflow-hidden">
            <Sidebar />
            <main className="flex-1 overflow-auto">
              <Routes>
                <Route path="/" element={<Dashboard />} />
                <Route path="/query" element={<QueryInterface />} />
                <Route path="/upload" element={<UploadDocuments />} />
                <Route path="/collections" element={<MyCollections />} />
                <Route path="/history" element={<QueryHistory />} />
                <Route path="/settings" element={<Settings darkMode={darkMode} toggleDarkMode={toggleDarkMode} />} />
              </Routes>
            </main>
          </div>
          <Toaster position="top-right" />
        </BrowserRouter>
      </div>
    </div>
  );
}

export default App;
