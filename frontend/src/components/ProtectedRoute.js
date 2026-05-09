/**
 * ProtectedRoute.js — Auth Guard
 * ================================
 * Wraps any route that requires authentication.
 * Redirects unauthenticated users to /login.
 * Shows a loading spinner while the session resolves on first render.
 */
import { Navigate } from 'react-router-dom';
import { useAuth } from '@/contexts/AuthContext';
import { Brain } from 'lucide-react';

const ProtectedRoute = ({ children }) => {
  const { user, loading } = useAuth();

  // Resolving initial session — avoid premature redirect
  if (loading) {
    return (
      <div className="flex h-screen items-center justify-center bg-slate-950">
        <div className="flex flex-col items-center gap-4">
          <div className="relative">
            <div className="w-16 h-16 rounded-2xl bg-gradient-to-br from-cyan-500 to-blue-600 flex items-center justify-center animate-pulse">
              <Brain className="w-8 h-8 text-white" strokeWidth={1.5} />
            </div>
            <div className="absolute -inset-1 rounded-2xl bg-gradient-to-br from-cyan-500/30 to-blue-600/30 blur-sm animate-pulse" />
          </div>
          <p className="text-slate-400 text-sm font-medium tracking-wide animate-pulse">
            Verifying session…
          </p>
        </div>
      </div>
    );
  }

  if (!user) {
    return <Navigate to="/login" replace />;
  }

  return children;
};

export default ProtectedRoute;
