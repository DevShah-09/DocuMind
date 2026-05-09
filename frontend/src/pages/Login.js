/**
 * Login.js — Simple Email/Password Authentication
 */
import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import { Brain, Mail, ArrowRight, Shield, Lock, Zap, AlertCircle, UserPlus, LogIn } from 'lucide-react';
import { useAuth } from '@/contexts/AuthContext';
import { toast } from 'sonner';

const Login = () => {
  const { user, loading, signIn, signUp } = useAuth();
  const navigate = useNavigate();

  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [isSignUp, setIsSignUp] = useState(false);
  const [authLoading, setAuthLoading] = useState(false);
  const [error, setError] = useState('');

  // Redirect if already authenticated
  useEffect(() => {
    if (!loading && user) {
      navigate('/', { replace: true });
    }
  }, [user, loading, navigate]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!email.trim() || !password.trim()) return;
    
    setError('');
    setAuthLoading(true);
    
    try {
      if (isSignUp) {
        await signUp(email.trim(), password);
        toast.success('Account created! You can now log in.');
        setIsSignUp(false);
      } else {
        await signIn(email.trim(), password);
        toast.success('Welcome back!');
      }
    } catch (err) {
      setError(err.message || 'Authentication failed. Please check your credentials.');
      toast.error(err.message || 'Authentication failed');
    } finally {
      setAuthLoading(false);
    }
  };

  const features = [
    { icon: Shield, text: 'End-to-end encrypted documents' },
    { icon: Lock, text: 'Isolated per-user storage' },
    { icon: Zap, text: 'Instant AI-powered search' },
  ];

  return (
    <div className="min-h-screen bg-slate-950 flex overflow-hidden">
      {/* ── Left Panel ────────────────────────────────────────────────── */}
      <motion.div
        initial={{ opacity: 0, x: -40 }}
        animate={{ opacity: 1, x: 0 }}
        transition={{ duration: 0.6 }}
        className="hidden lg:flex lg:w-1/2 relative flex-col justify-between p-12"
      >
        <div className="absolute inset-0 overflow-hidden pointer-events-none">
          <div className="absolute -top-32 -left-32 w-96 h-96 rounded-full bg-cyan-500/10 blur-3xl" />
          <div className="absolute top-1/2 -right-20 w-80 h-80 rounded-full bg-blue-600/10 blur-3xl" />
        </div>

        <div className="relative flex items-center gap-3">
          <div className="p-3 rounded-2xl bg-gradient-to-br from-cyan-500 to-blue-600 shadow-lg shadow-cyan-500/25">
            <Brain className="w-7 h-7 text-white" strokeWidth={1.5} />
          </div>
          <div>
            <h1 className="text-2xl font-bold text-white tracking-tight">DocuMind AI</h1>
            <p className="text-xs text-slate-400 font-medium">Intelligent Document Platform</p>
          </div>
        </div>

        <div className="relative space-y-6">
          <h2 className="text-5xl font-bold text-white leading-tight tracking-tight">
            Your documents,<br/>
            <span className="bg-gradient-to-r from-cyan-400 to-blue-500 bg-clip-text text-transparent">
              secure & intelligent.
            </span>
          </h2>
          <div className="space-y-3">
            {features.map(({ icon: Icon, text }, i) => (
              <div key={text} className="flex items-center gap-3">
                <div className="p-1.5 rounded-lg bg-cyan-500/10 border border-cyan-500/20">
                  <Icon className="w-4 h-4 text-cyan-400" strokeWidth={1.5} />
                </div>
                <span className="text-sm text-slate-300 font-medium">{text}</span>
              </div>
            ))}
          </div>
        </div>

        <div className="relative">
          <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-slate-800/50 border border-slate-700/50 text-xs text-slate-400">
            <div className="w-1.5 h-1.5 rounded-full bg-green-400 animate-pulse" />
            ChaCha20-Poly1305 Encryption Active
          </div>
        </div>
      </motion.div>

      {/* ── Right Panel ───────────────────────────────────────────────── */}
      <motion.div
        initial={{ opacity: 0, x: 40 }}
        animate={{ opacity: 1, x: 0 }}
        transition={{ duration: 0.6 }}
        className="w-full lg:w-1/2 flex items-center justify-center p-6 relative"
      >
        <div className="absolute inset-0 bg-gradient-to-br from-slate-900 to-slate-950 lg:border-l lg:border-slate-800/50" />

        <div className="relative w-full max-w-md space-y-8">
          <div>
            <h2 className="text-3xl font-bold text-white tracking-tight">
              {isSignUp ? 'Create an account' : 'Welcome back'}
            </h2>
            <p className="mt-2 text-slate-400">
              {isSignUp ? 'Join the future of document management' : 'Sign in to access your secure workspace'}
            </p>
          </div>

          <AnimatePresence mode="wait">
            {error && (
              <motion.div
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: 'auto' }}
                exit={{ opacity: 0, height: 0 }}
                className="flex items-start gap-3 p-4 rounded-xl bg-red-500/10 border border-red-500/20 text-red-400 text-sm overflow-hidden"
              >
                <AlertCircle className="w-4 h-4 mt-0.5 flex-shrink-0" />
                <span>{error}</span>
              </motion.div>
            )}
          </AnimatePresence>

          <form onSubmit={handleSubmit} className="space-y-4">
            <div className="space-y-4">
              <div className="relative">
                <Mail className="absolute left-4 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-500" />
                <input
                  type="email"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  placeholder="Email address"
                  required
                  className="w-full pl-11 pr-4 py-4 bg-slate-800/60 border border-slate-700/60 hover:border-slate-600 focus:border-cyan-500/50 focus:ring-2 focus:ring-cyan-500/10 rounded-2xl text-white outline-none transition-all"
                />
              </div>
              <div className="relative">
                <Lock className="absolute left-4 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-500" />
                <input
                  type="password"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  placeholder="Password"
                  required
                  className="w-full pl-11 pr-4 py-4 bg-slate-800/60 border border-slate-700/60 hover:border-slate-600 focus:border-cyan-500/50 focus:ring-2 focus:ring-cyan-500/10 rounded-2xl text-white outline-none transition-all"
                />
              </div>
            </div>

            <motion.button
              type="submit"
              disabled={authLoading}
              whileHover={{ scale: 1.01 }}
              whileTap={{ scale: 0.99 }}
              className="w-full flex items-center justify-center gap-2 px-6 py-4 rounded-2xl bg-gradient-to-r from-cyan-500 to-blue-600 hover:from-cyan-400 hover:to-blue-500 text-white font-semibold shadow-lg shadow-cyan-500/20 transition-all disabled:opacity-60"
            >
              {authLoading ? (
                <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
              ) : (
                <>
                  {isSignUp ? <UserPlus className="w-4 h-4" /> : <LogIn className="w-4 h-4" />}
                  {isSignUp ? 'Sign Up' : 'Sign In'}
                  <ArrowRight className="w-4 h-4" />
                </>
              )}
            </motion.button>
          </form>

          <div className="text-center">
            <button
              onClick={() => { setIsSignUp(!isSignUp); setError(''); }}
              className="text-sm text-slate-400 hover:text-cyan-400 transition-colors"
            >
              {isSignUp ? 'Already have an account? Sign In' : "Don't have an account? Sign Up"}
            </button>
          </div>
        </div>
      </motion.div>
    </div>
  );
};

export default Login;
