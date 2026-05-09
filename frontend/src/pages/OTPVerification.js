/**
 * OTPVerification.js — Email OTP Code Entry
 * ==========================================
 * Receives the email address from the Login page via React Router state.
 * The user enters the 6-digit OTP sent to their inbox.
 * On success, Supabase establishes a session and the user is redirected.
 */
import { useState, useEffect } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import { Brain, Mail, ArrowRight, RotateCcw, CheckCircle2, AlertCircle, ArrowLeft } from 'lucide-react';
import { useAuth } from '@/contexts/AuthContext';
import { toast } from 'sonner';

const OTPVerification = () => {
  const { verifyOTP, signInWithOTP, user, loading } = useAuth();
  const navigate = useNavigate();
  const location = useLocation();

  const email = location.state?.email ?? '';
  const [otp, setOtp] = useState(['', '', '', '', '', '']);
  const [verifying, setVerifying] = useState(false);
  const [resending, setResending] = useState(false);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState(false);
  const [cooldown, setCooldown] = useState(0);

  // Redirect if no email passed (direct URL access)
  useEffect(() => {
    if (!loading && !email) {
      navigate('/login', { replace: true });
    }
  }, [email, loading, navigate]);

  // Redirect if already authenticated
  useEffect(() => {
    if (!loading && user) {
      navigate('/', { replace: true });
    }
  }, [user, loading, navigate]);

  // Cooldown timer for resend
  useEffect(() => {
    if (cooldown <= 0) return;
    const timer = setTimeout(() => setCooldown((c) => c - 1), 1000);
    return () => clearTimeout(timer);
  }, [cooldown]);

  // ── OTP input handling ───────────────────────────────────────────────────────
  const handleOtpChange = (index, value) => {
    if (!/^\d*$/.test(value)) return; // digits only
    const next = [...otp];
    next[index] = value.slice(-1); // single digit
    setOtp(next);
    setError('');

    // Auto-advance to next box
    if (value && index < 5) {
      document.getElementById(`otp-${index + 1}`)?.focus();
    }
    // Auto-submit when all 6 digits are filled
    if (value && index === 5 && next.every((d) => d !== '')) {
      handleVerify(next.join(''));
    }
  };

  const handleOtpKeyDown = (index, e) => {
    if (e.key === 'Backspace' && !otp[index] && index > 0) {
      document.getElementById(`otp-${index - 1}`)?.focus();
    }
  };

  const handleOtpPaste = (e) => {
    e.preventDefault();
    const pasted = e.clipboardData.getData('text').replace(/\D/g, '').slice(0, 6);
    const next = pasted.split('');
    while (next.length < 6) next.push('');
    setOtp(next);
    if (pasted.length === 6) handleVerify(pasted);
  };

  // ── Verify OTP ───────────────────────────────────────────────────────────────
  const handleVerify = async (code = otp.join('')) => {
    if (code.length !== 6) {
      setError('Please enter all 6 digits');
      return;
    }
    setError('');
    setVerifying(true);
    try {
      await verifyOTP(email, code);
      setSuccess(true);
      toast.success('Verified! Welcome to DocuMind AI');
      setTimeout(() => navigate('/', { replace: true }), 1000);
    } catch (err) {
      setError(err.message || 'Invalid or expired OTP. Please try again.');
      setOtp(['', '', '', '', '', '']);
      document.getElementById('otp-0')?.focus();
    } finally {
      setVerifying(false);
    }
  };

  // ── Resend OTP ───────────────────────────────────────────────────────────────
  const handleResend = async () => {
    if (cooldown > 0) return;
    setResending(true);
    setError('');
    try {
      await signInWithOTP(email);
      toast.success('New OTP sent to your inbox!');
      setOtp(['', '', '', '', '', '']);
      setCooldown(60);
    } catch (err) {
      setError(err.message || 'Failed to resend OTP.');
    } finally {
      setResending(false);
    }
  };

  return (
    <div className="min-h-screen bg-slate-950 flex items-center justify-center p-6 relative overflow-hidden">
      {/* Background orbs */}
      <div className="absolute inset-0 pointer-events-none">
        <div className="absolute top-1/4 -left-40 w-96 h-96 rounded-full bg-cyan-500/8 blur-3xl" />
        <div className="absolute bottom-1/4 -right-40 w-96 h-96 rounded-full bg-blue-600/8 blur-3xl" />
      </div>

      <motion.div
        initial={{ opacity: 0, y: 24 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
        className="relative w-full max-w-md"
      >
        {/* Card */}
        <div className="bg-slate-900/80 backdrop-blur-xl border border-slate-800/60 rounded-3xl p-8 shadow-2xl shadow-black/40 space-y-8">
          {/* Logo */}
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2.5">
              <div className="p-2 rounded-xl bg-gradient-to-br from-cyan-500 to-blue-600">
                <Brain className="w-5 h-5 text-white" strokeWidth={1.5} />
              </div>
              <span className="font-bold text-white text-lg">DocuMind AI</span>
            </div>
            <button
              onClick={() => navigate('/login')}
              className="flex items-center gap-1.5 text-sm text-slate-500 hover:text-slate-300 transition-colors"
            >
              <ArrowLeft className="w-3.5 h-3.5" />
              Back
            </button>
          </div>

          {/* Header */}
          <div className="space-y-2">
            <div className="p-3 w-fit rounded-2xl bg-cyan-500/10 border border-cyan-500/20">
              <Mail className="w-6 h-6 text-cyan-400" strokeWidth={1.5} />
            </div>
            <h2 className="text-2xl font-bold text-white tracking-tight">
              Check your inbox
            </h2>
            <p className="text-sm text-slate-400 leading-relaxed">
              We sent a 6-digit code to{' '}
              <span className="font-medium text-slate-200">{email}</span>.
              Enter it below to sign in.
            </p>
          </div>

          {/* OTP Input Grid */}
          <div
            className="flex gap-2 justify-center"
            onPaste={handleOtpPaste}
          >
            {otp.map((digit, i) => (
              <motion.input
                key={i}
                id={`otp-${i}`}
                type="text"
                inputMode="numeric"
                maxLength={1}
                value={digit}
                onChange={(e) => handleOtpChange(i, e.target.value)}
                onKeyDown={(e) => handleOtpKeyDown(i, e)}
                disabled={verifying || success}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: i * 0.05 }}
                className={`
                  w-12 h-14 text-center text-xl font-bold rounded-xl border transition-all duration-200
                  bg-slate-800/60 text-white outline-none
                  ${digit
                    ? 'border-cyan-500/60 bg-cyan-500/5 shadow-[0_0_0_3px_rgba(6,182,212,0.1)]'
                    : 'border-slate-700/60 hover:border-slate-600'
                  }
                  ${error ? 'border-red-500/50' : ''}
                  ${success ? 'border-green-500/50 bg-green-500/5' : ''}
                  focus:border-cyan-500/70 focus:shadow-[0_0_0_3px_rgba(6,182,212,0.15)]
                  disabled:opacity-60
                `}
              />
            ))}
          </div>

          {/* Error */}
          <AnimatePresence>
            {error && (
              <motion.div
                initial={{ opacity: 0, y: -8 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0 }}
                className="flex items-center gap-2 text-red-400 text-sm bg-red-500/10 border border-red-500/20 rounded-xl px-4 py-3"
              >
                <AlertCircle className="w-4 h-4 flex-shrink-0" />
                {error}
              </motion.div>
            )}
          </AnimatePresence>

          {/* Success state */}
          <AnimatePresence>
            {success && (
              <motion.div
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{ opacity: 1, scale: 1 }}
                className="flex items-center justify-center gap-2 text-green-400 text-sm font-medium"
              >
                <CheckCircle2 className="w-5 h-5" />
                Verified! Redirecting…
              </motion.div>
            )}
          </AnimatePresence>

          {/* Verify button */}
          <motion.button
            id="verify-otp-btn"
            onClick={() => handleVerify()}
            disabled={verifying || success || otp.some((d) => !d)}
            whileHover={{ scale: 1.01 }}
            whileTap={{ scale: 0.99 }}
            className="w-full flex items-center justify-center gap-2 px-6 py-4 rounded-2xl bg-gradient-to-r from-cyan-500 to-blue-600 hover:from-cyan-400 hover:to-blue-500 text-white font-semibold text-base shadow-lg shadow-cyan-500/20 transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {verifying ? (
              <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
            ) : (
              <>
                Verify & Sign In
                <ArrowRight className="w-4 h-4" />
              </>
            )}
          </motion.button>

          {/* Resend */}
          <div className="text-center">
            <p className="text-sm text-slate-500">
              Didn't receive the code?{' '}
              <button
                id="resend-otp-btn"
                onClick={handleResend}
                disabled={resending || cooldown > 0}
                className="text-cyan-400 hover:text-cyan-300 font-medium disabled:opacity-50 disabled:cursor-not-allowed transition-colors inline-flex items-center gap-1"
              >
                {resending ? (
                  <div className="w-3 h-3 border border-cyan-400 border-t-transparent rounded-full animate-spin" />
                ) : (
                  <RotateCcw className="w-3 h-3" />
                )}
                {cooldown > 0 ? `Resend in ${cooldown}s` : 'Resend OTP'}
              </button>
            </p>
          </div>
        </div>
      </motion.div>
    </div>
  );
};

export default OTPVerification;
