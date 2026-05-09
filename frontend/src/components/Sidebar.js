import { Link, useLocation } from 'react-router-dom';
import {
  Brain,
  LayoutDashboard,
  Upload,
  FolderOpen,
  History,
  Settings,
  PlusCircle,
  Activity,
  LogOut,
  User,
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { cn } from '@/lib/utils';
import { useAuth } from '@/contexts/AuthContext';
import { toast } from 'sonner';

const Sidebar = () => {
  const location = useLocation();
  const { user, signOut } = useAuth();

  const navItems = [
    { path: '/', icon: LayoutDashboard, label: 'Dashboard' },
    { path: '/upload', icon: Upload, label: 'Upload Documents' },
    { path: '/collections', icon: FolderOpen, label: 'My Collections' },
    { path: '/history', icon: History, label: 'Query History' },
    { path: '/settings', icon: Settings, label: 'Settings' },
  ];

  const handleSignOut = async () => {
    try {
      await signOut();
      toast.success('Signed out successfully');
    } catch {
      toast.error('Sign out failed');
    }
  };

  // Get display name and avatar from user metadata
  const displayName =
    user?.user_metadata?.full_name ||
    user?.email?.split('@')[0] ||
    'User';
  const avatarUrl = user?.user_metadata?.avatar_url;
  const email = user?.email;

  return (
    <aside
      className="w-[260px] h-screen glassmorphism border-r border-slate-200/50 dark:border-slate-800/50 flex flex-col"
      data-testid="sidebar"
    >
      {/* Logo */}
      <div className="px-6 py-8">
        <Link to="/" className="flex items-center gap-3 group" data-testid="logo-link">
          <div className="p-2 rounded-lg bg-gradient-to-br from-slate-900 to-slate-800 dark:from-slate-100 dark:to-slate-200 group-hover:scale-105 transition-transform duration-200">
            <Brain className="w-6 h-6 text-white dark:text-slate-900" strokeWidth={1.5} />
          </div>
          <div>
            <h1 className="text-xl font-bold text-slate-900 dark:text-white tracking-tight">
              DocuMind AI
            </h1>
            <p className="text-xs text-slate-500 dark:text-slate-400 font-medium">
              Intelligent Search
            </p>
          </div>
        </Link>
      </div>

      {/* New Query Button */}
      <div className="px-6 mb-6">
        <Link to="/query" data-testid="new-query-button">
          <Button className="w-full bg-gradient-to-r from-slate-900 to-slate-800 hover:from-slate-800 hover:to-slate-700 text-white dark:from-slate-100 dark:to-slate-200 dark:text-slate-900 dark:hover:from-slate-200 dark:hover:to-slate-300 shadow-sm rounded-lg h-11 font-semibold transition-all duration-200">
            <PlusCircle className="w-5 h-5 mr-2" strokeWidth={2} />
            New Query
          </Button>
        </Link>
      </div>

      {/* Navigation */}
      <nav className="flex-1 px-4 space-y-1">
        {navItems.map((item) => {
          const Icon = item.icon;
          const isActive = location.pathname === item.path;
          return (
            <Link
              key={item.path}
              to={item.path}
              data-testid={`nav-${item.label.toLowerCase().replace(/\s+/g, '-')}`}
              className={cn(
                'flex items-center gap-3 px-4 py-3 rounded-lg font-medium text-sm transition-all duration-200',
                isActive
                  ? 'bg-slate-100 dark:bg-slate-800 text-slate-900 dark:text-white shadow-sm'
                  : 'text-slate-600 dark:text-slate-400 hover:bg-slate-100/50 dark:hover:bg-slate-800/50 hover:text-slate-900 dark:hover:text-white'
              )}
            >
              <Icon className="w-5 h-5" strokeWidth={1.5} />
              {item.label}
            </Link>
          );
        })}
      </nav>

      {/* System Status */}
      <div
        className="px-6 py-4 border-t border-slate-200/50 dark:border-slate-800/50"
        data-testid="system-status"
      >
        <div className="space-y-2 mb-4">
          <div className="flex items-center gap-2">
            <div className="w-2 h-2 rounded-full bg-green-500 animate-pulse" />
            <span className="text-xs font-medium text-slate-600 dark:text-slate-400">
              Qdrant Connected
            </span>
          </div>
          <div className="flex items-center gap-2">
            <Activity className="w-3 h-3 text-cyan-500" strokeWidth={2} />
            <span className="text-xs font-medium text-slate-600 dark:text-slate-400">
              LLM Active
            </span>
          </div>
        </div>

        {/* User profile + sign out */}
        {user && (
          <div
            className="flex items-center gap-3 pt-3 border-t border-slate-200/50 dark:border-slate-800/50"
            data-testid="user-profile"
          >
            {/* Avatar */}
            <div className="w-8 h-8 rounded-full overflow-hidden flex-shrink-0 bg-gradient-to-br from-cyan-500 to-blue-600 flex items-center justify-center">
              {avatarUrl ? (
                <img
                  src={avatarUrl}
                  alt={displayName}
                  className="w-full h-full object-cover"
                  referrerPolicy="no-referrer"
                />
              ) : (
                <User className="w-4 h-4 text-white" strokeWidth={1.5} />
              )}
            </div>

            {/* Name & email */}
            <div className="flex-1 min-w-0">
              <p className="text-xs font-semibold text-slate-800 dark:text-slate-200 truncate">
                {displayName}
              </p>
              <p className="text-[10px] text-slate-500 dark:text-slate-500 truncate">
                {email}
              </p>
            </div>

            {/* Sign out */}
            <button
              id="sign-out-btn"
              onClick={handleSignOut}
              title="Sign out"
              className="p-1.5 rounded-lg text-slate-400 hover:text-red-500 hover:bg-red-50 dark:hover:bg-red-900/20 transition-all duration-200 flex-shrink-0"
            >
              <LogOut className="w-4 h-4" strokeWidth={1.5} />
            </button>
          </div>
        )}
      </div>
    </aside>
  );
};

export default Sidebar;
