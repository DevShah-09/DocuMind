import { Link, useLocation, useNavigate } from 'react-router-dom';
import {
  Brain,
  Upload,
  FolderOpen,
  History,
  Settings,
  PlusCircle,
  Activity,
  LogOut,
  User,
  ChevronDown,
  Moon,
  Sun,
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import {
  DropdownMenu,
  DropdownMenuTrigger,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuLabel,
} from '@/components/ui/dropdown-menu';
import { cn } from '@/lib/utils';
import { useAuth } from '@/contexts/AuthContext';
import { toast } from 'sonner';

const Sidebar = ({ onNewQuery, darkMode, toggleDarkMode }) => {
  const location = useLocation();
  const navigate = useNavigate();
  const { user, signOut } = useAuth();

  const handleNewQuery = () => {
    if (onNewQuery) onNewQuery();
    navigate('/query');
  };

  const navItems = [
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
      {/* ── Logo with Favicon Menu (Black Logo as earlier) ─────────────────── */}
      <div className="px-6 py-6 border-b border-slate-200/40 dark:border-slate-800/40">
        <DropdownMenu>
          <div className="flex items-center justify-between">
            <Link
              to="/query"
              className="flex items-center gap-3 group flex-1 min-w-0"
              data-testid="logo-link"
              title="DocuMind AI Home"
            >
              {/* Black Logo as it was earlier */}
              <div className="p-2 rounded-lg bg-gradient-to-br from-slate-900 to-slate-800 dark:from-slate-100 dark:to-slate-200 group-hover:scale-105 transition-transform duration-200 flex-shrink-0">
                <Brain className="w-6 h-6 text-white dark:text-slate-900" strokeWidth={1.5} />
              </div>
              <div className="min-w-0 flex-1">
                <h1 className="text-base font-bold text-slate-900 dark:text-white tracking-tight truncate">
                  DocuMind AI
                </h1>
                <p className="text-[11px] text-slate-500 dark:text-slate-400 font-medium truncate">
                  Intelligent Search
                </p>
              </div>
            </Link>

            {/* Favicon Menu Trigger */}
            <DropdownMenuTrigger asChild>
              <button
                className="p-1.5 rounded-lg text-slate-400 hover:text-slate-700 dark:hover:text-slate-200 hover:bg-slate-100 dark:hover:bg-slate-800 transition-colors focus:outline-none"
                title="DocuMind Quick Menu"
                data-testid="favicon-menu-button"
              >
                <ChevronDown className="w-4 h-4" />
              </button>
            </DropdownMenuTrigger>
          </div>

          {/* Favicon Menu Dropdown */}
          <DropdownMenuContent align="start" className="w-56 mt-2">
            <DropdownMenuLabel className="flex items-center gap-2 text-xs font-semibold text-slate-500 dark:text-slate-400">
              <Brain className="w-3.5 h-3.5 text-slate-700 dark:text-slate-300" />
              Quick Navigation
            </DropdownMenuLabel>
            <DropdownMenuSeparator />
            <DropdownMenuItem
              onClick={handleNewQuery}
              className="cursor-pointer gap-2"
            >
              <PlusCircle className="w-4 h-4 text-cyan-500" />
              <span>New Query</span>
            </DropdownMenuItem>
            <DropdownMenuItem
              onClick={() => navigate('/upload')}
              className="cursor-pointer gap-2"
            >
              <Upload className="w-4 h-4 text-violet-500" />
              <span>Upload Documents</span>
            </DropdownMenuItem>
            <DropdownMenuItem
              onClick={() => navigate('/collections')}
              className="cursor-pointer gap-2"
            >
              <FolderOpen className="w-4 h-4 text-amber-500" />
              <span>My Collections</span>
            </DropdownMenuItem>
            <DropdownMenuItem
              onClick={() => navigate('/history')}
              className="cursor-pointer gap-2"
            >
              <History className="w-4 h-4 text-blue-500" />
              <span>Query History</span>
            </DropdownMenuItem>
            <DropdownMenuItem
              onClick={() => navigate('/settings')}
              className="cursor-pointer gap-2"
            >
              <Settings className="w-4 h-4 text-slate-500" />
              <span>Settings</span>
            </DropdownMenuItem>

            {toggleDarkMode && (
              <>
                <DropdownMenuSeparator />
                <DropdownMenuItem
                  onClick={toggleDarkMode}
                  className="cursor-pointer gap-2"
                >
                  {darkMode ? (
                    <>
                      <Sun className="w-4 h-4 text-yellow-500" />
                      <span>Light Mode</span>
                    </>
                  ) : (
                    <>
                      <Moon className="w-4 h-4 text-slate-600" />
                      <span>Dark Mode</span>
                    </>
                  )}
                </DropdownMenuItem>
              </>
            )}

            <DropdownMenuSeparator />
            <DropdownMenuItem
              onClick={handleSignOut}
              className="cursor-pointer gap-2 text-red-600 focus:text-red-600 dark:text-red-400"
            >
              <LogOut className="w-4 h-4" />
              <span>Sign Out</span>
            </DropdownMenuItem>
          </DropdownMenuContent>
        </DropdownMenu>
      </div>

      {/* New Query Button */}
      <div className="px-5 py-4">
        <Button
          onClick={handleNewQuery}
          data-testid="new-query-button"
          className="w-full bg-gradient-to-r from-slate-900 to-slate-800 hover:from-slate-800 hover:to-slate-700 text-white dark:from-slate-100 dark:to-slate-200 dark:text-slate-900 dark:hover:from-slate-200 dark:hover:to-slate-300 shadow-sm rounded-xl h-10 font-semibold transition-all duration-200"
        >
          <PlusCircle className="w-4 h-4 mr-2" strokeWidth={2} />
          New Query
        </Button>
      </div>

      {/* Navigation (Search Query removed as requested) */}
      <nav className="flex-1 px-3 space-y-1 overflow-y-auto">
        {navItems.map((item) => {
          const Icon = item.icon;
          const isActive = location.pathname === item.path;
          return (
            <Link
              key={item.path}
              to={item.path}
              data-testid={`nav-${item.label.toLowerCase().replace(/\s+/g, '-')}`}
              className={cn(
                'flex items-center gap-3 px-3.5 py-2.5 rounded-xl font-medium text-sm transition-all duration-200',
                isActive
                  ? 'bg-slate-100 dark:bg-slate-800/80 text-cyan-600 dark:text-cyan-400 shadow-sm font-semibold'
                  : 'text-slate-600 dark:text-slate-400 hover:bg-slate-100/60 dark:hover:bg-slate-800/40 hover:text-slate-900 dark:hover:text-white'
              )}
            >
              <Icon className="w-4 h-4" strokeWidth={1.75} />
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
        <div className="space-y-2 mb-3">
          <div className="flex items-center gap-2">
            <div className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
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
