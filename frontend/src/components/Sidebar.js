import { Link, useLocation } from 'react-router-dom';
import { Brain, LayoutDashboard, Upload, FolderOpen, History, Settings, PlusCircle, Activity } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { cn } from '@/lib/utils';

const Sidebar = () => {
  const location = useLocation();

  const navItems = [
    { path: '/', icon: LayoutDashboard, label: 'Dashboard' },
    { path: '/upload', icon: Upload, label: 'Upload Documents' },
    { path: '/collections', icon: FolderOpen, label: 'My Collections' },
    { path: '/history', icon: History, label: 'Query History' },
    { path: '/settings', icon: Settings, label: 'Settings' },
  ];

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
            <h1 className="text-xl font-bold text-slate-900 dark:text-white tracking-tight">DocuMind AI</h1>
            <p className="text-xs text-slate-500 dark:text-slate-400 font-medium">Intelligent Search</p>
          </div>
        </Link>
      </div>

      {/* New Query Button */}
      <div className="px-6 mb-6">
        <Link to="/query" data-testid="new-query-button">
          <Button
            className="w-full bg-gradient-to-r from-slate-900 to-slate-800 hover:from-slate-800 hover:to-slate-700 text-white dark:from-slate-100 dark:to-slate-200 dark:text-slate-900 dark:hover:from-slate-200 dark:hover:to-slate-300 shadow-sm rounded-lg h-11 font-semibold transition-all duration-200"
          >
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
      <div className="px-6 py-6 border-t border-slate-200/50 dark:border-slate-800/50" data-testid="system-status">
        <div className="space-y-2">
          <div className="flex items-center gap-2">
            <div className="w-2 h-2 rounded-full bg-green-500 animate-pulse"></div>
            <span className="text-xs font-medium text-slate-600 dark:text-slate-400">Qdrant Connected</span>
          </div>
          <div className="flex items-center gap-2">
            <Activity className="w-3 h-3 text-cyan-500" strokeWidth={2} />
            <span className="text-xs font-medium text-slate-600 dark:text-slate-400">LLM Active</span>
          </div>
        </div>
      </div>
    </aside>
  );
};

export default Sidebar;
