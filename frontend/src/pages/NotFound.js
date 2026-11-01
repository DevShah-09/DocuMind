import { useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import { FileQuestion, ArrowLeft, Search, FolderOpen, Upload, Home } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent } from '@/components/ui/card';

const NotFound = () => {
  const navigate = useNavigate();

  return (
    <div
      className="min-h-full flex items-center justify-center p-6 md:p-12"
      data-testid="not-found-page"
    >
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.4 }}
        className="max-w-xl w-full text-center"
      >
        {/* Visual 404 Badge & Graphic */}
        <div className="relative inline-block mb-6">
          <div className="w-28 h-28 mx-auto rounded-3xl bg-gradient-to-tr from-cyan-500/20 via-blue-500/10 to-violet-500/20 dark:from-cyan-500/10 dark:via-blue-500/5 dark:to-violet-500/10 border border-cyan-500/30 flex items-center justify-center relative shadow-xl shadow-cyan-500/5">
            <motion.div
              animate={{ rotate: [0, -5, 5, -5, 0] }}
              transition={{ repeat: Infinity, duration: 6, ease: 'easeInOut' }}
            >
              <FileQuestion className="w-14 h-14 text-cyan-500" strokeWidth={1.5} />
            </motion.div>
          </div>
          <div className="absolute -bottom-2 -right-2 px-3 py-1 rounded-full bg-gradient-to-r from-cyan-600 to-blue-600 text-white font-mono font-bold text-xs shadow-md">
            404
          </div>
        </div>

        {/* Heading & description */}
        <h1 className="text-3xl md:text-4xl font-bold text-slate-900 dark:text-white tracking-tight mb-3">
          Page Not Found
        </h1>
        <p className="text-base text-slate-600 dark:text-slate-400 max-w-md mx-auto mb-8">
          The link you followed doesn't exist, has been moved, or is temporarily unavailable in DocuMind AI.
        </p>

        {/* Primary Action Buttons */}
        <div className="flex flex-wrap items-center justify-center gap-3 mb-8">
          <Button
            onClick={() => navigate(-1)}
            variant="outline"
            className="border-slate-200 dark:border-slate-800 hover:bg-slate-100 dark:hover:bg-slate-800"
          >
            <ArrowLeft className="w-4 h-4 mr-2" />
            Go Back
          </Button>

          <Button
            onClick={() => navigate('/query')}
            className="bg-gradient-to-r from-cyan-600 to-blue-600 hover:from-cyan-500 hover:to-blue-500 text-white shadow-md"
          >
            <Home className="w-4 h-4 mr-2" />
            Go to Query
          </Button>
        </div>

        {/* Quick Links Card */}
        <Card className="border-slate-200 dark:border-slate-800 bg-white/50 dark:bg-slate-900/50 backdrop-blur-sm text-left">
          <CardContent className="p-4 sm:p-6">
            <p className="text-xs font-semibold uppercase tracking-wider text-slate-400 dark:text-slate-500 mb-3">
              Helpful Destinations
            </p>
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-2">
              <button
                onClick={() => navigate('/query')}
                className="flex items-center gap-2 p-2.5 rounded-lg border border-slate-100 dark:border-slate-800 hover:bg-slate-50 dark:hover:bg-slate-800/60 transition-colors group text-left"
              >
                <div className="p-1.5 rounded-md bg-cyan-100 dark:bg-cyan-900/30 text-cyan-600 dark:text-cyan-400">
                  <Search className="w-4 h-4" />
                </div>
                <div className="min-w-0">
                  <p className="text-xs font-medium text-slate-800 dark:text-slate-200 group-hover:text-cyan-600 dark:group-hover:text-cyan-400 transition-colors">
                    Search AI
                  </p>
                  <p className="text-[10px] text-slate-400 truncate">Ask questions</p>
                </div>
              </button>

              <button
                onClick={() => navigate('/collections')}
                className="flex items-center gap-2 p-2.5 rounded-lg border border-slate-100 dark:border-slate-800 hover:bg-slate-50 dark:hover:bg-slate-800/60 transition-colors group text-left"
              >
                <div className="p-1.5 rounded-md bg-violet-100 dark:bg-violet-900/30 text-violet-600 dark:text-violet-400">
                  <FolderOpen className="w-4 h-4" />
                </div>
                <div className="min-w-0">
                  <p className="text-xs font-medium text-slate-800 dark:text-slate-200 group-hover:text-violet-600 dark:group-hover:text-violet-400 transition-colors">
                    Collections
                  </p>
                  <p className="text-[10px] text-slate-400 truncate">Your documents</p>
                </div>
              </button>

              <button
                onClick={() => navigate('/upload')}
                className="flex items-center gap-2 p-2.5 rounded-lg border border-slate-100 dark:border-slate-800 hover:bg-slate-50 dark:hover:bg-slate-800/60 transition-colors group text-left"
              >
                <div className="p-1.5 rounded-md bg-blue-100 dark:bg-blue-900/30 text-blue-600 dark:text-blue-400">
                  <Upload className="w-4 h-4" />
                </div>
                <div className="min-w-0">
                  <p className="text-xs font-medium text-slate-800 dark:text-slate-200 group-hover:text-blue-600 dark:group-hover:text-blue-400 transition-colors">
                    Upload
                  </p>
                  <p className="text-[10px] text-slate-400 truncate">Add new files</p>
                </div>
              </button>
            </div>
          </CardContent>
        </Card>
      </motion.div>
    </div>
  );
};

export default NotFound;

