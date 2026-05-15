import { useState, useEffect, useRef } from 'react';
import { motion } from 'framer-motion';
import { Clock, MessageSquare, FileText, ChevronRight, Search, Trash2, Pencil, Check, X } from 'lucide-react';
import { Card, CardContent } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { ScrollArea } from '@/components/ui/scroll-area';
import { useNavigate } from 'react-router-dom';

// ── Time-ago helper ──────────────────────────────────────────────────────────
function timeAgo(isoString) {
  const diff = Date.now() - new Date(isoString).getTime();
  const mins  = Math.floor(diff / 60_000);
  const hours = Math.floor(diff / 3_600_000);
  const days  = Math.floor(diff / 86_400_000);
  if (mins  < 1)  return 'Just now';
  if (mins  < 60) return `${mins} minute${mins > 1 ? 's' : ''} ago`;
  if (hours < 24) return `${hours} hour${hours > 1 ? 's' : ''} ago`;
  if (days  < 7)  return `${days} day${days > 1 ? 's' : ''} ago`;
  return new Date(isoString).toLocaleDateString();
}

// ── Inline rename component ───────────────────────────────────────────────────
function InlineRename({ value, onSave, onCancel }) {
  const [draft, setDraft] = useState(value);
  const inputRef = useRef(null);

  useEffect(() => { inputRef.current?.focus(); inputRef.current?.select(); }, []);

  const commit = () => {
    const trimmed = draft.trim();
    if (trimmed && trimmed !== value) onSave(trimmed);
    else onCancel();
  };

  return (
    <div className="flex items-center gap-2 flex-1" onClick={(e) => e.stopPropagation()}>
      <input
        ref={inputRef}
        value={draft}
        onChange={(e) => setDraft(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === 'Enter')  commit();
          if (e.key === 'Escape') onCancel();
        }}
        className="flex-1 text-lg font-medium text-slate-900 dark:text-white bg-transparent border-b-2 border-cyan-500 outline-none py-0.5"
      />
      <button
        onClick={commit}
        className="p-1 rounded text-emerald-600 hover:bg-emerald-50 dark:hover:bg-emerald-900/20"
        title="Save"
      >
        <Check className="w-4 h-4" />
      </button>
      <button
        onClick={onCancel}
        className="p-1 rounded text-slate-400 hover:bg-slate-100 dark:hover:bg-slate-800"
        title="Cancel"
      >
        <X className="w-4 h-4" />
      </button>
    </div>
  );
}

// ── Main Component ────────────────────────────────────────────────────────────
const QueryHistory = () => {
  const navigate = useNavigate();
  const [history, setHistory] = useState([]);
  const [editingId, setEditingId] = useState(null); // id of item being renamed

  // Load history from localStorage on mount
  useEffect(() => {
    try {
      const stored = JSON.parse(localStorage.getItem('qi_history') || '[]');
      setHistory(stored);
    } catch {
      setHistory([]);
    }
  }, []);

  const persist = (updated) => {
    setHistory(updated);
    try { localStorage.setItem('qi_history', JSON.stringify(updated)); } catch { /* ignore */ }
  };

  const handleDeleteItem = (e, itemId) => {
    e.stopPropagation();
    persist(history.filter((h) => h.id !== itemId));
  };

  const handleRename = (itemId, newName) => {
    persist(history.map((h) => h.id === itemId ? { ...h, label: newName } : h));
    setEditingId(null);
  };

  const handleRerun = (item) => {
    if (editingId) return; // don't navigate while editing
    if (item.snapshot) {
      try {
        sessionStorage.setItem('qi_messages', JSON.stringify(item.snapshot.messages));
        sessionStorage.setItem('qi_results', JSON.stringify(item.snapshot.results));
        sessionStorage.setItem('qi_query', '');
        navigate('/query');
        return;
      } catch { /* fall through */ }
    }
    sessionStorage.removeItem('qi_messages');
    sessionStorage.removeItem('qi_results');
    sessionStorage.setItem('qi_query', item.query);
    navigate(`/query?q=${encodeURIComponent(item.query)}`);
  };

  const handleClearHistory = () => persist([]);

  return (
    <div className="px-8 py-12" data-testid="query-history">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.3 }}
      >
        {/* Header */}
        <div className="mb-12 flex items-center justify-between">
          <div>
            <h1 className="text-4xl md:text-5xl font-bold text-slate-900 dark:text-white tracking-tight mb-3">
              Query History
            </h1>
            <p className="text-lg text-slate-600 dark:text-slate-300">
              Review and re-run your previous searches
            </p>
          </div>
          {history.length > 0 && (
            <Button
              variant="outline"
              onClick={handleClearHistory}
              className="text-red-500 border-red-200 hover:bg-red-50 hover:text-red-600 dark:border-red-800 dark:hover:bg-red-900/20"
            >
              <Trash2 className="w-4 h-4 mr-2" />
              Clear History
            </Button>
          )}
        </div>

        {/* Empty State */}
        {history.length === 0 && (
          <div className="flex flex-col items-center justify-center py-24 border-2 border-dashed border-slate-200 dark:border-slate-700 rounded-xl">
            <div className="p-5 rounded-2xl bg-slate-100 dark:bg-slate-800 mb-5">
              <Search className="w-10 h-10 text-slate-400" strokeWidth={1.5} />
            </div>
            <h2 className="text-xl font-semibold text-slate-700 dark:text-slate-300 mb-2">
              No queries yet
            </h2>
            <p className="text-slate-500 dark:text-slate-400 text-sm text-center max-w-sm">
              Your search history will appear here after you run queries on the Query page.
            </p>
          </div>
        )}

        {/* History List */}
        {history.length > 0 && (
          <div className="max-w-4xl mx-auto">
            <ScrollArea className="h-[calc(100vh-250px)]">
              <div className="space-y-4">
                {history.map((item, index) => (
                  <motion.div
                    key={item.id}
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ duration: 0.3, delay: index * 0.04 }}
                  >
                    <Card
                      className="hover:shadow-lg transition-all duration-200 border-slate-200 dark:border-slate-800 cursor-pointer group"
                      data-testid={`history-item-${item.id}`}
                      onClick={() => handleRerun(item)}
                    >
                      <CardContent className="p-6">
                        <div className="flex items-start justify-between gap-4">
                          <div className="flex-1 space-y-3">
                            <div className="flex items-start gap-3">
                              <div className="p-2 rounded-lg bg-cyan-100 dark:bg-cyan-900/20 mt-1 flex-shrink-0">
                                <MessageSquare className="w-5 h-5 text-cyan-600 dark:text-cyan-400" strokeWidth={1.5} />
                              </div>
                              <div className="flex-1">
                                {/* Title — editable or display */}
                                {editingId === item.id ? (
                                  <InlineRename
                                    value={item.label || item.query}
                                    onSave={(name) => handleRename(item.id, name)}
                                    onCancel={() => setEditingId(null)}
                                  />
                                ) : (
                                  <p className="text-lg font-medium text-slate-900 dark:text-white mb-2 line-clamp-2">
                                    {item.label || item.query}
                                  </p>
                                )}

                                <div className="flex items-center gap-3 flex-wrap">
                                  <div className="flex items-center gap-1 text-sm text-slate-500 dark:text-slate-400">
                                    <Clock className="w-4 h-4" strokeWidth={1.5} />
                                    {timeAgo(item.timestamp)}
                                  </div>
                                  <Badge
                                    variant="outline"
                                    className="bg-cyan-50 dark:bg-cyan-900/20 text-cyan-700 dark:text-cyan-400 border-cyan-200 dark:border-cyan-800"
                                  >
                                    {item.resultCount} result{item.resultCount !== 1 ? 's' : ''}
                                  </Badge>
                                  {item.topFile && (
                                    <Badge
                                      variant="outline"
                                      className="bg-blue-50 dark:bg-blue-900/20 text-blue-700 dark:text-blue-400 border-blue-200 dark:border-blue-800"
                                    >
                                      <FileText className="w-3 h-3 mr-1" />
                                      {item.topFile.length > 30
                                        ? item.topFile.slice(0, 30) + '…'
                                        : item.topFile}
                                    </Badge>
                                  )}
                                </div>
                              </div>
                            </div>
                          </div>

                          {/* Action buttons */}
                          <div className="flex items-center gap-1 flex-shrink-0">
                            {/* Rename */}
                            <Button
                              variant="ghost"
                              size="icon"
                              className="opacity-0 group-hover:opacity-100 transition-opacity text-slate-400 hover:text-cyan-600 hover:bg-cyan-50 dark:hover:bg-cyan-900/20"
                              data-testid={`rename-query-${item.id}`}
                              onClick={(e) => { e.stopPropagation(); setEditingId(item.id); }}
                              title="Rename"
                            >
                              <Pencil className="w-4 h-4" />
                            </Button>
                            {/* Delete */}
                            <Button
                              variant="ghost"
                              size="icon"
                              className="opacity-0 group-hover:opacity-100 transition-opacity text-red-400 hover:text-red-600 hover:bg-red-50 dark:hover:bg-red-900/20"
                              data-testid={`delete-query-${item.id}`}
                              onClick={(e) => handleDeleteItem(e, item.id)}
                              title="Delete"
                            >
                              <Trash2 className="w-4 h-4" />
                            </Button>
                            {/* Open */}
                            <Button
                              variant="ghost"
                              size="icon"
                              className="opacity-0 group-hover:opacity-100 transition-opacity"
                              data-testid={`view-query-${item.id}`}
                            >
                              <ChevronRight className="w-5 h-5" />
                            </Button>
                          </div>
                        </div>
                      </CardContent>
                    </Card>
                  </motion.div>
                ))}
              </div>
            </ScrollArea>
          </div>
        )}
      </motion.div>
    </div>
  );
};

export default QueryHistory;
