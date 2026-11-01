import { useState, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Search, Send, ChevronRight, ChevronLeft, Sparkles,
  FileText, ExternalLink, Eye, Calendar, Tag, Loader2,
  AlertCircle, X, Download, ZoomIn, ZoomOut, RotateCw
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Badge } from '@/components/ui/badge';
import { Card } from '@/components/ui/card';
import { ScrollArea } from '@/components/ui/scroll-area';
import { useAuth } from '@/contexts/AuthContext';
import { useSearchParams } from 'react-router-dom';
import RichAnswer from '@/components/RichAnswer';

const API_BASE = process.env.REACT_APP_API_URL || 'http://localhost:8000';

// ── API helper ─────────────────────────────────────────────────────────────────
async function chatWithLLM(query, token, history = [], activeDocIds = []) {
  const payload = {
    query,
    limit: 5,
    score_threshold: 0.10,
    history: history.length > 0 ? history : undefined,
    active_document_ids: activeDocIds.length > 0 ? activeDocIds : undefined,
  };

  const res = await fetch(`${API_BASE}/chat`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${token}`,
    },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `Chat failed (${res.status})`);
  }
  return res.json();
}

async function fetchPreviewUrl(documentId, token) {
  const res = await fetch(`${API_BASE}/documents/${documentId}/preview`, {
    headers: { Authorization: `Bearer ${token}` },
  });
  if (!res.ok) throw new Error('Could not load preview');
  const data = await res.json();
  return data.preview_url;
}

// ── Score → colour helper ──────────────────────────────────────────────────────
function scoreColor(score) {
  if (score >= 0.75) return 'text-emerald-600 dark:text-emerald-400';
  if (score >= 0.50) return 'text-amber-600 dark:text-amber-400';
  return 'text-slate-500 dark:text-slate-400';
}

function scoreBg(score) {
  if (score >= 0.75) return 'bg-emerald-50 dark:bg-emerald-900/20 border-emerald-200 dark:border-emerald-800';
  if (score >= 0.50) return 'bg-amber-50 dark:bg-amber-900/20 border-amber-200 dark:border-amber-800';
  return 'bg-slate-50 dark:bg-slate-800/50 border-slate-200 dark:border-slate-700';
}



// ── Result Card ────────────────────────────────────────────────────────────────
function ResultCard({ result, rank, token, onPreview }) {
  const scorePercent = Math.round(result.score * 100);
  let rawType = result.document_type || 'document';
  if ((result.filename || '').toLowerCase().includes('report') && (rawType === 'research_paper' || rawType === 'academic_paper')) {
    rawType = 'report';
  }
  const prettyMap = {
    report: 'Project Report',
    research_paper: 'Research Paper',
    student_id_card: 'Student ID Card',
    secondary_school_certificate: '10th / 12th Marksheet',
    aadhaar: 'Aadhaar',
    pan_card: 'PAN Card',
    electricity_bill: 'Electricity Bill',
    mobile_bill: 'Mobile Bill',
    bank_statement: 'Bank Statement',
  };
  const docTypePretty = prettyMap[rawType] || rawType
    .replace(/_/g, ' ')
    .replace(/\b\w/g, (c) => c.toUpperCase());

  const summaryText = ((result.filename || '').toLowerCase().includes('report') && result.summary)
    ? result.summary.replace(/\ba research paper\b/gi, 'a project report')
    : result.summary;

  return (
    <Card className={`p-4 border transition-all hover:shadow-md overflow-hidden ${scoreBg(result.score)}`}>
      <div className="space-y-3 min-w-0">
        {/* Header row */}
        <div className="flex items-start justify-between gap-2">
          <Badge className="bg-cyan-100 dark:bg-cyan-900/30 text-cyan-700 dark:text-cyan-400 border-0 text-xs">
            #{rank}
          </Badge>
          <span className={`text-xs font-bold ${scoreColor(result.score)}`}>
            {scorePercent}% match
          </span>
        </div>

        {/* Filename */}
        <div className="min-w-0">
          <p className="text-xs font-medium text-slate-500 dark:text-slate-400 mb-0.5">Document</p>
          <p className="text-sm font-semibold text-slate-900 dark:text-white leading-tight break-all line-clamp-2">
            {result.filename}
          </p>
        </div>

        {/* Doc type + date */}
        <div className="flex flex-wrap gap-1.5">
          <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs
                           bg-violet-100 dark:bg-violet-900/30 text-violet-700 dark:text-violet-300">
            <Tag className="w-3 h-3" />
            {docTypePretty}
          </span>
          {(result.year || result.month) && (
            <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs
                             bg-slate-100 dark:bg-slate-800 text-slate-600 dark:text-slate-300">
              <Calendar className="w-3 h-3" />
              {[result.month, result.year].filter(Boolean).join(' ')}
            </span>
          )}
        </div>

        {/* Summary */}
        {summaryText && (
          <div className="min-w-0">
            <p className="text-xs font-medium text-slate-500 dark:text-slate-400 mb-0.5">Summary</p>
            <p className="text-xs text-slate-600 dark:text-slate-300 line-clamp-3 leading-relaxed break-words">
              {summaryText}
            </p>
          </div>
        )}

        {/* Best chunk preview */}
        {result.matched_chunks?.[0]?.text && (
          <div className="min-w-0">
            <p className="text-xs font-medium text-slate-500 dark:text-slate-400 mb-0.5">Matched text</p>
            <p className="text-xs text-slate-600 dark:text-slate-300 line-clamp-2 leading-relaxed italic break-words">
              "…{result.matched_chunks[0].text.slice(0, 180)}…"
            </p>
          </div>
        )}

        {/* Action button */}
        <Button
          onClick={() => onPreview(result)}
          size="sm"
          className="w-full text-xs bg-gradient-to-r from-cyan-600 to-cyan-500
                     hover:from-cyan-500 hover:to-cyan-600 text-white gap-1.5"
        >
          <Eye className="w-3.5 h-3.5" />
          View PDF
        </Button>
      </div>
    </Card>
  );
}

// ── Main Component ─────────────────────────────────────────────────────────────
const QueryInterface = () => {
  const { getAccessToken } = useAuth();
  const [searchParams, setSearchParams] = useSearchParams();
  
  // Persist chat state across browser refreshes using sessionStorage
  const [query, setQuery] = useState(() => {
    return sessionStorage.getItem('qi_query') || searchParams.get('q') || '';
  });
  const [messages, setMessages] = useState(() => {
    try { return JSON.parse(sessionStorage.getItem('qi_messages') || '[]'); }
    catch { return []; }
  });
  const [searchResults, setSearchResults] = useState(() => {
    try { return JSON.parse(sessionStorage.getItem('qi_results') || '[]'); }
    catch { return []; }
  });
  const [showRightPanel, setShowRightPanel] = useState(true);
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef(null);

  // Keep track of the last searched query to avoid double submission on param updates
  const lastSearchedQueryRef = useRef(sessionStorage.getItem('qi_last_searched_query') || '');

  // Mirror of messages state — updated synchronously so history saves are accurate
  const messagesRef = useRef(
    (() => { try { return JSON.parse(sessionStorage.getItem('qi_messages') || '[]'); } catch { return []; } })()
  );

  // Session ID — one per chat session, cleared by "New Query"
  const sessionIdRef = useRef(
    sessionStorage.getItem('qi_session_id') || (() => {
      const id = `s_${Date.now()}`;
      sessionStorage.setItem('qi_session_id', id);
      return id;
    })()
  );

  // Persist messages to sessionStorage whenever they change
  useEffect(() => {
    sessionStorage.setItem('qi_messages', JSON.stringify(messages));
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Persist search results to sessionStorage
  useEffect(() => {
    sessionStorage.setItem('qi_results', JSON.stringify(searchResults));
  }, [searchResults]);

  // Persist query input to sessionStorage
  useEffect(() => {
    sessionStorage.setItem('qi_query', query);
  }, [query]);

  // Execute search if query param is present or updated (e.g. from history click)
  useEffect(() => {
    const q = searchParams.get('q');
    if (q && q !== lastSearchedQueryRef.current) {
      lastSearchedQueryRef.current = q;
      sessionStorage.setItem('qi_last_searched_query', q);
      setQuery(q);
      // Wait for React to update the state, then execute search
      setTimeout(() => {
        const sendBtn = document.getElementById('send-button');
        if (sendBtn && !sendBtn.disabled) {
          sendBtn.click();
        }
      }, 100);
    }
  }, [searchParams]);

  const handlePreviewClick = async (result) => {
    // Open tab immediately to bypass popup blocker
    const newWindow = window.open('', '_blank');
    if (newWindow) {
      newWindow.document.write('<div style="font-family: sans-serif; padding: 2rem; text-align: center; color: #666;">Loading secure document...</div>');
    }
    
    try {
      const url = await fetchPreviewUrl(result.document_id, getAccessToken());
      if (newWindow) {
        newWindow.location.href = url;
      } else {
        window.location.href = url;
      }
    } catch (e) {
      console.error(e);
      if (newWindow) newWindow.close();
      alert("Failed to load document preview");
    }
  };

  const handleSendQuery = async () => {
    const q = query.trim();
    if (!q || isLoading) return;

    const token = getAccessToken();
    if (!token) return;

    // Update the last searched query ref and session storage to prevent overwrite loops
    lastSearchedQueryRef.current = q;
    sessionStorage.setItem('qi_last_searched_query', q);

    // Build user message and update ref + state synchronously
    const userMessage = { id: Date.now(), type: 'user', content: q, timestamp: new Date().toLocaleTimeString() };
    const messagesWithUser = [...messagesRef.current, userMessage];
    messagesRef.current = messagesWithUser;
    setMessages(messagesWithUser);

    // Persist query in URL params
    setSearchParams({ q });

    setQuery('');
    setIsLoading(true);
    setSearchResults([]);

    // ── Greeting shortcut (no network round-trip needed) ─────────────────────
    const cleanQuery = q.toLowerCase().replace(/[?!.]/g, '').trim();
    const greetings = ['hi', 'hello', 'hey', 'greetings', 'good morning', 'good afternoon', 'good evening', 'yo', 'sup', 'hi there', 'hello there', 'help', 'how are you', 'what can you do', 'who are you', 'test'];
    if (greetings.includes(cleanQuery)) {
      const greetContent = (cleanQuery === 'help' || cleanQuery === 'what can you do' || cleanQuery === 'who are you')
        ? "I am your Document Intelligence assistant! You can upload documents (PDFs, JPGs, PNGs) and ask me anything about them.\n\nFor example:\n- \"Summarize my Aadhaar card\"\n- \"What is my electricity bill amount for January 2024?\"\n- \"Find my latest bank statement\"\n\nHow can I help you today?"
        : "Hello! I'm your Document Intelligence assistant powered by Qwen AI. Ask me anything about your uploaded documents!";
      const greetMsg = { id: Date.now() + 1, type: 'ai', content: greetContent, resultCount: 0, parsedQuery: null, latency: null, timestamp: new Date().toLocaleTimeString() };
      const finalMessages = [...messagesRef.current, greetMsg];
      messagesRef.current = finalMessages;
      setMessages(finalMessages);

      // Persist greeting in history as well
      const entryId = `h_${Date.now()}_${Math.random().toString(36).substring(2, 7)}`;
      const historyEntry = {
        id: entryId,
        query: q,
        timestamp: new Date().toISOString(),
        resultCount: 0,
        topFile: null,
        snapshot: { messages: finalMessages, results: [] },
      };
      try {
        const existing = JSON.parse(localStorage.getItem('qi_history') || '[]');
        const updated = [historyEntry, ...existing.filter((e) => e.id !== entryId)].slice(0, 200);
        localStorage.setItem('qi_history', JSON.stringify(updated));
        window.dispatchEvent(new Event('qi_history_updated'));
      } catch { /* ignore storage errors */ }

      if (token) {
        fetch(`${API_BASE}/history`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            Authorization: `Bearer ${token}`,
          },
          body: JSON.stringify(historyEntry),
        }).catch((err) => console.error('Failed to sync greeting history:', err));
      }

      setIsLoading(false);
      return;
    }

    try {
      // Build conversation history for multi-turn GPT-like continuity
      const historyPayload = messagesRef.current
        .filter((m) => m.type === 'user' || m.type === 'ai')
        .slice(-6)
        .map((m) => ({
          role: m.type === 'user' ? 'user' : 'assistant',
          content: m.content || '',
        }));

      // Active document context from the currently viewed sources
      const activeDocIds = searchResults
        .map((s) => s.document_id)
        .filter(Boolean);

      // Call POST /chat — the conversational RAG + LLM pipeline
      const data = await chatWithLLM(q, token, historyPayload, activeDocIds);

      // Sources from the RAG pipeline → right panel document cards
      const sources = data.sources || [];
      setSearchResults(sources);

      // The LLM-generated answer goes directly into the chat bubble
      const aiContent = data.answer || (sources.length === 0
        ? `I couldn't find any relevant documents for "${q}". Try uploading more documents or rephrasing your query.`
        : data.answer);

      const aiMessage = {
        id: Date.now() + 1,
        type: 'ai',
        content: aiContent,
        resultCount: sources.length,
        parsedQuery: data.parsed_query,
        dynamicK: data.dynamic_k,
        latency: data.latency_ms,
        timestamp: new Date().toLocaleTimeString(),
      };

      const finalMessages = [...messagesRef.current, aiMessage];
      messagesRef.current = finalMessages;
      setMessages(finalMessages);

      // ── Upsert query entry in history ──────────────────────────────────
      const entryId = `h_${Date.now()}_${Math.random().toString(36).substring(2, 7)}`;
      const historyEntry = {
        id: entryId,
        query: q,
        timestamp: new Date().toISOString(),
        resultCount: sources.length,
        topFile: sources[0]?.filename || null,
        snapshot: { messages: finalMessages, results: sources },
      };

      // 1. Sync to localStorage immediately
      try {
        const existing = JSON.parse(localStorage.getItem('qi_history') || '[]');
        const updated = [historyEntry, ...existing.filter((e) => e.id !== entryId)].slice(0, 200);
        localStorage.setItem('qi_history', JSON.stringify(updated));
        // Dispatch local event so QueryHistory updates immediately if open or cached
        window.dispatchEvent(new Event('qi_history_updated'));
      } catch { /* ignore storage errors */ }

      // 2. Sync to Backend API for cross-device persistence
      if (token) {
        fetch(`${API_BASE}/history`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            Authorization: `Bearer ${token}`,
          },
          body: JSON.stringify(historyEntry),
        }).catch((err) => console.error('Failed to sync history to server:', err));
      }

    } catch (err) {
      const errMessage = {
        id: Date.now() + 1,
        type: 'error',
        content: err.message || 'Search failed. Please try again.',
        timestamp: new Date().toLocaleTimeString(),
      };
      const finalMessages = [...messagesRef.current, errMessage];
      messagesRef.current = finalMessages;
      setMessages(finalMessages);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="h-screen flex relative" data-testid="query-interface">
      {/* ── Main Chat Area ───────────────────────────────────────────────────── */}
      <div className="flex-1 flex flex-col min-w-0">
        {/* Top header */}
        <div className="px-6 py-5 border-b border-slate-200 dark:border-slate-800 bg-white/60 dark:bg-slate-900/60 backdrop-blur-sm">
          <div className="max-w-3xl mx-auto flex items-center gap-3">
            <div className="p-2 rounded-xl bg-gradient-to-br from-cyan-500 to-cyan-600">
              <Sparkles className="w-5 h-5 text-white" strokeWidth={1.5} />
            </div>
            <div>
              <h1 className="text-base font-semibold text-slate-900 dark:text-white">Document Intelligence</h1>
              <p className="text-xs text-slate-500 dark:text-slate-400">Ask anything — your PDFs will be returned</p>
            </div>
          </div>
        </div>

        {/* Messages */}
        <ScrollArea className="flex-1 px-6 py-6">
          <div className="max-w-3xl mx-auto space-y-5">
            {messages.length === 0 ? (
              <div className="flex flex-col items-center justify-center py-24" data-testid="empty-state">
                <div className="p-5 rounded-2xl bg-slate-100 dark:bg-slate-800 mb-5">
                  <Search className="w-10 h-10 text-cyan-500" strokeWidth={1.5} />
                </div>
                <h2 className="text-xl font-semibold text-slate-900 dark:text-white mb-2">
                  Search your documents
                </h2>
                <p className="text-slate-500 dark:text-slate-400 text-center text-sm max-w-sm">
                  Try: "Give me SSC marksheet", "Show SBI bank statement from January 2024",
                  or "Find insurance papers"
                </p>
                <div className="flex flex-wrap gap-2 mt-5 justify-center">
                  {[
                    'Show my latest bank statement',
                    'Find electricity bill 2024',
                    'Medical report with diabetes',
                  ].map((s) => (
                    <button
                      key={s}
                      onClick={() => setQuery(s)}
                      className="px-3 py-1.5 text-xs rounded-full border border-slate-200 dark:border-slate-700
                                 bg-white dark:bg-slate-800 text-slate-600 dark:text-slate-300
                                 hover:border-cyan-400 hover:text-cyan-600 transition-colors"
                    >
                      {s}
                    </button>
                  ))}
                </div>
              </div>
            ) : (
              <AnimatePresence initial={false}>
                {messages.map((message) => (
                  <motion.div
                    key={message.id}
                    initial={{ opacity: 0, y: 8 }}
                    animate={{ opacity: 1, y: 0 }}
                    className={`flex ${message.type === 'user' ? 'justify-end' : 'justify-start'}`}
                  >
                    {message.type === 'user' ? (
                      <div className="bg-slate-100 dark:bg-slate-800 text-slate-900 dark:text-slate-100
                                      rounded-2xl rounded-tr-sm px-4 py-3 max-w-[80%] text-sm"
                           data-testid="user-message">
                        {message.content}
                      </div>
                    ) : message.type === 'error' ? (
                      <div className="flex items-start gap-2 max-w-[90%]">
                        <AlertCircle className="w-4 h-4 text-red-500 mt-0.5 flex-shrink-0" />
                        <p className="text-sm text-red-500">{message.content}</p>
                      </div>
                    ) : (
                      <div className="w-full space-y-3" data-testid="ai-message">
                        <div className="text-sm text-slate-700 dark:text-slate-200 leading-relaxed">
                          <RichAnswer content={message.content} />
                        </div>
                      </div>
                    )}
                  </motion.div>
                ))}
                {/* Loading indicator */}
                {isLoading && (
                  <motion.div
                    initial={{ opacity: 0, y: 8 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="flex items-center gap-3 text-slate-500 dark:text-slate-400"
                  >
                    <Loader2 className="w-4 h-4 animate-spin text-cyan-500" />
                    <span className="text-sm">Thinking… reading your documents</span>
                  </motion.div>
                )}
              </AnimatePresence>
            )}
            <div ref={messagesEndRef} />
          </div>
        </ScrollArea>

        {/* Bottom Input */}
        <div className="px-6 py-5 border-t border-slate-200 dark:border-slate-800
                        bg-white/60 dark:bg-slate-900/60 backdrop-blur-sm">
          <div className="max-w-3xl mx-auto">
            <div className="relative">
              <Input
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                onKeyDown={(e) => e.key === 'Enter' && !e.shiftKey && handleSendQuery()}
                placeholder="e.g. Give me SSC marksheet, Show SBI bank statement from Jan 2024…"
                disabled={isLoading}
                className="h-12 pl-5 pr-12 rounded-xl border-slate-200 dark:border-slate-800
                           focus:ring-2 focus:ring-cyan-500 text-sm disabled:opacity-60 bg-white dark:bg-slate-900 shadow-sm"
                data-testid="query-input"
              />
              <Button
                onClick={handleSendQuery}
                disabled={!query.trim() || isLoading}
                className="absolute right-2 top-1/2 -translate-y-1/2 w-8 h-8 p-0 rounded-lg
                           bg-gradient-to-r from-cyan-600 to-cyan-500
                           hover:from-cyan-500 hover:to-cyan-600 text-white shadow-sm flex items-center justify-center disabled:opacity-40"
                data-testid="send-button"
              >
                {isLoading
                  ? <Loader2 className="w-4 h-4 animate-spin" />
                  : <Send className="w-4 h-4" strokeWidth={2} />
                }
              </Button>
            </div>
          </div>
        </div>
      </div>

      {/* ── Right Panel — Retrieved Documents ────────────────────────────────── */}
      <AnimatePresence>
        {showRightPanel && (
          <motion.div
            initial={{ width: 0, opacity: 0 }}
            animate={{ width: 340, opacity: 1 }}
            exit={{ width: 0, opacity: 0 }}
            transition={{ type: 'spring', stiffness: 300, damping: 30 }}
            className="border-l border-slate-200 dark:border-slate-800
                       bg-white/80 dark:bg-slate-900/80 backdrop-blur-xl overflow-hidden flex-shrink-0"
            data-testid="right-panel"
          >
            <div className="h-full flex flex-col w-[340px]">
              {/* Panel header */}
              <div className="px-5 py-4 border-b border-slate-200 dark:border-slate-800 flex items-center justify-between">
                <h3 className="text-sm font-semibold text-slate-900 dark:text-white flex items-center gap-2">
                  <FileText className="w-4 h-4 text-cyan-500" strokeWidth={1.5} />
                  AI Sources
                </h3>
                {searchResults.length > 0 && (
                  <Badge className="bg-cyan-100 dark:bg-cyan-900/30 text-cyan-700 dark:text-cyan-400 border-0 text-xs">
                    {searchResults.length}
                  </Badge>
                )}
              </div>

              <ScrollArea className="flex-1 px-4 py-4 overflow-x-hidden">
                <div className="space-y-3 w-full">
                  {searchResults.length === 0 && !isLoading && (
                    <div className="flex flex-col items-center justify-center py-12 text-center">
                      <FileText className="w-8 h-8 text-slate-300 dark:text-slate-600 mb-3" strokeWidth={1} />
                      <p className="text-xs text-slate-400 dark:text-slate-500">
                        Results will appear here after you search
                      </p>
                    </div>
                  )}
                  {isLoading && (
                    <div className="flex flex-col items-center justify-center py-12 gap-3">
                      <Loader2 className="w-6 h-6 text-cyan-500 animate-spin" />
                      <p className="text-xs text-slate-400">Searching…</p>
                    </div>
                  )}
                  <AnimatePresence>
                    {searchResults.map((result, idx) => (
                      <motion.div
                        key={result.document_id}
                        initial={{ opacity: 0, y: 10 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: idx * 0.05 }}
                      >
                        <ResultCard
                          result={result}
                          rank={idx + 1}
                          token={getAccessToken()}
                          onPreview={handlePreviewClick}
                        />
                      </motion.div>
                    ))}
                  </AnimatePresence>
                </div>
              </ScrollArea>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* ── Toggle Right Panel ────────────────────────────────────────────────── */}
      <button
        onClick={() => setShowRightPanel((p) => !p)}
        className="absolute right-0 top-1/2 -translate-y-1/2 translate-x-0
                   w-6 h-12 flex items-center justify-center
                   bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700
                   rounded-l-lg shadow-md hover:shadow-lg text-slate-500 dark:text-slate-400
                   transition-all z-10"
        data-testid="toggle-right-panel"
        style={{ right: showRightPanel ? '340px' : '0px' }}
      >
        {showRightPanel
          ? <ChevronRight className="w-3 h-3" />
          : <ChevronLeft className="w-3 h-3" />
        }
      </button>

    </div>
  );
};

export default QueryInterface;
