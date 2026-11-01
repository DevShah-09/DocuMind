import { useState, useRef, useEffect, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Upload, FileText, File, CheckCircle2, Loader2,
  AlertCircle, Tag, Percent, Info, ChevronDown, ChevronUp, X, Ban
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent } from '@/components/ui/card';
import { Progress } from '@/components/ui/progress';
import { toast } from 'sonner';
import { useAuth } from '@/contexts/AuthContext';

// Map category slugs to display labels + colors
const CATEGORY_META = {
  electricity_bill:  { label: 'Electricity Bill',  color: 'text-yellow-500',  bg: 'bg-yellow-100 dark:bg-yellow-900/20' },
  water_bill:        { label: 'Water Bill',         color: 'text-blue-500',    bg: 'bg-blue-100 dark:bg-blue-900/20' },
  gas_bill:          { label: 'Gas Bill',           color: 'text-orange-500',  bg: 'bg-orange-100 dark:bg-orange-900/20' },
  mobile_bill:       { label: 'Mobile Bill',        color: 'text-purple-500',  bg: 'bg-purple-100 dark:bg-purple-900/20' },
  bank_statement:    { label: 'Bank Statement',     color: 'text-green-500',   bg: 'bg-green-100 dark:bg-green-900/20' },
  invoice:           { label: 'Invoice',            color: 'text-indigo-500',  bg: 'bg-indigo-100 dark:bg-indigo-900/20' },
  receipt:           { label: 'Receipt',            color: 'text-pink-500',    bg: 'bg-pink-100 dark:bg-pink-900/20' },
  marksheet:         { label: 'Marksheet',          color: 'text-cyan-500',    bg: 'bg-cyan-100 dark:bg-cyan-900/20' },
  certificate:       { label: 'Certificate',        color: 'text-teal-500',    bg: 'bg-teal-100 dark:bg-teal-900/20' },
  aadhaar:           { label: 'Aadhaar Card',       color: 'text-red-500',     bg: 'bg-red-100 dark:bg-red-900/20' },
  pan_card:          { label: 'PAN Card',           color: 'text-amber-500',   bg: 'bg-amber-100 dark:bg-amber-900/20' },
  passport:          { label: 'Passport',           color: 'text-blue-600',    bg: 'bg-blue-100 dark:bg-blue-900/20' },
  insurance:         { label: 'Insurance',          color: 'text-violet-500',  bg: 'bg-violet-100 dark:bg-violet-900/20' },
  medical_report:    { label: 'Medical Report',     color: 'text-rose-500',    bg: 'bg-rose-100 dark:bg-rose-900/20' },
  prescription:      { label: 'Prescription',       color: 'text-rose-400',    bg: 'bg-rose-100 dark:bg-rose-900/20' },
  legal_document:    { label: 'Legal Document',     color: 'text-slate-500',   bg: 'bg-slate-100 dark:bg-slate-800' },
  tax_document:      { label: 'Tax Document',       color: 'text-lime-600',    bg: 'bg-lime-100 dark:bg-lime-900/20' },
  resume:            { label: 'Resume / CV',        color: 'text-sky-500',     bg: 'bg-sky-100 dark:bg-sky-900/20' },
  student_id_card:   { label: 'Student Identity Card', color: 'text-indigo-600', bg: 'bg-indigo-100 dark:bg-indigo-900/20' },
  student_identity_card: { label: 'Student Identity Card', color: 'text-indigo-600', bg: 'bg-indigo-100 dark:bg-indigo-900/20' },
  miscellaneous:     { label: 'Miscellaneous',      color: 'text-gray-500',    bg: 'bg-gray-100 dark:bg-gray-800' },
};

const getCategoryMeta = (category) => {
  if (CATEGORY_META[category]) return CATEGORY_META[category];
  const prettyLabel = category
    ? category.split('_').map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(' ')
    : 'Document';
  return { label: prettyLabel, color: 'text-slate-500', bg: 'bg-slate-100 dark:bg-slate-800' };
};

const STAGES = [
  { id: 'uploading',     label: 'Uploading file...' },
  { id: 'reading',       label: 'Reading document pages...' },
  { id: 'analyzing',     label: 'Analyzing with Qwen AI...' },
  { id: 'categorizing',  label: 'Categorizing & extracting metadata...' },
  { id: 'indexing',      label: 'Indexing in vector database...' },
];

const ALLOWED_TYPES = ['application/pdf', 'image/jpeg', 'image/jpg', 'image/png'];
const ALLOWED_EXT   = ['pdf', 'jpg', 'jpeg', 'png'];
const MAX_SIZE_MB    = 20;

// ── ResultCard ───────────────────────────────────────────────────────────────
const ResultCard = ({ result, onRemove }) => {
  const [expanded, setExpanded] = useState(false);
  const meta = getCategoryMeta(result.category);
  const rawConf = result.confidence ?? 0;
  const confidence = (typeof rawConf === 'number' && rawConf <= 1.0 && rawConf > 0)
    ? Math.round(rawConf * 100)
    : Math.round(rawConf);

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.97, y: 10 }}
      animate={{ opacity: 1, scale: 1, y: 0 }}
      exit={{ opacity: 0, scale: 0.95 }}
      transition={{ duration: 0.3 }}
    >
      <Card className="border-slate-200 dark:border-slate-800 overflow-hidden">
        <CardContent className="p-0">
          {/* Top bar */}
          <div className="flex items-center gap-4 p-5">
            <div className={`p-3 rounded-xl ${meta.bg} flex-shrink-0`}>
              <CheckCircle2 className={`w-6 h-6 ${meta.color}`} />
            </div>

            <div className="flex-1 min-w-0">
              <p className="font-semibold text-slate-900 dark:text-white truncate">{result.original_name}</p>
              <div className="flex items-center gap-3 mt-1 flex-wrap">
                {/* Category badge */}
                <span className={`inline-flex items-center gap-1 text-xs font-medium px-2 py-0.5 rounded-full ${meta.bg} ${meta.color}`}>
                  <Tag className="w-3 h-3" />
                  {meta.label}
                </span>
                {/* Confidence badge */}
                <span className="inline-flex items-center gap-1 text-xs font-medium px-2 py-0.5 rounded-full bg-slate-100 dark:bg-slate-800 text-slate-600 dark:text-slate-300">
                  <Percent className="w-3 h-3" />
                  {confidence}% confidence
                </span>
              </div>
            </div>

            <div className="flex items-center gap-2 flex-shrink-0">
              <button
                onClick={() => setExpanded(!expanded)}
                className="p-1.5 rounded-lg hover:bg-slate-100 dark:hover:bg-slate-800 text-slate-500 transition-colors"
                title="Toggle details"
              >
                {expanded ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
              </button>
              <button
                onClick={() => onRemove(result.id)}
                className="p-1.5 rounded-lg hover:bg-red-50 dark:hover:bg-red-900/20 text-slate-400 hover:text-red-500 transition-colors"
                title="Dismiss"
              >
                <X className="w-4 h-4" />
              </button>
            </div>
          </div>

          {/* Confidence bar */}
          <div className="px-5 pb-3">
            <div className="h-1.5 w-full bg-slate-100 dark:bg-slate-800 rounded-full overflow-hidden">
              <motion.div
                initial={{ width: 0 }}
                animate={{ width: `${confidence}%` }}
                transition={{ duration: 0.8, ease: 'easeOut' }}
                className={`h-full rounded-full ${
                  confidence >= 80 ? 'bg-green-500' :
                  confidence >= 50 ? 'bg-yellow-500' : 'bg-red-400'
                }`}
              />
            </div>
          </div>

          {/* Expandable: summary + metadata */}
          <AnimatePresence>
            {expanded && (
              <motion.div
                initial={{ height: 0, opacity: 0 }}
                animate={{ height: 'auto', opacity: 1 }}
                exit={{ height: 0, opacity: 0 }}
                transition={{ duration: 0.25 }}
                className="overflow-hidden"
              >
                <div className="border-t border-slate-100 dark:border-slate-800 p-5 space-y-4">
                  {/* Summary */}
                  {result.summary && (
                    <div>
                      <p className="text-xs font-semibold uppercase tracking-wider text-slate-400 dark:text-slate-500 mb-1.5">
                        AI Summary
                      </p>
                      <p className="text-sm text-slate-600 dark:text-slate-300 leading-relaxed">
                        {result.summary}
                      </p>
                    </div>
                  )}

                  {/* Extracted metadata */}
                  {result.metadata && Object.keys(result.metadata).length > 0 && (
                    <div>
                      <p className="text-xs font-semibold uppercase tracking-wider text-slate-400 dark:text-slate-500 mb-2">
                        Extracted Fields
                      </p>
                      <div className="grid grid-cols-2 gap-x-6 gap-y-2">
                        {Object.entries(result.metadata).map(([key, val]) => (
                          <div key={key} className="flex flex-col">
                            <span className="text-xs text-slate-400 dark:text-slate-500 capitalize">
                              {key.replace(/_/g, ' ')}
                            </span>
                            <span className="text-sm font-medium text-slate-800 dark:text-slate-200 truncate">
                              {String(val)}
                            </span>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </CardContent>
      </Card>
    </motion.div>
  );
};

// ── Main Component ───────────────────────────────────────────────────────────
const MAX_CONCURRENT_UPLOADS = 3;

const UploadDocuments = ({ queue, setQueue, onUploadComplete }) => {
  const [dragActive, setDragActive] = useState(false);
  const { getAccessToken }          = useAuth();
  const fileInputRef                = useRef(null);
  const activeProcessingRef         = useRef(new Set());
  // Map of itemId -> { abortController, eventSource } for cancellation
  const cancellationRef             = useRef({});

  const isProcessing = queue.some(item => item.status === 'processing');

  // ── Cancel handler ─────────────────────────────────────────────────────────
  const handleCancel = useCallback((itemId) => {
    const handles = cancellationRef.current[itemId];
    if (handles) {
      // Abort the fetch if still in-flight
      if (handles.abortController) {
        handles.abortController.abort();
      }
      // Close SSE stream if open
      if (handles.eventSource) {
        handles.eventSource.close();
        handles.eventSource = null;
      }
      delete cancellationRef.current[itemId];
    }
    activeProcessingRef.current.delete(itemId);
    setQueue(prev =>
      prev.map(item =>
        item.id === itemId
          ? { ...item, status: 'cancelled', error: 'Upload cancelled' }
          : item
      )
    );
    toast.info('Upload cancelled.');
  }, [setQueue]);

  const processQueueItem = useCallback(async (itemToProcess) => {
    // Mark item as actively processing in state
    setQueue(prev => prev.map(item =>
      item.id === itemToProcess.id ? { ...item, status: 'processing', stageIndex: 0, progress: 5 } : item
    ));

    const abortController = new AbortController();
    cancellationRef.current[itemToProcess.id] = { abortController, eventSource: null };

    let activeEventSource = null;

    try {
      const formData = new FormData();
      formData.append('file', itemToProcess.file);

      const token = getAccessToken();
      const response = await fetch('http://localhost:8000/documents/upload', {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${token}`
        },
        body: formData,
        signal: abortController.signal,
      });

      if (!response.ok) {
        const errData = await response.json().catch(() => ({ detail: 'Unknown error' }));
        throw new Error(errData.detail || `Server error: ${response.status}`);
      }

      const uploadInit = await response.json();

      // If backend returned immediate synchronous result (sync fallback)
      if (uploadInit.data && !uploadInit.job_id) {
        const data = uploadInit.data;
        delete cancellationRef.current[itemToProcess.id];
        setQueue(prev => prev.map(item =>
          item.id === itemToProcess.id
            ? {
                ...item,
                status: 'done',
                stageIndex: STAGES.length - 1,
                progress: 100,
                result: {
                  id: data.id,
                  original_name: data.original_name,
                  category: data.category,
                  confidence: data.confidence,
                  summary: data.summary,
                  metadata: data.metadata || {},
                }
              }
            : item
        ));
        toast.success(`"${itemToProcess.name}" categorized as "${getCategoryMeta(data.category).label}"!`);
        if (typeof onUploadComplete === 'function') {
          onUploadComplete();
        }
        window.dispatchEvent(new CustomEvent('documind:collections_updated'));
        try {
          const ch = new BroadcastChannel('documind_events');
          ch.postMessage({ type: 'COLLECTIONS_UPDATED' });
          ch.close();
        } catch (_) {}
        return;
      }

      // Genuine SSE Pipeline Stream via EventSource
      const jobId = uploadInit.job_id;
      const sseUrl = `http://localhost:8000/documents/upload/${jobId}/events?token=${encodeURIComponent(token)}`;

      await new Promise((resolve, reject) => {
        const es = new EventSource(sseUrl);
        activeEventSource = es;
        // Keep ref for cancellation
        if (cancellationRef.current[itemToProcess.id]) {
          cancellationRef.current[itemToProcess.id].eventSource = es;
        }

        es.addEventListener('milestone', (event) => {
          try {
            const payload = JSON.parse(event.data);
            setQueue(prev => prev.map(item =>
              item.id === itemToProcess.id
                ? {
                    ...item,
                    stageIndex: typeof payload.stageIndex === 'number' ? payload.stageIndex : item.stageIndex,
                    progress: typeof payload.progress === 'number' ? payload.progress : item.progress,
                  }
                : item
            ));
          } catch (e) {
            console.error('[SSE] Failed to parse milestone:', e);
          }
        });

        es.addEventListener('done', (event) => {
          try {
            const payload = JSON.parse(event.data);
            const data = payload.data || {};
            es.close();
            activeEventSource = null;
            delete cancellationRef.current[itemToProcess.id];

            setQueue(prev => prev.map(item =>
              item.id === itemToProcess.id
                ? {
                    ...item,
                    status: 'done',
                    stageIndex: STAGES.length - 1,
                    progress: 100,
                    result: {
                      id: data.id,
                      original_name: data.original_name,
                      category: data.category,
                      confidence: data.confidence,
                      summary: data.summary,
                      metadata: data.metadata || {},
                    }
                  }
                : item
            ));
            toast.success(`"${itemToProcess.name}" categorized as "${getCategoryMeta(data.category).label}"!`);
            if (typeof onUploadComplete === 'function') {
              onUploadComplete();
            }
            window.dispatchEvent(new CustomEvent('documind:collections_updated'));
            try {
              const ch = new BroadcastChannel('documind_events');
              ch.postMessage({ type: 'COLLECTIONS_UPDATED' });
              ch.close();
            } catch (_) {}
            resolve(data);
          } catch (err) {
            es.close();
            activeEventSource = null;
            reject(err);
          }
        });

        es.addEventListener('error', (event) => {
          try {
            if (event.data) {
              const payload = JSON.parse(event.data);
              es.close();
              activeEventSource = null;
              reject(new Error(payload.message || payload.error || 'Upload processing failed'));
              return;
            }
          } catch (_) {}

          // If connection was closed or encountered network terminal failure
          if (es.readyState === EventSource.CLOSED) {
            es.close();
            activeEventSource = null;
            reject(new Error('Connection to upload stream interrupted.'));
          }
        });
      });

    } catch (err) {
      if (activeEventSource) {
        activeEventSource.close();
        activeEventSource = null;
      }
      delete cancellationRef.current[itemToProcess.id];

      // If aborted by user cancel, skip setting error state (handleCancel already set it)
      if (err.name === 'AbortError') return;

      console.error(err);
      const msg = err.message.includes('fetch')
        ? 'Cannot reach backend. Is the server running on port 8000?'
        : err.message;

      setQueue(prev => prev.map(item =>
        item.id === itemToProcess.id
          ? { ...item, status: 'error', error: msg }
          : item
      ));
      toast.error(`Failed to upload "${itemToProcess.name}": ${msg}`);
    } finally {
      // Small pause so user sees completion state
      await new Promise(resolve => setTimeout(resolve, 600));
      activeProcessingRef.current.delete(itemToProcess.id);
      // Trigger next tick for any remaining pending items
      setQueue(prev => [...prev]);
    }
  }, [getAccessToken, setQueue, onUploadComplete]);

  // Parallel processing loop: dispatch up to MAX_CONCURRENT_UPLOADS items simultaneously
  useEffect(() => {
    const currentlyProcessing = activeProcessingRef.current.size;
    const availableSlots = MAX_CONCURRENT_UPLOADS - currentlyProcessing;

    if (availableSlots > 0) {
      const pendingItems = queue.filter(
        item => item.status === 'pending' && !activeProcessingRef.current.has(item.id)
      );
      const itemsToStart = pendingItems.slice(0, availableSlots);

      itemsToStart.forEach(item => {
        activeProcessingRef.current.add(item.id);
        processQueueItem(item);
      });
    }
  }, [queue, processQueueItem]);

  // ── Drag handlers ──────────────────────────────────────────────────────────
  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') setDragActive(true);
    else if (e.type === 'dragleave') setDragActive(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files?.length) handleFiles(e.dataTransfer.files);
  };

  // ── Validation ─────────────────────────────────────────────────────────────
  const validateFile = (file) => {
    const ext = file.name.split('.').pop().toLowerCase();
    if (!ALLOWED_EXT.includes(ext)) {
      return `"${file.name}" is not supported. Allowed: PDF, JPG, PNG.`;
    }
    if (file.size > MAX_SIZE_MB * 1024 * 1024) {
      return `"${file.name}" exceeds the ${MAX_SIZE_MB}MB limit.`;
    }
    return null;
  };

  // ── File list handler ──────────────────────────────────────────────────────
  const handleFiles = (fileList) => {
    const newItems = Array.from(fileList).map(file => {
      const id = `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
      const validationError = validateFile(file);
      return {
        id,
        file,
        name: file.name,
        size: file.size,
        status: validationError ? 'error' : 'pending',
        stageIndex: 0,
        progress: 0,
        result: null,
        error: validationError,
      };
    });

    setQueue(prev => [...prev, ...newItems]);
  };

  const handleFileInputChange = (e) => {
    if (e.target.files?.length) {
      handleFiles(e.target.files);
      e.target.value = ''; // Reset file input so that selecting same file fires onChange again
    }
  };

  const clearQueue = () => {
    // Only clear non-processing items to avoid breaking the active fetch
    setQueue(prev => prev.filter(item => item.status === 'processing'));
  };

  // ── Render ─────────────────────────────────────────────────────────────────
  return (
    <div className="px-8 py-12" data-testid="upload-documents">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.35 }}
      >
        {/* Header */}
        <div className="mb-10">
          <h1 className="text-4xl md:text-5xl font-bold text-slate-900 dark:text-white tracking-tight mb-2">
            Upload Documents
          </h1>
          <p className="text-lg text-slate-500 dark:text-slate-400">
            Drop files — Qwen AI will read, categorize, and index them automatically.
          </p>
        </div>

        <div className="max-w-3xl mx-auto space-y-6">

          {/* Drop Zone */}
          <Card className={`border-2 border-dashed transition-all duration-200 ${
            dragActive
              ? 'border-cyan-500 bg-cyan-50 dark:bg-cyan-900/10 shadow-lg'
              : isProcessing
              ? 'border-violet-400 bg-violet-50 dark:bg-violet-900/10'
              : 'border-slate-300 dark:border-slate-700 hover:border-slate-400 dark:hover:border-slate-600'
          }`}>
            <CardContent className="p-12">
              <div
                onDragEnter={handleDrag}
                onDragLeave={handleDrag}
                onDragOver={handleDrag}
                onDrop={handleDrop}
                className="flex flex-col items-center justify-center text-center space-y-6"
                data-testid="upload-dropzone"
              >
                {/* Icon */}
                <div className={`p-6 rounded-2xl transition-colors ${
                  isProcessing ? 'bg-violet-100 dark:bg-violet-900/30' : 'bg-slate-100 dark:bg-slate-800'
                }`}>
                  {isProcessing
                    ? <Loader2 className="w-12 h-12 text-violet-500 animate-spin" strokeWidth={1.5} />
                    : <Upload  className="w-12 h-12 text-cyan-500" strokeWidth={1.5} />
                  }
                </div>

                <div>
                  <h3 className="text-xl font-semibold text-slate-900 dark:text-white mb-1">
                    {isProcessing ? 'Processing documents...' : 'Drag & drop files here'}
                  </h3>
                  <p className="text-slate-500 dark:text-slate-400 text-sm">
                    {isProcessing
                      ? `${queue.filter(q => q.status === 'pending' || q.status === 'processing').length} files remaining in queue`
                      : `Supported: PDF, JPG, PNG — max ${MAX_SIZE_MB}MB`
                    }
                  </p>
                </div>

                <div className="flex items-center gap-4">
                  <Button
                    onClick={() => fileInputRef.current?.click()}
                    className="bg-gradient-to-r from-cyan-500 to-violet-600 hover:from-cyan-400 hover:to-violet-500 text-white shadow-md"
                    data-testid="browse-files-button"
                  >
                    <FileText className="w-4 h-4 mr-2" />
                    Browse Files
                  </Button>
                </div>

                <input
                  ref={fileInputRef}
                  id="file-input"
                  type="file"
                  className="hidden"
                  accept=".pdf,.jpg,.jpeg,.png"
                  multiple
                  onChange={handleFileInputChange}
                  data-testid="file-input"
                />
              </div>
            </CardContent>
          </Card>

          {/* How it works */}
          {queue.length === 0 && (
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
              {[
                { icon: Upload,       label: 'Upload',    desc: 'PDF or image' },
                { icon: FileText,     label: 'OCR + Read', desc: 'Pages extracted' },
                { icon: Info,         label: 'Qwen AI',   desc: 'Classifies type' },
                { icon: Tag,          label: 'Auto-Indexed', desc: 'Ready to search' },
              ].map(({ icon: Icon, label, desc }) => (
                <div key={label} className="flex flex-col items-center gap-2 p-4 rounded-xl bg-slate-50 dark:bg-slate-900 border border-slate-200 dark:border-slate-800 text-center">
                  <div className="p-2 rounded-lg bg-white dark:bg-slate-800 shadow-sm">
                    <Icon className="w-5 h-5 text-cyan-500" />
                  </div>
                  <p className="text-xs font-semibold text-slate-800 dark:text-slate-200">{label}</p>
                  <p className="text-xs text-slate-500 dark:text-slate-400">{desc}</p>
                </div>
              ))}
            </div>
          )}

          {/* Queue List */}
          {queue.length > 0 && (
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <h3 className="text-lg font-semibold text-slate-900 dark:text-white">
                  Documents Queue ({queue.length})
                </h3>
                <button
                  onClick={clearQueue}
                  className="text-xs text-slate-400 hover:text-red-500 transition-colors"
                >
                  Clear completed / failed
                </button>
              </div>

              <div className="space-y-3">
                <AnimatePresence initial={false}>
                  {queue.map((item) => {
                    if (item.status === 'done') {
                      return (
                        <ResultCard
                          key={item.id}
                          result={item.result}
                          onRemove={() => {
                            setQueue(prev => prev.filter(q => q.id !== item.id));
                          }}
                        />
                      );
                    }

                    return (
                      <motion.div
                        key={item.id}
                        initial={{ opacity: 0, y: 10 }}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{ opacity: 0, scale: 0.95 }}
                        transition={{ duration: 0.2 }}
                      >
                        <Card className="border-slate-200 dark:border-slate-800 overflow-hidden">
                          <CardContent className="p-4 flex flex-col md:flex-row md:items-center justify-between gap-4">
                            <div className="flex items-center gap-3 min-w-0 flex-1">
                              <div className={`p-2.5 rounded-xl flex-shrink-0 ${
                                item.status === 'processing'
                                  ? 'bg-violet-100 dark:bg-violet-900/20 text-violet-500'
                                  : item.status === 'error' || item.status === 'cancelled'
                                  ? 'bg-red-100 dark:bg-red-900/20 text-red-500'
                                  : 'bg-slate-100 dark:bg-slate-800 text-slate-400'
                              }`}>
                                {item.status === 'processing' ? (
                                  <Loader2 className="w-5 h-5 animate-spin" />
                                ) : item.status === 'error' ? (
                                  <AlertCircle className="w-5 h-5" />
                                ) : item.status === 'cancelled' ? (
                                  <Ban className="w-5 h-5" />
                                ) : (
                                  <File className="w-5 h-5" />
                                )}
                              </div>
                              <div className="min-w-0 flex-1">
                                <p className="font-medium text-sm text-slate-900 dark:text-white truncate">
                                  {item.name}
                                </p>
                                <p className="text-xs text-slate-400 dark:text-slate-500">
                                  {(item.size / 1024 / 1024).toFixed(2)} MB
                                </p>
                              </div>
                            </div>

                            {/* Details based on status */}
                            <div className="flex flex-col md:items-end gap-2 flex-shrink-0 min-w-[240px]">
                              {item.status === 'pending' && (
                                <span className="text-xs font-medium px-2 py-1 rounded bg-slate-100 dark:bg-slate-800 text-slate-500">
                                  Waiting in queue
                                </span>
                              )}

                              {item.status === 'processing' && (
                                <div className="w-full space-y-2">
                                  {/* Stage label and progress */}
                                  <div className="flex items-center justify-between text-xs text-slate-500 dark:text-slate-400">
                                    <span className="truncate">{STAGES[item.stageIndex]?.label}</span>
                                    <span>{item.progress}%</span>
                                  </div>
                                  <Progress value={item.progress} className="h-1.5 w-full" />
                                  {/* Stage dots */}
                                  <div className="flex items-center justify-start md:justify-end gap-1.5 pt-0.5">
                                    {STAGES.map((s, idx) => (
                                      <div
                                        key={s.id}
                                        className={`h-1 rounded-full transition-all duration-300 ${
                                          idx < item.stageIndex
                                            ? 'w-3 bg-violet-500'
                                            : idx === item.stageIndex
                                            ? 'w-4 bg-violet-400'
                                            : 'w-1 bg-slate-200 dark:bg-slate-700'
                                        }`}
                                      />
                                    ))}
                                  </div>
                                </div>
                              )}

                              {item.status === 'error' && (
                                <div className="flex items-center gap-2 text-xs text-red-500 max-w-[300px]">
                                  <span className="truncate" title={item.error}>{item.error}</span>
                                </div>
                              )}

                              {item.status === 'cancelled' && (
                                <span className="text-xs font-medium px-2 py-1 rounded bg-red-50 dark:bg-red-900/20 text-red-400">
                                  Cancelled
                                </span>
                              )}
                            </div>

                            {/* Action buttons */}
                            <div className="flex items-center justify-end flex-shrink-0 gap-1">
                              {/* Cancel button — only while processing */}
                              {item.status === 'processing' && (
                                <button
                                  onClick={() => handleCancel(item.id)}
                                  className="flex items-center gap-1 px-2 py-1 rounded-lg text-xs font-medium bg-red-50 dark:bg-red-900/20 text-red-500 hover:bg-red-100 dark:hover:bg-red-900/40 transition-colors"
                                  title="Cancel upload"
                                >
                                  <Ban className="w-3.5 h-3.5" />
                                  Cancel
                                </button>
                              )}

                              {/* Dismiss button — for pending / error / cancelled */}
                              {item.status !== 'processing' && (
                                <button
                                  onClick={() => setQueue(prev => prev.filter(q => q.id !== item.id))}
                                  className="p-1 rounded-lg hover:bg-slate-100 dark:hover:bg-slate-800 text-slate-400 hover:text-slate-600 dark:hover:text-slate-300 transition-colors"
                                  title="Remove from queue"
                                >
                                  <X className="w-4 h-4" />
                                </button>
                              )}
                            </div>
                          </CardContent>
                        </Card>
                      </motion.div>
                    );
                  })}
                </AnimatePresence>
              </div>
            </div>
          )}

        </div>
      </motion.div>
    </div>
  );
};

export default UploadDocuments;
