import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Search, Mic, Paperclip, Send, ChevronRight, ChevronLeft, Sparkles, FileText } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Switch } from '@/components/ui/switch';
import { Label } from '@/components/ui/label';
import { Badge } from '@/components/ui/badge';
import { Card } from '@/components/ui/card';
import { ScrollArea } from '@/components/ui/scroll-area';

const QueryInterface = () => {
  const [query, setQuery] = useState('');
  const [messages, setMessages] = useState([]);
  const [showRightPanel, setShowRightPanel] = useState(true);
  const [selectedCollection, setSelectedCollection] = useState('all');
  const [showAdvanced, setShowAdvanced] = useState(false);

  const mockRetrievedChunks = [
    {
      id: 1,
      similarity: 94.2,
      rank: 1,
      document: 'Research_Paper_AI_2024.pdf',
      content: 'Large language models have demonstrated remarkable capabilities in understanding and generating human-like text...'
    },
    {
      id: 2,
      similarity: 89.7,
      rank: 2,
      document: 'Technical_Specs_v2.docx',
      content: 'The vector database implementation uses cosine similarity for efficient document retrieval...'
    },
    {
      id: 3,
      similarity: 87.3,
      rank: 3,
      document: 'Documentation_Guide.pdf',
      content: 'When implementing RAG systems, it is crucial to maintain proper chunk size and overlap...'
    },
  ];

  const handleSendQuery = () => {
    if (!query.trim()) return;

    const newMessage = {
      id: Date.now(),
      type: 'user',
      content: query,
      timestamp: new Date().toLocaleTimeString(),
    };

    setMessages([...messages, newMessage]);

    // Simulate AI response
    setTimeout(() => {
      const aiResponse = {
        id: Date.now() + 1,
        type: 'ai',
        content: 'Based on the retrieved documents, I can provide you with comprehensive information about your query. The analysis shows multiple relevant sources that address your question with high confidence.',
        confidence: 92,
        sources: 3,
        timestamp: new Date().toLocaleTimeString(),
      };
      setMessages(prev => [...prev, aiResponse]);
    }, 1000);

    setQuery('');
  };

  return (
    <div className="h-screen flex" data-testid="query-interface">
      {/* Main Chat Area */}
      <div className="flex-1 flex flex-col">
        {/* Top Search Bar */}
        <div className="px-8 py-6 border-b border-slate-200 dark:border-slate-800 bg-white/50 dark:bg-slate-900/50 backdrop-blur-sm">
          <div className="max-w-4xl mx-auto">
            <div className="relative">
              <Search className="absolute left-6 top-1/2 -translate-y-1/2 w-5 h-5 text-slate-400" strokeWidth={1.5} />
              <Input
                placeholder="Ask anything about your documents..."
                className="h-14 pl-14 pr-32 text-lg rounded-xl border-slate-200 dark:border-slate-800 shadow-lg focus:ring-2 focus:ring-cyan-500 dark:bg-slate-900"
                data-testid="search-input"
              />
              <div className="absolute right-3 top-1/2 -translate-y-1/2 flex items-center gap-2">
                <Button variant="ghost" size="icon" className="rounded-full" data-testid="mic-button">
                  <Mic className="w-5 h-5 text-slate-600 dark:text-slate-400" strokeWidth={1.5} />
                </Button>
                <Select value={selectedCollection} onValueChange={setSelectedCollection}>
                  <SelectTrigger className="w-32 h-9 rounded-lg" data-testid="collection-select">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="all">All</SelectItem>
                    <SelectItem value="research">Research</SelectItem>
                    <SelectItem value="docs">Docs</SelectItem>
                    <SelectItem value="legal">Legal</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            </div>

            {/* Advanced Toggle */}
            <div className="flex items-center gap-2 mt-4">
              <Switch
                id="advanced"
                checked={showAdvanced}
                onCheckedChange={setShowAdvanced}
                data-testid="advanced-toggle"
              />
              <Label htmlFor="advanced" className="text-sm text-slate-600 dark:text-slate-400 cursor-pointer">
                Advanced Options
              </Label>
            </div>

            {/* Advanced Options */}
            <AnimatePresence>
              {showAdvanced && (
                <motion.div
                  initial={{ opacity: 0, height: 0 }}
                  animate={{ opacity: 1, height: 'auto' }}
                  exit={{ opacity: 0, height: 0 }}
                  className="mt-4 flex gap-4"
                >
                  <div className="flex items-center gap-2">
                    <Switch id="reranker" data-testid="reranker-toggle" />
                    <Label htmlFor="reranker" className="text-sm">Use Re-Ranker</Label>
                  </div>
                  <div className="flex items-center gap-2">
                    <Switch id="reasoning" data-testid="reasoning-toggle" />
                    <Label htmlFor="reasoning" className="text-sm">Show Reasoning</Label>
                  </div>
                  <div className="flex items-center gap-2">
                    <Switch id="strict" data-testid="strict-toggle" />
                    <Label htmlFor="strict" className="text-sm">Strict Context</Label>
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        </div>

        {/* Chat Messages */}
        <ScrollArea className="flex-1 px-8 py-8">
          <div className="max-w-4xl mx-auto space-y-6">
            {messages.length === 0 ? (
              <div className="flex flex-col items-center justify-center h-full py-20" data-testid="empty-state">
                <div className="p-6 rounded-2xl bg-slate-100 dark:bg-slate-800 mb-6">
                  <Sparkles className="w-12 h-12 text-cyan-500" strokeWidth={1.5} />
                </div>
                <h2 className="text-2xl font-semibold text-slate-900 dark:text-white mb-2">Ready to assist you</h2>
                <p className="text-slate-600 dark:text-slate-400 text-center max-w-md">
                  Ask any question about your documents and I'll provide accurate answers with source references
                </p>
              </div>
            ) : (
              <AnimatePresence>
                {messages.map((message) => (
                  <motion.div
                    key={message.id}
                    initial={{ opacity: 0, scale: 0.95, y: 10 }}
                    animate={{ opacity: 1, scale: 1, y: 0 }}
                    className={`flex ${message.type === 'user' ? 'justify-end' : 'justify-start'}`}
                  >
                    {message.type === 'user' ? (
                      <div className="bg-slate-100 dark:bg-slate-800 text-slate-900 dark:text-slate-100 rounded-2xl rounded-tr-sm px-5 py-3.5 max-w-[80%]" data-testid="user-message">
                        {message.content}
                      </div>
                    ) : (
                      <div className="w-full space-y-4" data-testid="ai-message">
                        <div className="bg-transparent text-slate-800 dark:text-slate-200 rounded-2xl py-3.5 leading-relaxed">
                          {message.content}
                        </div>
                        <div className="flex items-center gap-4">
                          <Badge variant="outline" className="bg-green-50 dark:bg-green-900/20 text-green-700 dark:text-green-400 border-green-200 dark:border-green-800">
                            Confidence: {message.confidence}%
                          </Badge>
                          <Badge variant="outline" className="bg-blue-50 dark:bg-blue-900/20 text-blue-700 dark:text-blue-400 border-blue-200 dark:border-blue-800">
                            {message.sources} sources
                          </Badge>
                        </div>
                      </div>
                    )}
                  </motion.div>
                ))}
              </AnimatePresence>
            )}
          </div>
        </ScrollArea>

        {/* Bottom Input */}
        <div className="px-8 py-6 border-t border-slate-200 dark:border-slate-800 bg-white/50 dark:bg-slate-900/50 backdrop-blur-sm">
          <div className="max-w-4xl mx-auto">
            <div className="relative">
              <Input
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                onKeyPress={(e) => e.key === 'Enter' && handleSendQuery()}
                placeholder="Type your query here..."
                className="h-14 pl-6 pr-28 rounded-xl border-slate-200 dark:border-slate-800"
                data-testid="query-input"
              />
              <div className="absolute right-3 top-1/2 -translate-y-1/2 flex items-center gap-2">
                <Button variant="ghost" size="icon" className="rounded-full" data-testid="attach-button">
                  <Paperclip className="w-5 h-5 text-slate-600 dark:text-slate-400" strokeWidth={1.5} />
                </Button>
                <Button
                  onClick={handleSendQuery}
                  className="bg-gradient-to-r from-cyan-600 to-cyan-500 hover:from-cyan-500 hover:to-cyan-600 text-white rounded-full w-10 h-10 p-0"
                  data-testid="send-button"
                >
                  <Send className="w-5 h-5" strokeWidth={2} />
                </Button>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Right Panel - Retrieved Documents */}
      <AnimatePresence>
        {showRightPanel && (
          <motion.div
            initial={{ width: 0, opacity: 0 }}
            animate={{ width: 320, opacity: 1 }}
            exit={{ width: 0, opacity: 0 }}
            className="border-l border-slate-200 dark:border-slate-800 bg-white/70 dark:bg-slate-900/70 backdrop-blur-xl overflow-hidden"
            data-testid="right-panel"
          >
            <div className="h-full flex flex-col">
              <div className="px-6 py-6 border-b border-slate-200 dark:border-slate-800">
                <h3 className="text-lg font-semibold text-slate-900 dark:text-white flex items-center gap-2">
                  <FileText className="w-5 h-5 text-cyan-500" strokeWidth={1.5} />
                  Retrieved Sources
                </h3>
              </div>

              <ScrollArea className="flex-1 px-6 py-4">
                <div className="space-y-4">
                  {mockRetrievedChunks.map((chunk) => (
                    <Card key={chunk.id} className="p-4 hover:shadow-md transition-shadow border-slate-200 dark:border-slate-800" data-testid={`chunk-${chunk.id}`}>
                      <div className="space-y-3">
                        <div className="flex items-start justify-between">
                          <Badge className="bg-cyan-100 dark:bg-cyan-900/30 text-cyan-700 dark:text-cyan-400">
                            #{chunk.rank}
                          </Badge>
                          <span className="text-xs font-semibold text-green-600 dark:text-green-400">
                            {chunk.similarity}%
                          </span>
                        </div>
                        <div>
                          <p className="text-xs font-medium text-slate-500 dark:text-slate-400 mb-1">Source</p>
                          <p className="text-sm font-semibold text-slate-900 dark:text-white truncate">
                            {chunk.document}
                          </p>
                        </div>
                        <div>
                          <p className="text-xs font-medium text-slate-500 dark:text-slate-400 mb-1">Preview</p>
                          <p className="text-xs text-slate-600 dark:text-slate-300 line-clamp-3 leading-relaxed">
                            {chunk.content}
                          </p>
                        </div>
                        <Button variant="ghost" size="sm" className="w-full text-xs" data-testid={`expand-chunk-${chunk.id}`}>
                          View Full Chunk
                          <ChevronRight className="w-3 h-3 ml-1" />
                        </Button>
                      </div>
                    </Card>
                  ))}
                </div>
              </ScrollArea>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Toggle Right Panel Button */}
      <Button
        onClick={() => setShowRightPanel(!showRightPanel)}
        variant="ghost"
        size="icon"
        className="absolute right-4 top-4 rounded-full bg-white dark:bg-slate-800 shadow-lg hover:shadow-xl"
        data-testid="toggle-right-panel"
      >
        {showRightPanel ? <ChevronRight className="w-5 h-5" /> : <ChevronLeft className="w-5 h-5" />}
      </Button>
    </div>
  );
};

export default QueryInterface;
