import { motion } from 'framer-motion';
import { Clock, MessageSquare, FileText, ChevronRight } from 'lucide-react';
import { Card, CardContent } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { ScrollArea } from '@/components/ui/scroll-area';

const QueryHistory = () => {
  const history = [
    {
      id: 1,
      query: 'What are the key findings in the AI research papers from 2024?',
      timestamp: '2 hours ago',
      sources: 5,
      confidence: 94,
    },
    {
      id: 2,
      query: 'Explain the implementation of vector databases',
      timestamp: '5 hours ago',
      sources: 3,
      confidence: 89,
    },
    {
      id: 3,
      query: 'Summarize the technical specifications document',
      timestamp: 'Yesterday',
      sources: 4,
      confidence: 92,
    },
    {
      id: 4,
      query: 'What are the legal requirements mentioned in the compliance docs?',
      timestamp: '2 days ago',
      sources: 6,
      confidence: 87,
    },
    {
      id: 5,
      query: 'Compare the different product specifications',
      timestamp: '3 days ago',
      sources: 8,
      confidence: 91,
    },
    {
      id: 6,
      query: 'What are the main topics covered in the training materials?',
      timestamp: '4 days ago',
      sources: 4,
      confidence: 88,
    },
  ];

  return (
    <div className="px-8 py-12" data-testid="query-history">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.3 }}
      >
        {/* Header */}
        <div className="mb-12">
          <h1 className="text-4xl md:text-5xl font-bold text-slate-900 dark:text-white tracking-tight mb-3">
            Query History
          </h1>
          <p className="text-lg text-slate-600 dark:text-slate-300">
            Review your previous questions and answers
          </p>
        </div>

        {/* History List */}
        <div className="max-w-4xl mx-auto">
          <ScrollArea className="h-[calc(100vh-250px)]">
            <div className="space-y-4">
              {history.map((item, index) => (
                <motion.div
                  key={item.id}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ duration: 0.3, delay: index * 0.05 }}
                >
                  <Card className="hover:shadow-lg transition-all duration-200 border-slate-200 dark:border-slate-800 cursor-pointer group" data-testid={`history-item-${item.id}`}>
                    <CardContent className="p-6">
                      <div className="flex items-start justify-between gap-4">
                        <div className="flex-1 space-y-3">
                          <div className="flex items-start gap-3">
                            <div className="p-2 rounded-lg bg-cyan-100 dark:bg-cyan-900/20 mt-1">
                              <MessageSquare className="w-5 h-5 text-cyan-600 dark:text-cyan-400" strokeWidth={1.5} />
                            </div>
                            <div className="flex-1">
                              <p className="text-lg font-medium text-slate-900 dark:text-white mb-2">
                                {item.query}
                              </p>
                              <div className="flex items-center gap-4 flex-wrap">
                                <div className="flex items-center gap-1 text-sm text-slate-600 dark:text-slate-400">
                                  <Clock className="w-4 h-4" strokeWidth={1.5} />
                                  {item.timestamp}
                                </div>
                                <Badge variant="outline" className="bg-green-50 dark:bg-green-900/20 text-green-700 dark:text-green-400 border-green-200 dark:border-green-800">
                                  {item.confidence}% confidence
                                </Badge>
                                <Badge variant="outline" className="bg-blue-50 dark:bg-blue-900/20 text-blue-700 dark:text-blue-400 border-blue-200 dark:border-blue-800">
                                  <FileText className="w-3 h-3 mr-1" />
                                  {item.sources} sources
                                </Badge>
                              </div>
                            </div>
                          </div>
                        </div>
                        <Button
                          variant="ghost"
                          size="icon"
                          className="opacity-0 group-hover:opacity-100 transition-opacity"
                          data-testid={`view-query-${item.id}`}
                        >
                          <ChevronRight className="w-5 h-5" />
                        </Button>
                      </div>
                    </CardContent>
                  </Card>
                </motion.div>
              ))}
            </div>
          </ScrollArea>
        </div>
      </motion.div>
    </div>
  );
};

export default QueryHistory;
