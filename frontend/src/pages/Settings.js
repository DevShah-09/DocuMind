import { motion } from 'framer-motion';
import { Moon, Sun, Zap, Brain, Gauge, Shield } from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Label } from '@/components/ui/label';
import { Switch } from '@/components/ui/switch';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Separator } from '@/components/ui/separator';
import { Button } from '@/components/ui/button';

const Settings = ({ darkMode, toggleDarkMode }) => {
  return (
    <div className="px-8 py-12" data-testid="settings">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.3 }}
      >
        {/* Header */}
        <div className="mb-12">
          <h1 className="text-4xl md:text-5xl font-bold text-slate-900 dark:text-white tracking-tight mb-3">
            Settings
          </h1>
          <p className="text-lg text-slate-600 dark:text-slate-300">
            Customize your DocuMind AI experience
          </p>
        </div>

        {/* Settings Cards */}
        <div className="max-w-4xl mx-auto space-y-6">
          {/* Appearance */}
          <Card className="border-slate-200 dark:border-slate-800">
            <CardHeader>
              <CardTitle className="text-2xl font-semibold flex items-center gap-2">
                {darkMode ? <Moon className="w-5 h-5 text-slate-700 dark:text-slate-300" /> : <Sun className="w-5 h-5 text-slate-700" />}
                Appearance
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="flex items-center justify-between">
                <div className="space-y-1">
                  <Label htmlFor="dark-mode" className="text-base font-medium">Dark Mode</Label>
                  <p className="text-sm text-slate-600 dark:text-slate-400">Toggle dark mode theme</p>
                </div>
                <Switch
                  id="dark-mode"
                  checked={darkMode}
                  onCheckedChange={toggleDarkMode}
                  data-testid="dark-mode-toggle"
                />
              </div>
            </CardContent>
          </Card>

          {/* Model Configuration */}
          <Card className="border-slate-200 dark:border-slate-800">
            <CardHeader>
              <CardTitle className="text-2xl font-semibold flex items-center gap-2">
                <Brain className="w-5 h-5 text-cyan-500" />
                Model Configuration
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="space-y-3">
                <Label htmlFor="model-select" className="text-base font-medium">Language Model</Label>
                <Select defaultValue="gpt4">
                  <SelectTrigger id="model-select" data-testid="model-select">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="gpt4">GPT-4 Turbo</SelectItem>
                    <SelectItem value="gpt35">GPT-3.5 Turbo</SelectItem>
                    <SelectItem value="llama">LLaMA 3</SelectItem>
                    <SelectItem value="miniLM">MiniLM</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <Separator />

              <div className="space-y-3">
                <Label htmlFor="embedding-model" className="text-base font-medium">Embedding Model</Label>
                <Select defaultValue="openai">
                  <SelectTrigger id="embedding-model" data-testid="embedding-model-select">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="openai">OpenAI Embeddings</SelectItem>
                    <SelectItem value="miniLM">Sentence Transformers (MiniLM)</SelectItem>
                    <SelectItem value="cohere">Cohere Embeddings</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            </CardContent>
          </Card>

          {/* Advanced Options */}
          <Card className="border-slate-200 dark:border-slate-800">
            <CardHeader>
              <CardTitle className="text-2xl font-semibold flex items-center gap-2">
                <Zap className="w-5 h-5 text-orange-500" />
                Advanced Options
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="flex items-center justify-between">
                <div className="space-y-1">
                  <Label htmlFor="use-reranker" className="text-base font-medium">Use Re-Ranker</Label>
                  <p className="text-sm text-slate-600 dark:text-slate-400">Improve result relevance with re-ranking</p>
                </div>
                <Switch id="use-reranker" defaultChecked data-testid="use-reranker-toggle" />
              </div>

              <Separator />

              <div className="flex items-center justify-between">
                <div className="space-y-1">
                  <Label htmlFor="show-reasoning" className="text-base font-medium">Show Reasoning Steps</Label>
                  <p className="text-sm text-slate-600 dark:text-slate-400">Display AI reasoning process</p>
                </div>
                <Switch id="show-reasoning" data-testid="show-reasoning-toggle" />
              </div>

              <Separator />

              <div className="flex items-center justify-between">
                <div className="space-y-1">
                  <Label htmlFor="strict-context" className="text-base font-medium">Strict Context Mode</Label>
                  <p className="text-sm text-slate-600 dark:text-slate-400">Only use information from documents</p>
                </div>
                <Switch id="strict-context" defaultChecked data-testid="strict-context-toggle" />
              </div>
            </CardContent>
          </Card>

          {/* Performance */}
          <Card className="border-slate-200 dark:border-slate-800">
            <CardHeader>
              <CardTitle className="text-2xl font-semibold flex items-center gap-2">
                <Gauge className="w-5 h-5 text-blue-500" />
                Performance
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="space-y-3">
                <Label htmlFor="chunk-size" className="text-base font-medium">Chunk Size</Label>
                <Select defaultValue="medium">
                  <SelectTrigger id="chunk-size" data-testid="chunk-size-select">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="small">Small (256 tokens)</SelectItem>
                    <SelectItem value="medium">Medium (512 tokens)</SelectItem>
                    <SelectItem value="large">Large (1024 tokens)</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <Separator />

              <div className="space-y-3">
                <Label htmlFor="top-k" className="text-base font-medium">Top-K Results</Label>
                <Select defaultValue="5">
                  <SelectTrigger id="top-k" data-testid="top-k-select">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="3">3 results</SelectItem>
                    <SelectItem value="5">5 results</SelectItem>
                    <SelectItem value="10">10 results</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            </CardContent>
          </Card>

          {/* Save Button */}
          <div className="flex justify-end gap-4">
            <Button variant="outline" data-testid="reset-settings-button">Reset to Defaults</Button>
            <Button
              className="bg-gradient-to-r from-slate-900 to-slate-800 hover:from-slate-800 hover:to-slate-700 text-white"
              data-testid="save-settings-button"
            >
              Save Changes
            </Button>
          </div>
        </div>
      </motion.div>
    </div>
  );
};

export default Settings;
  