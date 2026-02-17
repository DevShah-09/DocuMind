import { useState } from 'react';
import { motion } from 'framer-motion';
import { Upload, FileText, File, Link as LinkIcon, CheckCircle2, Loader2 } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Switch } from '@/components/ui/switch';
import { Card, CardContent } from '@/components/ui/card';
import { Progress } from '@/components/ui/progress';
import { toast } from 'sonner';

const UploadDocuments = () => {
  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [dragActive, setDragActive] = useState(false);
  const [showChunkPreview, setShowChunkPreview] = useState(false);
  const [uploadedFiles, setUploadedFiles] = useState([]);

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFiles(e.dataTransfer.files);
    }
  };

  const handleFiles = (files) => {
    setUploading(true);
    setUploadProgress(0);

    // Simulate upload progress
    const interval = setInterval(() => {
      setUploadProgress((prev) => {
        if (prev >= 100) {
          clearInterval(interval);
          setUploading(false);
          setUploadedFiles([
            ...uploadedFiles,
            { name: files[0].name, status: 'completed', chunks: 45, embeddings: 1024 }
          ]);
          toast.success('Document uploaded successfully!');
          return 100;
        }
        return prev + 10;
      });
    }, 300);
  };

  return (
    <div className="px-8 py-12" data-testid="upload-documents">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.3 }}
      >
        {/* Header */}
        <div className="mb-12">
          <h1 className="text-4xl md:text-5xl font-bold text-slate-900 dark:text-white tracking-tight mb-3">
            Upload Documents
          </h1>
          <p className="text-lg text-slate-600 dark:text-slate-300">
            Add documents to your knowledge base
          </p>
        </div>

        {/* Upload Area */}
        <div className="max-w-3xl mx-auto space-y-8">
          <Card className="border-slate-200 dark:border-slate-800">
            <CardContent className="p-0">
              <div
                onDragEnter={handleDrag}
                onDragLeave={handleDrag}
                onDragOver={handleDrag}
                onDrop={handleDrop}
                className={`p-12 border-2 border-dashed rounded-xl transition-all duration-200 ${
                  dragActive
                    ? 'border-cyan-500 bg-cyan-50 dark:bg-cyan-900/10'
                    : 'border-slate-300 dark:border-slate-700'
                }`}
                data-testid="upload-dropzone"
              >
                <div className="flex flex-col items-center justify-center text-center space-y-6">
                  <div className="p-6 rounded-2xl bg-slate-100 dark:bg-slate-800">
                    <Upload className="w-12 h-12 text-cyan-500" strokeWidth={1.5} />
                  </div>
                  <div>
                    <h3 className="text-xl font-semibold text-slate-900 dark:text-white mb-2">
                      Drag & drop your files here
                    </h3>
                    <p className="text-slate-600 dark:text-slate-400">
                      Supports PDF, TXT, DOCX, and URLs
                    </p>
                  </div>
                  <div className="flex items-center gap-4">
                    <Button
                      onClick={() => document.getElementById('file-input').click()}
                      className="bg-gradient-to-r from-slate-900 to-slate-800 hover:from-slate-800 hover:to-slate-700 text-white"
                      data-testid="browse-files-button"
                    >
                      <FileText className="w-4 h-4 mr-2" />
                      Browse Files
                    </Button>
                    <span className="text-sm text-slate-500">or</span>
                    <Button variant="outline" data-testid="add-url-button">
                      <LinkIcon className="w-4 h-4 mr-2" />
                      Add URL
                    </Button>
                  </div>
                  <input
                    id="file-input"
                    type="file"
                    className="hidden"
                    onChange={(e) => e.target.files && handleFiles(e.target.files)}
                    accept=".pdf,.txt,.docx"
                  />
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Upload Progress */}
          {uploading && (
            <motion.div
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
            >
              <Card className="border-slate-200 dark:border-slate-800" data-testid="upload-progress">
                <CardContent className="p-6">
                  <div className="space-y-4">
                    <div className="flex items-center justify-between">
                      <span className="text-sm font-medium text-slate-900 dark:text-white">Uploading...</span>
                      <span className="text-sm text-slate-600 dark:text-slate-400">{uploadProgress}%</span>
                    </div>
                    <Progress value={uploadProgress} className="h-2" />
                    <div className="flex items-center gap-2 text-sm text-slate-600 dark:text-slate-400">
                      <Loader2 className="w-4 h-4 animate-spin" />
                      Processing document and generating embeddings...
                    </div>
                  </div>
                </CardContent>
              </Card>
            </motion.div>
          )}

          {/* Options */}
          <Card className="border-slate-200 dark:border-slate-800">
            <CardContent className="p-6 space-y-4">
              <div className="flex items-center justify-between">
                <div>
                  <Label htmlFor="chunk-preview" className="text-base font-medium">Chunk Preview</Label>
                  <p className="text-sm text-slate-600 dark:text-slate-400 mt-1">View document chunks after processing</p>
                </div>
                <Switch
                  id="chunk-preview"
                  checked={showChunkPreview}
                  onCheckedChange={setShowChunkPreview}
                  data-testid="chunk-preview-toggle"
                />
              </div>
              <div className="flex items-center justify-between">
                <div>
                  <Label htmlFor="auto-embed" className="text-base font-medium">Auto-Generate Embeddings</Label>
                  <p className="text-sm text-slate-600 dark:text-slate-400 mt-1">Automatically create vector embeddings</p>
                </div>
                <Switch id="auto-embed" defaultChecked data-testid="auto-embed-toggle" />
              </div>
            </CardContent>
          </Card>

          {/* Uploaded Files */}
          {uploadedFiles.length > 0 && (
            <div className="space-y-4">
              <h3 className="text-xl font-semibold text-slate-900 dark:text-white">Recently Uploaded</h3>
              {uploadedFiles.map((file, index) => (
                <motion.div
                  key={index}
                  initial={{ opacity: 0, scale: 0.95 }}
                  animate={{ opacity: 1, scale: 1 }}
                >
                  <Card className="border-slate-200 dark:border-slate-800" data-testid={`uploaded-file-${index}`}>
                    <CardContent className="p-6">
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-4">
                          <div className="p-3 rounded-lg bg-green-100 dark:bg-green-900/20">
                            <File className="w-6 h-6 text-green-600 dark:text-green-400" />
                          </div>
                          <div>
                            <p className="font-medium text-slate-900 dark:text-white">{file.name}</p>
                            <p className="text-sm text-slate-600 dark:text-slate-400">
                              {file.chunks} chunks • {file.embeddings} embeddings
                            </p>
                          </div>
                        </div>
                        <CheckCircle2 className="w-6 h-6 text-green-600 dark:text-green-400" />
                      </div>
                    </CardContent>
                  </Card>
                </motion.div>
              ))}
            </div>
          )}
        </div>
      </motion.div>
    </div>
  );
};

export default UploadDocuments;
