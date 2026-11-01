import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import {
  FolderOpen,
  FileText,
  MoreVertical,
  Trash2,
  Edit,
  Loader2,
  ExternalLink,
  Upload,
  Plus,
} from 'lucide-react';
import { Card, CardContent } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Input } from '@/components/ui/input';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu';
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogFooter,
} from '@/components/ui/dialog';
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from '@/components/ui/alert-dialog';
import { toast } from 'sonner';
import { useAuth } from '@/contexts/AuthContext';

const colorMap = {
  blue:   'bg-blue-100 dark:bg-blue-900/20 text-blue-600 dark:text-blue-400',
  green:  'bg-green-100 dark:bg-green-900/20 text-green-600 dark:text-green-400',
  purple: 'bg-purple-100 dark:bg-purple-900/20 text-purple-600 dark:text-purple-400',
  orange: 'bg-orange-100 dark:bg-orange-900/20 text-orange-600 dark:text-orange-400',
  pink:   'bg-pink-100 dark:bg-pink-900/20 text-pink-600 dark:text-pink-400',
  cyan:   'bg-cyan-100 dark:bg-cyan-900/20 text-cyan-600 dark:text-cyan-400',
};

/**
 * MyCollections — receives state from AppShell so collections
 * persist across page navigations without re-fetching on every mount.
 */
const MyCollections = ({ collections, setCollections, collectionsLoading, fetchCollections }) => {
  const navigate = useNavigate();
  const [selectedCollection, setSelectedCollection] = useState(null);
  const [docToDelete, setDocToDelete] = useState(null);
  const [colToRename, setColToRename] = useState(null);
  const [colToDelete, setColToDelete] = useState(null);
  const [isDeletingCol, setIsDeletingCol] = useState(false);
  const { getAccessToken } = useAuth();

  // Fetch collections on mount — no polling interval needed
  useEffect(() => {
    fetchCollections();
  }, [fetchCollections]);

  // ── Delete single document ────────────────────────────────────────────────
  const handleDeleteDocument = async () => {
    if (!docToDelete) return;

    try {
      const token = getAccessToken();
      const response = await fetch(`http://localhost:8000/documents/${docToDelete.id}`, {
        method: 'DELETE',
        headers: { Authorization: `Bearer ${token}` }
      });

      if (response.ok) {
        toast.success('Document deleted successfully');
        // Optimistically update local state — guard col.docs with fallback to []
        setCollections(prev =>
          prev
            .map(col => {
              const filteredDocs = (col.docs || []).filter(d => d.id !== docToDelete.id);
              return { ...col, docs: filteredDocs, documents: filteredDocs.length };
            })
            .filter(col => col.documents > 0)
        );
        if (selectedCollection) {
          const updatedDocs = (selectedCollection.docs || []).filter(d => d.id !== docToDelete.id);
          if (updatedDocs.length === 0) {
            setSelectedCollection(null);
          } else {
            setSelectedCollection(prev => ({
              ...prev,
              docs: updatedDocs,
              documents: updatedDocs.length,
            }));
          }
        }
        // Refresh from server after a short delay and broadcast update event
        setTimeout(() => {
          fetchCollections();
          window.dispatchEvent(new CustomEvent('documind:collections_updated'));
          try {
            const ch = new BroadcastChannel('documind_events');
            ch.postMessage({ type: 'COLLECTIONS_UPDATED' });
            ch.close();
          } catch (_) {}
        }, 300);
      } else {
        const errJson = await response.json().catch(() => ({}));
        toast.error(errJson.detail || 'Failed to delete document');
      }
    } catch (error) {
      console.error('Delete error:', error);
      toast.error('An error occurred while deleting: ' + error.message);
    } finally {
      setDocToDelete(null);
    }
  };

  // ── Rename collection ─────────────────────────────────────────────────────
  const handleRenameCollection = () => {
    if (!colToRename || !colToRename.newName.trim()) return;
    const trimmed = colToRename.newName.trim();

    setCollections(prev =>
      prev.map(c => (c.id === colToRename.id ? { ...c, name: trimmed } : c))
    );
    toast.success(`Collection renamed to "${trimmed}"`);
    setColToRename(null);
  };

  // ── Delete entire collection ──────────────────────────────────────────────
  const handleDeleteCollection = async () => {
    if (!colToDelete) return;
    setIsDeletingCol(true);

    try {
      const token = getAccessToken();
      const docs = colToDelete.docs || [];
      
      // Delete documents in parallel
      await Promise.all(
        docs.map(doc =>
          fetch(`http://localhost:8000/documents/${doc.id}`, {
            method: 'DELETE',
            headers: { Authorization: `Bearer ${token}` },
          }).catch(err => console.error(`Failed to delete doc ${doc.id}:`, err))
        )
      );

      // Optimistically remove collection from local state
      setCollections(prev => prev.filter(c => c.id !== colToDelete.id));
      if (selectedCollection?.id === colToDelete.id) {
        setSelectedCollection(null);
      }

      toast.success(`Collection "${colToDelete.name}" deleted successfully`);

      // Refresh from server & notify channels
      setTimeout(() => {
        fetchCollections();
        window.dispatchEvent(new CustomEvent('documind:collections_updated'));
        try {
          const ch = new BroadcastChannel('documind_events');
          ch.postMessage({ type: 'COLLECTIONS_UPDATED' });
          ch.close();
        } catch (_) {}
      }, 400);
    } catch (err) {
      toast.error('Failed to delete collection: ' + err.message);
    } finally {
      setIsDeletingCol(false);
      setColToDelete(null);
    }
  };

  const handleViewDocument = async (doc) => {
    try {
      const token = getAccessToken();
      if (!token) { toast.error("You are not logged in"); return; }

      const response = await fetch(`http://localhost:8000/documents/view/${doc.id}`, {
        headers: { 'Authorization': `Bearer ${token}` }
      });

      if (!response.ok) throw new Error('Failed to fetch decrypted document');

      const blob = await response.blob();
      const url = URL.createObjectURL(blob);
      const newWindow = window.open(url, '_blank');
      if (!newWindow) toast.error("Popup blocked! Please allow popups for this site.");
    } catch (error) {
      console.error('View error:', error);
      toast.error('Failed to open document: ' + error.message);
    }
  };

  return (
    <div className="px-8 py-12" data-testid="my-collections">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.3 }}
      >
        {/* Header */}
        <div className="mb-12 flex flex-col sm:flex-row sm:items-center justify-between gap-4">
          <div>
            <h1 className="text-4xl md:text-5xl font-bold text-slate-900 dark:text-white tracking-tight mb-3">
              My Collections
            </h1>
            <p className="text-lg text-slate-600 dark:text-slate-300 flex items-center gap-2">
              AI-categorized documents organized automatically
              {collectionsLoading && collections.length > 0 && (
                <Loader2 className="w-4 h-4 animate-spin text-slate-400" />
              )}
            </p>
          </div>

          <div className="flex items-center gap-3">
            <Button
              variant="outline"
              onClick={() => navigate('/upload')}
              className="border-slate-200 dark:border-slate-800 hover:bg-slate-100 dark:hover:bg-slate-800"
            >
              <Upload className="w-4 h-4 mr-2" />
              Upload Files
            </Button>
            <Button
              className="bg-gradient-to-r from-slate-900 to-slate-800 hover:from-slate-800 hover:to-slate-700 text-white dark:from-slate-100 dark:to-slate-200 dark:text-slate-900 dark:hover:from-slate-200 dark:hover:to-slate-300 shadow-sm"
              data-testid="new-collection-button"
              onClick={fetchCollections}
              disabled={collectionsLoading}
            >
              {collectionsLoading
                ? <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                : <FolderOpen className="w-4 h-4 mr-2" />
              }
              Refresh
            </Button>
          </div>
        </div>

        {/* First-time loading spinner (no cached data yet) */}
        {collectionsLoading && collections.length === 0 && (
          <div className="flex items-center justify-center py-20">
            <Loader2 className="w-10 h-10 animate-spin text-slate-400" />
          </div>
        )}

        {/* Empty State with Upload Action Link */}
        {!collectionsLoading && collections.length === 0 && (
          <div className="flex flex-col items-center justify-center py-20 border-2 border-dashed border-slate-200 dark:border-slate-800 rounded-2xl bg-white/40 dark:bg-slate-900/40 text-center p-6">
            <div className="p-4 rounded-2xl bg-slate-100 dark:bg-slate-800 text-slate-400 mb-4">
              <FolderOpen className="w-10 h-10" strokeWidth={1.5} />
            </div>
            <h3 className="text-lg font-semibold text-slate-800 dark:text-slate-200 mb-1">
              No documents processed yet
            </h3>
            <p className="text-slate-500 dark:text-slate-400 text-sm max-w-sm mb-6">
              Upload documents (PDFs, images) to see them categorized automatically into intelligent collections.
            </p>
            <Button
              onClick={() => navigate('/upload')}
              className="bg-gradient-to-r from-cyan-600 to-blue-600 hover:from-cyan-500 hover:to-blue-500 text-white shadow-md"
            >
              <Upload className="w-4 h-4 mr-2" />
              Upload Your First Document
            </Button>
          </div>
        )}

        {/* Collections Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {collections.map((collection, index) => (
            <motion.div
              key={collection.id}
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ duration: 0.3, delay: index * 0.05 }}
            >
              <Card
                className="hover:shadow-lg transition-all duration-200 border-slate-200 dark:border-slate-800 group cursor-pointer"
                onClick={() => setSelectedCollection(collection)}
              >
                <CardContent className="p-6">
                  <div className="flex items-start justify-between mb-4">
                    <div className={`p-3 rounded-lg ${colorMap[collection.color]}`}>
                      <FolderOpen className="w-6 h-6" strokeWidth={1.5} />
                    </div>
                    {/* Fixed three-dot dropdown menu */}
                    <DropdownMenu>
                      <DropdownMenuTrigger asChild>
                        <Button
                          variant="ghost"
                          size="icon"
                          onClick={(e) => e.stopPropagation()}
                          className="opacity-0 group-hover:opacity-100 transition-opacity"
                        >
                          <MoreVertical className="w-5 h-5" />
                        </Button>
                      </DropdownMenuTrigger>
                      <DropdownMenuContent
                        align="end"
                        onClick={(e) => e.stopPropagation()}
                      >
                        <DropdownMenuItem
                          onClick={(e) => {
                            e.stopPropagation();
                            setColToRename({ id: collection.id, newName: collection.name });
                          }}
                          className="cursor-pointer gap-2"
                        >
                          <Edit className="w-4 h-4" /> Rename
                        </DropdownMenuItem>
                        <DropdownMenuItem
                          onClick={(e) => {
                            e.stopPropagation();
                            setColToDelete(collection);
                          }}
                          className="cursor-pointer gap-2 text-red-600 focus:text-red-600 dark:text-red-400"
                        >
                          <Trash2 className="w-4 h-4" /> Delete
                        </DropdownMenuItem>
                      </DropdownMenuContent>
                    </DropdownMenu>
                  </div>
                  <h3 className="text-xl font-semibold text-slate-900 dark:text-white mb-2">
                    {collection.name}
                  </h3>
                  <div className="flex items-center gap-3">
                    <Badge variant="outline" className="bg-slate-50 dark:bg-slate-800">
                      <FileText className="w-3 h-3 mr-1" />
                      {collection.documents} docs
                    </Badge>
                  </div>
                </CardContent>
              </Card>
            </motion.div>
          ))}
        </div>

        {/* Document List Dialog */}
        <Dialog open={!!selectedCollection} onOpenChange={(open) => !open && setSelectedCollection(null)}>
          <DialogContent className="max-w-2xl max-h-[80vh] overflow-hidden flex flex-col">
            <DialogHeader>
              <DialogTitle className="text-2xl flex items-center gap-3">
                <div className={`p-2 rounded-lg ${selectedCollection ? colorMap[selectedCollection.color] : ''}`}>
                  <FolderOpen className="w-5 h-5" />
                </div>
                {selectedCollection?.name}
              </DialogTitle>
            </DialogHeader>

            <div className="flex-1 overflow-y-auto mt-4 pr-2">
              <div className="space-y-3">
                {Array.from(
                  new Map(
                    (selectedCollection?.docs || []).map((d) => [
                      (d.name || d.id).toLowerCase().trim(),
                      d,
                    ])
                  ).values()
                ).map((doc) => (
                  <div
                    key={doc.id}
                    className="flex items-center justify-between p-4 rounded-xl border border-slate-100 dark:border-slate-800 hover:bg-slate-50 dark:hover:bg-slate-900 transition-colors group"
                  >
                    <div className="flex items-center gap-3">
                      <div className="p-2 rounded-lg bg-slate-100 dark:bg-slate-800 text-slate-500">
                        <FileText className="w-5 h-5" />
                      </div>
                      <div>
                        <p className="font-medium text-slate-900 dark:text-white truncate max-w-[300px]">
                          {doc.name}
                        </p>
                      </div>
                    </div>
                    <div className="flex items-center gap-2">
                      <Button
                        variant="outline"
                        size="sm"
                        className="opacity-0 group-hover:opacity-100 transition-opacity"
                        onClick={() => handleViewDocument(doc)}
                      >
                        <ExternalLink className="w-4 h-4 mr-2" />
                        Open PDF
                      </Button>
                      <Button
                        variant="ghost"
                        size="sm"
                        className="opacity-0 group-hover:opacity-100 transition-opacity text-red-500 hover:text-red-600 hover:bg-red-50 dark:hover:bg-red-900/20"
                        onClick={(e) => {
                          e.stopPropagation();
                          setDocToDelete(doc);
                        }}
                      >
                        <Trash2 className="w-4 h-4" />
                      </Button>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </DialogContent>
        </Dialog>

        {/* Rename Collection Dialog */}
        <Dialog open={!!colToRename} onOpenChange={(open) => !open && setColToRename(null)}>
          <DialogContent className="max-w-md">
            <DialogHeader>
              <DialogTitle>Rename Collection</DialogTitle>
            </DialogHeader>
            <div className="py-4">
              <Input
                value={colToRename?.newName || ''}
                onChange={(e) =>
                  setColToRename(prev => prev ? { ...prev, newName: e.target.value } : null)
                }
                placeholder="Enter new collection name"
                onKeyDown={(e) => e.key === 'Enter' && handleRenameCollection()}
                autoFocus
              />
            </div>
            <DialogFooter>
              <Button variant="outline" onClick={() => setColToRename(null)}>
                Cancel
              </Button>
              <Button onClick={handleRenameCollection}>
                Save Name
              </Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>

        {/* Delete Collection Confirmation Dialog */}
        <AlertDialog open={!!colToDelete} onOpenChange={(open) => !open && setColToDelete(null)}>
          <AlertDialogContent>
            <AlertDialogHeader>
              <AlertDialogTitle>Delete Collection?</AlertDialogTitle>
              <AlertDialogDescription>
                This will delete <span className="font-semibold text-slate-900 dark:text-white">"{colToDelete?.name}"</span> and all {colToDelete?.documents || 0} associated document(s) permanently.
              </AlertDialogDescription>
            </AlertDialogHeader>
            <AlertDialogFooter>
              <AlertDialogCancel disabled={isDeletingCol}>Cancel</AlertDialogCancel>
              <AlertDialogAction
                onClick={handleDeleteCollection}
                disabled={isDeletingCol}
                className="bg-red-600 hover:bg-red-700 text-white"
              >
                {isDeletingCol ? <Loader2 className="w-4 h-4 mr-2 animate-spin" /> : null}
                Delete Collection
              </AlertDialogAction>
            </AlertDialogFooter>
          </AlertDialogContent>
        </AlertDialog>

        {/* Delete Single Document Confirmation Dialog */}
        <AlertDialog open={!!docToDelete} onOpenChange={(open) => !open && setDocToDelete(null)}>
          <AlertDialogContent>
            <AlertDialogHeader>
              <AlertDialogTitle>Are you absolutely sure?</AlertDialogTitle>
              <AlertDialogDescription>
                This will permanently delete <span className="font-semibold text-slate-900 dark:text-white">"{docToDelete?.name}"</span> from your storage and indexed data. This action cannot be undone.
              </AlertDialogDescription>
            </AlertDialogHeader>
            <AlertDialogFooter>
              <AlertDialogCancel>Cancel</AlertDialogCancel>
              <AlertDialogAction
                onClick={handleDeleteDocument}
                className="bg-red-600 hover:bg-red-700 text-white"
              >
                Delete
              </AlertDialogAction>
            </AlertDialogFooter>
          </AlertDialogContent>
        </AlertDialog>
      </motion.div>
    </div>
  );
};

export default MyCollections;
