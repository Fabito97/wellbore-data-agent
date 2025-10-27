// src/features/documents/documentsSlice.ts
import { createSlice, type PayloadAction } from '@reduxjs/toolkit';

export interface Document {
  id: string;
  name: string;
  type: string;
  size: number;
  uploadedAt: number;
  status: 'uploading' | 'processing' | 'indexed' | 'error';
  progress?: number;
  metadata?: {
    wellName?: string;
    reportType?: string;
    pageCount?: number;
  };
}

interface DocumentsState {
  documents: Document[];
  selectedDocumentId: string | null;
  isUploading: boolean;
}

const initialState: DocumentsState = {
  documents: [],
  selectedDocumentId: null,
  isUploading: false,
};

const documentsSlice = createSlice({
  name: 'documents',
  initialState,
  reducers: {
    addDocument: (state, action: PayloadAction<Document>) => {
      state.documents.push(action.payload);
    },
    
    updateDocumentStatus: (
      state,
      action: PayloadAction<{ id: string; status: Document['status']; progress?: number }>
    ) => {
      const doc = state.documents.find(d => d.id === action.payload.id);
      if (doc) {
        doc.status = action.payload.status;
        if (action.payload.progress !== undefined) {
          doc.progress = action.payload.progress;
        }
      }
    },
    
    selectDocument: (state, action: PayloadAction<string | null>) => {
      state.selectedDocumentId = action.payload;
    },
    
    setIsUploading: (state, action: PayloadAction<boolean>) => {
      state.isUploading = action.payload;
    },
    
    removeDocument: (state, action: PayloadAction<string>) => {
      state.documents = state.documents.filter(d => d.id !== action.payload);
    },
  },
});

export const {
  addDocument,
  updateDocumentStatus,
  selectDocument,
  setIsUploading,
  removeDocument,
} = documentsSlice.actions;

export default documentsSlice.reducer;