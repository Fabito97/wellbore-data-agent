// src/features/documents/documentsApi.ts
import { createApi, fetchBaseQuery } from "@reduxjs/toolkit/query/react";
import { BASE_URL } from "./chatApi";

// -----------------------------
// Types EXACTLY matching backend
// -----------------------------

export interface DocumentUploadData {
  document_id: string;
  filename: string;
  status: string;
  page_count: number;
  word_count: number;
  table_count: number;
  chunk_count: number;
  uploaded_at: string;
}

export interface DocumentUploadResponse {
  status: string;
  message: string;
  data: DocumentUploadData;
}

export interface DocumentListItem {
  document_id: string;
  filename: string;
  status: string;
  uploaded_at: string;
  page_count: number;
  word_count: number;
  table_count: number;
  chunk_count: number;
}

export interface DocumentListResponse {
  status: string;
  total: number;
  documents: DocumentListItem[];
}

export interface DocumentContentResponse {
  status: string;
  data: any; // FastAPI returns full structured metadata; can refine later
}

export interface DeleteDocumentResponse {
  status: string;
  message: string;
}

// ----------------------------------
// RTK Query API
// ----------------------------------

export const documentsApi = createApi({
  reducerPath: "documentsApi",
  baseQuery: fetchBaseQuery({
    baseUrl: `${BASE_URL}/api/v1`,
  }),

  tagTypes: ["Documents", "Document"],

  endpoints: (builder) => ({
    // ------------------------------------
    // UPLOAD DOCUMENT
    // ------------------------------------
    uploadDocument: builder.mutation<DocumentUploadResponse, FormData>({
      query: (formData) => ({
        url: "/documents/upload",
        method: "POST",
        body: formData,
      }),
      invalidatesTags: ["Documents"],
    }),

    // ------------------------------------
    // GET ALL DOCUMENTS
    // ------------------------------------
    getDocuments: builder.query<DocumentListResponse, void>({
      query: () => "/documents/",
      providesTags: (result) =>
        result?.documents
          ? [
              ...result.documents.map((doc) => ({
                type: "Document" as const,
                id: doc.document_id,
              })),
              { type: "Documents", id: "LIST" },
            ]
          : [{ type: "Documents", id: "LIST" }],
    }),

    // ------------------------------------
    // GET DOCUMENT METADATA
    // ------------------------------------
    getDocumentById: builder.query<DocumentContentResponse, string>({
      query: (documentId) => `/documents/${documentId}`,
      providesTags: (result, error, id) => [{ type: "Document", id }],
    }),

    // ------------------------------------
    // DELETE DOCUMENT
    // ------------------------------------
    deleteDocument: builder.mutation<DeleteDocumentResponse, string>({
      query: (documentId) => ({
        url: `/documents/${documentId}`,
        method: "DELETE",
      }),
      invalidatesTags: (result, error, id) => [
        { type: "Document", id },
        { type: "Documents", id: "LIST" },
      ],
    }),
  }),
});

// Hooks
export const {
  useUploadDocumentMutation,
  useGetDocumentsQuery,
  useGetDocumentByIdQuery,
  useDeleteDocumentMutation,
} = documentsApi;
